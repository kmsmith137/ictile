// CPU kernel for ictile: fused (generate Gaussian random field modes) + (regrid kx -> k'x)
// + (regrid ky -> k'y). Same algorithm and output as the CUDA kernel (see grf_cuda.cu), except
// that each work item covers all k'y outputs, and data is stored as separate real/imaginary
// arrays so that the regridding loops vectorize.
//
// Work items are (kz, segment of k'x outputs), distributed over the XLA CPU thread pool. Each work
// item streams over kx rows: it generates a row of modes (all ky "columns" needed), scatters the row
// into a ring buffer of Kx accumulator rows (one per k'x), and when an accumulator row is complete,
// regrids it ky -> k'y and writes it to the output.

#include <algorithm>
#include <atomic>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "xla/ffi/api/ffi.h"
#include "grf_common.h"

namespace ffi = xla::ffi;

namespace ictile {
namespace {

// Mapping between ambient index n and bounding-box index o along one axis (see _tables.py and
// AxisMap in grf_cuda.cu).
struct CpuAxis {
    int M, A, P, K, dmin;
    const int32_t *so;   // [A]

    int window_start(int o) const { return (o / A) * P + so[o % A]; }   // o >= 0
};

inline int mod_index(int n, int N)
{
    int r = n % N;
    return (r < 0) ? (r + N) : r;
}


template <typename Real>
struct CpuProblem {
    using U = typename Fixed<Real>::U;

    ModeGenerator<Real> gen;
    int N, Bz, kz0;
    CpuAxis x, y;
    const int32_t *pf_x;    // [Ax]
    const Real *ws_x;       // [Px][Kx]
    const Real *gw_y;       // [Ay][LWy]
    int Ly, LWy;
    int seg_len, nseg;
    int ys, ncols;          // columns (unwrapped ky) [ys, ys + ncols), with ys and ncols even
    const U *phx, *phy, *phz;
    std::complex<Real> *out;   // [Bz][Mx][My]

    void init_columns()
    {
        ys = y.window_start(0) & ~1;
        ncols = y.window_start(y.M - 1 + y.K) - ys;
        ncols += (ncols & 1);
    }

    int row_len() const { return ncols + Ly; }

    // Per-worker scratch, in units of Real. Must agree with h2_ffi() in _ffi.py.
    size_t scratch_size() const { return 2 * (size_t(ncols) + size_t(x.K) * row_len()); }

    void run_item(int item, Real *scratch) const;
};


template <typename Real>
void CpuProblem<Real>::run_item(int item, Real *scratch) const
{
    const int b = item / nseg;
    const int seg = item - b * nseg;
    const int kz = kz0 + b;
    const int X0 = seg * seg_len;
    const int X1 = std::min(X0 + seg_len, x.M);
    const int My = y.M, Kx = x.K, R = row_len();
    std::complex<Real> *outb = out + size_t(b) * x.M * My;

    if (kz >= N/2) {
        std::memset((void *)(outb + size_t(X0) * My), 0, size_t(X1 - X0) * My * sizeof(std::complex<Real>));
        return;
    }

    Real *mre = scratch;                 // row of modes [ncols]
    Real *mim = mre + ncols;
    Real *are = mim + ncols;             // ring buffer of Kx accumulator rows [Kx][R]
    Real *aim = are + size_t(Kx) * R;    // (columns >= ncols are zero padding)
    std::memset(are, 0, 2 * size_t(Kx) * R * sizeof(Real));
    int head = 0;

    // Rows (unwrapped kx): groups q in [q, q_end], where all rows n in group q have
    // floor(n*Ax/Px) = q. After group q, output k'x = q + dmin_x is complete.
    int q = X0 - x.dmin - Kx + 1;
    const int q_end = X1 - 1 - x.dmin;
    int kx = mod_index(x.window_start(X0), N);
    int p = pf_x[X0 % x.A];
    const int ki_start = mod_index(ys, N);
    const U phz_kz = phz[kz];

    for (;;) {
        const int nx = signed_wavenumber(kx, N);
        if (nx != -N/2) {
            // Generate the row of modes.
            const U ph_row = phx[kx] + phz_kz;
            int ki0 = ki_start;
            for (int c = 0; c < ncols; c += 2) {
                const int ny0 = signed_wavenumber(ki0, N);
                const int ny1 = signed_wavenumber(ki0 + 1, N);
                const U ph0 = ph_row + phy[ki0];
                const U ph1 = ph_row + phy[ki0 + 1];
                Real v[4];
                if (kz > 0)
                    gen.pair(nx, ny0, ny1, kz, ph0, ph1, v);
                else {
                    gen.single_kz0(nx, ny0, ph0, v[0], v[1]);
                    gen.single_kz0(nx, ny1, ph1, v[2], v[3]);
                }
                mre[c] = v[0];
                mim[c] = v[1];
                mre[c+1] = v[2];
                mim[c+1] = v[3];
                ki0 = (ki0 + 2 == N) ? 0 : (ki0 + 2);
            }

            // Regrid kx -> k'x (scatter into the ring buffer).
            for (int k = 0; k < Kx; k++) {
                const Real w = ws_x[p * Kx + k];
                const int slot = (head + k) % Kx;
                Real *dre = are + size_t(slot) * R;
                Real *dim = aim + size_t(slot) * R;
#pragma omp simd
                for (int c = 0; c < ncols; c++) {
                    dre[c] += w * mre[c];
                    dim[c] += w * mim[c];
                }
            }
        }

        kx = (kx == N - 1) ? 0 : (kx + 1);
        p += x.A;
        if (p < x.P)
            continue;

        // Group q is complete, so output k'x = q + dmin_x is complete. Regrid ky -> k'y.
        p -= x.P;
        const int o = q + x.dmin;
        Real *sre = are + size_t(head) * R;
        Real *sim = aim + size_t(head) * R;
        if (o >= X0) {
            std::complex<Real> *dst = outb + size_t(o) * My;
            for (int i = 0; i < My; i++) {
                const int r = i % y.A;
                const Real *src_re = sre + (y.window_start(i) - ys);
                const Real *src_im = sim + (y.window_start(i) - ys);
                const Real *w = gw_y + size_t(r) * LWy;
                Real hr = 0, hi = 0;
#pragma omp simd reduction(+:hr,hi)
                for (int j = 0; j < Ly; j++) {
                    hr += w[j] * src_re[j];
                    hi += w[j] * src_im[j];
                }
                dst[i] = std::complex<Real>(hr, hi);
            }
        }
        std::fill(sre, sre + ncols, Real(0));
        std::fill(sim, sim + ncols, Real(0));
        head = (head + 1) % Kx;

        if (++q > q_end)
            break;
    }
}


// Runs fn(item, worker) for item in [0, nitems), on the XLA thread pool plus the calling thread.
// The calling thread participates, so this can't deadlock if the pool is busy.
template <typename F>
void parallel_for(ffi::ThreadPool &pool, int nitems, int nworkers, F fn)
{
    struct State {
        std::atomic<int> next_item{0};
        std::atomic<int> next_worker{0};
        std::mutex mutex;
        std::condition_variable cv;
        int ndone = 0;
    };
    auto state = std::make_shared<State>();

    auto work = [state, nitems, nworkers, &fn]() {
        int worker = state->next_worker.fetch_add(1);
        if (worker >= nworkers)
            return;
        int n = 0;
        for (;;) {
            int item = state->next_item.fetch_add(1);
            if (item >= nitems)
                break;
            fn(item, worker);
            n++;
        }
        if (n > 0) {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->ndone += n;
            state->cv.notify_all();
        }
    };

    // Helper tasks capture 'state' by value, and 'fn' by reference. A helper which starts after
    // all items are claimed returns without touching 'fn', so the reference never dangles.
    for (int i = 1; i < nworkers; i++) {
        auto task = work;
        pool.Schedule(std::move(task));
    }
    work();

    std::unique_lock<std::mutex> lock(state->mutex);
    state->cv.wait(lock, [&]() { return state->ndone == nitems; });
}


template <typename Real, ffi::DataType RT, ffi::DataType UT, ffi::DataType CT>
ffi::Error h2_cpu(ffi::ThreadPool pool,
                  ffi::Buffer<ffi::S32> kz0, ffi::Buffer<RT> sqrtp, ffi::Buffer<RT> ws_x,
                  ffi::Buffer<ffi::S32> so_x, ffi::Buffer<ffi::S32> pf_x,
                  ffi::Buffer<ffi::S32> so_y, ffi::Buffer<RT> gw_y,
                  ffi::Buffer<UT> phx, ffi::Buffer<UT> phy, ffi::Buffer<UT> phz,
                  ffi::Result<ffi::Buffer<CT>> out, ffi::Result<ffi::Buffer<RT>> workspace,
                  int64_t nside, int64_t dmin_x, int64_t dmin_y, int64_t py, int64_t ky, int64_t ly,
                  int64_t key0, int64_t key1, int64_t stream_id, int64_t fixed_amplitude, double tab_scale,
                  int64_t deriv)
{
    auto odims = out->dimensions();
    auto xdims = ws_x.dimensions();
    auto gdims = gw_y.dimensions();
    if ((odims.size() != 3) || (xdims.size() != 2) || (gdims.size() != 2))
        return ffi::Error::InvalidArgument("ictile: bad buffer shapes");

    CpuProblem<Real> prob;
    prob.N = int(nside);
    prob.Bz = int(odims[0]);
    prob.kz0 = kz0.typed_data()[0];
    prob.x = CpuAxis{int(odims[1]), int(so_x.element_count()), int(xdims[0]), int(xdims[1]), int(dmin_x), so_x.typed_data()};
    prob.y = CpuAxis{int(odims[2]), int(so_y.element_count()), int(py), int(ky), int(dmin_y), so_y.typed_data()};
    prob.pf_x = pf_x.typed_data();
    prob.Ly = int(ly);
    prob.LWy = int(gdims[1]);

    auto bad_axis = [&](const CpuAxis &ax) {
        return (ax.M < 1) || (ax.A < 1) || (ax.A > ax.P) || (ax.K < 1) || (int64_t(ax.M) * ax.P != int64_t(prob.N) * ax.A);
    };
    if ((prob.N % 2) || bad_axis(prob.x) || bad_axis(prob.y) || (phx.element_count() != size_t(prob.N)) ||
        (pf_x.element_count() != so_x.element_count()) || (int(gdims[0]) != prob.y.A) || (prob.LWy < prob.Ly))
        return ffi::Error::InvalidArgument("ictile: inconsistent nside/ntile/tables");
    if ((deriv < -1) || (deriv > 2))
        return ffi::Error::InvalidArgument("ictile: deriv must be -1, 0, 1, or 2");

    prob.gen.nside = prob.N;
    prob.gen.key0 = uint32_t(key0);
    prob.gen.key1 = uint32_t(key1);
    prob.gen.stream = uint32_t(stream_id);
    prob.gen.sqrtp = sqrtp.typed_data();
    prob.gen.ntab = int(sqrtp.element_count());
    prob.gen.tab_scale = Real(tab_scale);
    prob.gen.fixed_amplitude = int(fixed_amplitude);
    prob.gen.deriv = int(deriv);
    prob.ws_x = ws_x.typed_data();
    prob.gw_y = gw_y.typed_data();
    prob.phx = phx.typed_data();
    prob.phy = phy.typed_data();
    prob.phz = phz.typed_data();
    prob.out = reinterpret_cast<std::complex<Real> *>(out->untyped_data());
    prob.init_columns();

    // Split each kz plane into segments of k'x, so that there are enough work items for all
    // threads, but segments are long compared to the regridding halo (Kx-1).
    const int nthreads = std::max<int>(1, int(pool.num_threads()) + 1);
    const int Mx = prob.x.M, Kx = prob.x.K;
    int nseg = (2 * nthreads + prob.Bz - 1) / prob.Bz;
    nseg = std::max(1, std::min(nseg, Mx / (4 * Kx)));
    prob.seg_len = (Mx + nseg - 1) / nseg;
    prob.nseg = (Mx + prob.seg_len - 1) / prob.seg_len;

    // Per-thread scratch memory is a JAX-allocated "workspace" output (see _ffi.py).
    const int nitems = prob.Bz * prob.nseg;
    const size_t per_worker = prob.scratch_size();
    const int nworkers = std::min({nthreads, nitems, int(workspace->element_count() / per_worker)});
    if (nworkers < 1)
        return ffi::Error::InvalidArgument("ictile: CPU workspace buffer is too small");
    Real *sbuf = workspace->typed_data();

    parallel_for(pool, nitems, nworkers, [&](int item, int worker) {
        prob.run_item(item, sbuf + size_t(worker) * per_worker);
    });

    return ffi::Error::Success();
}


#define ICTILE_H2_ARGS(RT, UT, CT)                                                                   \
    ffi::Buffer<ffi::S32> kz0, ffi::Buffer<RT> sqrtp, ffi::Buffer<RT> ws_x,                           \
    ffi::Buffer<ffi::S32> so_x, ffi::Buffer<ffi::S32> pf_x, ffi::Buffer<ffi::S32> so_y,              \
    ffi::Buffer<RT> gw_y, ffi::Buffer<UT> phx, ffi::Buffer<UT> phy, ffi::Buffer<UT> phz,              \
    ffi::Result<ffi::Buffer<CT>> out, ffi::Result<ffi::Buffer<RT>> workspace,                        \
    int64_t nside, int64_t dmin_x, int64_t dmin_y, int64_t py, int64_t ky, int64_t ly,               \
    int64_t key0, int64_t key1, int64_t stream_id, int64_t fixed_amplitude, double tab_scale,       \
    int64_t deriv

#define ICTILE_H2_FORWARD                                                                            \
    kz0, sqrtp, ws_x, so_x, pf_x, so_y, gw_y, phx, phy, phz, out, workspace,                         \
    nside, dmin_x, dmin_y, py, ky, ly, key0, key1, stream_id, fixed_amplitude, tab_scale, deriv

ffi::Error h2_cpu_f32(ffi::ThreadPool pool, ICTILE_H2_ARGS(ffi::F32, ffi::U32, ffi::C64))
{
    return h2_cpu<float, ffi::F32, ffi::U32, ffi::C64>(pool, ICTILE_H2_FORWARD);
}

ffi::Error h2_cpu_f64(ffi::ThreadPool pool, ICTILE_H2_ARGS(ffi::F64, ffi::U64, ffi::C128))
{
    return h2_cpu<double, ffi::F64, ffi::U64, ffi::C128>(pool, ICTILE_H2_FORWARD);
}

}  // anonymous namespace
}  // namespace ictile


#define ICTILE_H2_BINDING(RT, UT, CT)                                   \
    ffi::Ffi::Bind()                                                    \
        .Ctx<ffi::ThreadPool>()                                         \
        .Arg<ffi::Buffer<ffi::S32>>()   /* kz0 */                       \
        .Arg<ffi::Buffer<RT>>()         /* sqrtp table */               \
        .Arg<ffi::Buffer<RT>>()         /* ws_x[Px,Kx] */               \
        .Arg<ffi::Buffer<ffi::S32>>()   /* so_x[Ax] */                  \
        .Arg<ffi::Buffer<ffi::S32>>()   /* pf_x[Ax] */                  \
        .Arg<ffi::Buffer<ffi::S32>>()   /* so_y[Ay] */                  \
        .Arg<ffi::Buffer<RT>>()         /* gw_y[Ay,LWy] */              \
        .Arg<ffi::Buffer<UT>>()         /* phase x */                   \
        .Arg<ffi::Buffer<UT>>()         /* phase y */                   \
        .Arg<ffi::Buffer<UT>>()         /* phase z */                   \
        .Ret<ffi::Buffer<CT>>()         /* H2[Bz,Mx,My] */              \
        .Ret<ffi::Buffer<RT>>()         /* workspace */                 \
        .Attr<int64_t>("nside")                                         \
        .Attr<int64_t>("dmin_x")                                        \
        .Attr<int64_t>("dmin_y")                                        \
        .Attr<int64_t>("py")                                            \
        .Attr<int64_t>("ky")                                            \
        .Attr<int64_t>("ly")                                            \
        .Attr<int64_t>("key0")                                          \
        .Attr<int64_t>("key1")                                          \
        .Attr<int64_t>("stream")                                        \
        .Attr<int64_t>("fixed_amplitude")                               \
        .Attr<double>("tab_scale")                                      \
        .Attr<int64_t>("deriv")

XLA_FFI_DEFINE_HANDLER_SYMBOL(IctileH2F32Cpu, ictile::h2_cpu_f32,
                              ICTILE_H2_BINDING(ffi::F32, ffi::U32, ffi::C64));

XLA_FFI_DEFINE_HANDLER_SYMBOL(IctileH2F64Cpu, ictile::h2_cpu_f64,
                              ICTILE_H2_BINDING(ffi::F64, ffi::U64, ffi::C128));
