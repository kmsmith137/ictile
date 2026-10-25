// CUDA kernel for ictile: fused (generate Gaussian random field modes) + (regrid kx -> k'x)
// + (regrid ky -> k'y), i.e. the first part of step 2(a) of tiled-FFT algorithm 2 in the notes.
//
// Output H2[b, k'x, k'y] for a batch of kz values (kz = kz0 + b), on the (Mx, My) bounding-box
// wavenumber grid, where (Mx, My) = 2*(ntile_x, ntile_y). The tile phase exp(i k.R_c) is included,
// via the fixed-point phase tables. M need not divide nside: along each axis, ambient index n sits at
// position n*M/nside = Q + p/P on the bounding-box grid (see _tables.py).
//
// Each thread block computes a tile of H2: outputs k'x in [X0, X1) (a "segment"), and k'y in
// [Y0, Y0+BY) (a "strip"). It needs ambient ky in a range of "columns" [ys, ye), and ambient kx in a
// range of "rows". The block processes its columns in chunks of 2*T, where T = blockDim.x, and
// alternates between two stages:
//
//   Stage 1 (regrid kx -> k'x): each thread owns two adjacent columns (one Philox call generates a
//   pair of modes adjacent in ky). The block streams over rows: each thread generates its two modes
//   in registers, and scatters them into a sliding window of K accumulators (one per k'x which
//   receives a contribution). When an accumulator is complete, it is written to a shared memory
//   buffer of G rows of k'x (a transpose).
//
//   Stage 2 (regrid ky -> k'y): once G rows of k'x are complete, each warp computes outputs k'y.
//   Within a warp, all lanes gather from the same window of ky, with the same weights, but
//   different k'x rows. Thus there are no shared memory bank conflicts or divergence, even though the
//   window offsets and weights depend on k'y in an irregular way when M doesn't divide nside.
//
// In stage 1, all threads see the same kx at the same time, so the irregular structure along x
// (which rows contribute to which outputs) is uniform across the block.

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"
#include "grf_common.h"

namespace ffi = xla::ffi;

// Compute capabilities this library was built for, e.g. 80_86_89 (set by setup.py; one preprocessor token).
#ifndef ICTILE_CUDA_ARCHS
#define ICTILE_CUDA_ARCHS unknown
#endif
#define ICTILE_STR2(x) #x
#define ICTILE_STR(x) ICTILE_STR2(x)

namespace ictile {
namespace {

// Error message for a CUDA error, with the GPU's compute capability and the architectures this library was
// built for (a mismatch gives "no kernel image is available for execution on the device").
std::string cuda_error(const char *what, cudaError_t err)
{
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    std::string archs = ICTILE_STR(ICTILE_CUDA_ARCHS);
    std::replace(archs.begin(), archs.end(), '_', ';');
    std::string cc = std::to_string(major) + std::to_string(minor);
    std::string msg = std::string("ictile: ") + what + ": " + cudaGetErrorString(err) + " (this GPU has compute capability "
        + std::to_string(major) + "." + std::to_string(minor) + ", and ictile's CUDA kernel was built for " + archs + ")";
    if ((err == cudaErrorNoKernelImageForDevice) || (err == cudaErrorInvalidDeviceFunction))
        msg += ". Rebuild ictile with ICTILE_CUDA_ARCH=" + cc + ", or a list that includes " + cc + ".";
    return msg;
}

template <typename Real> struct Cplx;
template <> struct Cplx<float> { using T = float2; };
template <> struct Cplx<double> { using T = double2; };

// Per-precision tuning: G = number of k'x rows in the transpose buffer, so that NG = 32/G lanes
// share each row in stage 2. (Shared memory serves 8-byte loads 16 lanes at a time, and 16-byte
// loads 8 lanes at a time, so stage 2 is conflict-free if the row stride is NG mod 16.)
// TMAX = maximum threads per block.
template <typename Real> struct Tuning;
template <> struct Tuning<float> { static constexpr int G = 8, TMAX = 512; };
template <> struct Tuning<double> { static constexpr int G = 4, TMAX = 256; };

constexpr int kMaxStrip = 256;   // max k'y outputs per thread block


// Mapping between ambient index n and bounding-box index o along one axis (see _tables.py).
// Output o receives contributions from the ambient "window" [window_start(o), window_start(o+K)).
struct AxisMap {
    int M;          // bounding-box size
    int A, P;       // M/nside = A/P, in lowest terms
    int K;          // number of regridding taps per ambient index
    int dmin;       // ambient n (with n*A = Q*P + p) contributes to o = Q + dmin + k, for 0 <= k < K
    FastDiv divA;
    const int32_t *so;   // [A] window_start(r), for 0 <= r < A

    // Device version (o >= 0), using the table.
    __device__ __forceinline__ int window_start(int o) const
    {
        int m = divA.div(o);
        return m * P + __ldg(so + (o - m * A));
    }

    // Host version, from the formula window_start(o) = ceil((o - dmin - K + 1) * P / A).
    int64_t window_start_host(int64_t o) const
    {
        int64_t a = (o - dmin - K + 1) * int64_t(P);
        return (a >= 0) ? ((a + A - 1) / A) : -((-a) / A);
    }
};

struct Geometry {
    int nside;
    AxisMap x;               // streamed over rows (stage 1)
    AxisMap y;               // gathered (stage 2)
    const int32_t *pf_x;     // [Ax] phase index p of ambient index window_start(r)
    int Ly;                  // max window length along y
    int LWy;                 // row stride of the y gather table (>= Ly + NG, zero-padded)
    int BY;                  // k'y outputs per thread block
    int seg_len;             // k'x outputs per thread block
    int nch;                 // number of column chunks
};


// Stores two adjacent complex values (16-byte aligned in float32).
__device__ __forceinline__ void store2(float2 *dst, float a, float b, float c, float d)
{
    *reinterpret_cast<float4 *>(dst) = make_float4(a, b, c, d);
}

__device__ __forceinline__ void store2(double2 *dst, double a, double b, double c, double d)
{
    dst[0] = make_double2(a, b);
    dst[1] = make_double2(c, d);
}


__device__ __forceinline__ int mod_index(int n, int N)
{
    int r = n % N;
    return (r < 0) ? (r + N) : r;
}


template <typename Real, int KMAX>
__global__ void __launch_bounds__(Tuning<Real>::TMAX)
h2_kernel(ModeGenerator<Real> gen, Geometry g, const int32_t *__restrict__ kz0_ptr,
          const Real *__restrict__ ws_x, const Real *__restrict__ gw_y,
          const typename Fixed<Real>::U *__restrict__ phx,
          const typename Fixed<Real>::U *__restrict__ phy,
          const typename Fixed<Real>::U *__restrict__ phz,
          typename Cplx<Real>::T *__restrict__ out)
{
    using U = typename Fixed<Real>::U;
    using C = typename Cplx<Real>::T;
    constexpr int G = Tuning<Real>::G;
    constexpr int NG = 32 / G;

    const int N = g.nside, h = N / 2;
    const int T = blockDim.x;
    const int W = 2 * T + NG;     // row stride of the transpose buffer (= NG mod 16)
    const int tid = threadIdx.x;
    const int Mx = g.x.M, My = g.y.M;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    C *buf = reinterpret_cast<C *>(smem_raw);   // [G][W], columns >= 2*T are zero padding
    C *sout = buf + G * W;                        // [G][BY]

    const int Y0 = blockIdx.x * g.BY;
    const int nby = min(g.BY, My - Y0);
    const int X0 = blockIdx.y * g.seg_len;
    const int X1 = min(X0 + g.seg_len, Mx);
    const int kz = *kz0_ptr + int(blockIdx.z);
    C *outb = out + size_t(blockIdx.z) * Mx * My + Y0;

    if (kz >= h) {
        // Padding at the end of the last kz batch.
        for (int o = X0; o < X1; o++)
            for (int i = tid; i < nby; i += T)
                outb[size_t(o) * My + i] = C{Real(0), Real(0)};
        return;
    }

    for (int j = tid; j < G * W; j += T)
        buf[j] = C{Real(0), Real(0)};
    __syncthreads();

    // Columns (unwrapped ky) needed by the strip. Start at even ky, since Philox pairs are (2j, 2j+1).
    const int ys = g.y.window_start(Y0) & ~1;
    const int ye = g.y.window_start(Y0 + nby - 1 + g.y.K);

    // Rows (unwrapped kx): groups q in [q_start, q_end], where all rows n in group q have
    // floor(n*Ax/Px) = q. After group q, output k'x = q + dmin_x is complete.
    const int Ax = g.x.A, Px = g.x.P, Kx = g.x.K;
    const int q_start = X0 - g.x.dmin - Kx + 1;
    const int q_end = X1 - 1 - g.x.dmin;
    const int n_start = g.x.window_start(X0);
    const int p_start = __ldg(g.pf_x + (X0 - g.x.divA.div(X0) * Ax));
    const int kx_start = mod_index(n_start, N);
    const U phz_kz = phz[kz];

    const int warp = tid >> 5, nwarps = T >> 5;
    const int line = (tid & 31) / NG, lq = tid & (NG - 1);

    // Number of rows: the last row is the last one in group q_end.
    const int nrows = g.x.window_start(X1 + Kx - 1) - n_start;

    for (int ch = 0; ch < g.nch; ch++) {
        const int col_lo = ys + ch * 2 * T;
        const int c0 = col_lo + 2 * tid;
        const bool active = (c0 < ye);
        const int ki0 = mod_index(c0, N);
        const int ny0 = signed_wavenumber(ki0, N);
        const int ny1 = signed_wavenumber(ki0 + 1, N);
        const U phy0 = phy[ki0];
        const U phy1 = phy[ki0 + 1];

        Real a0r[KMAX], a0i[KMAX], a1r[KMAX], a1i[KMAX];
#pragma unroll
        for (int k = 0; k < KMAX; k++)
            a0r[k] = a0i[k] = a1r[k] = a1i[k] = Real(0);

        int p = p_start, q = q_start;
        int nl = 0, o_first = X0;

        // Stage 2: regrid ky -> k'y, for k'x rows [o_first, o_first + nl) in the transpose buffer.
        auto flush = [&]() {
            __syncthreads();

            for (int i = warp; i < nby; i += nwarps) {
                const int o = Y0 + i;
                const int m = g.y.divA.div(o);
                const int r = o - m * g.y.A;
                const int s = m * g.y.P + __ldg(g.y.so + r) - col_lo;   // window start, in buffer columns
                const int jlo = max(0, -s);
                const int jhi = min(g.Ly, 2 * T - s);
                const int jend = jlo + ((jhi - jlo + NG - 1) / NG) * NG;
                const C *src = buf + line * W + s;
                const Real *wr = gw_y + size_t(r) * g.LWy;

                Real hr = Real(0), hi = Real(0);
                for (int j = jlo + lq; j < jend; j += NG) {
                    const Real wj = __ldg(wr + j);   // zero for j >= window length
                    const C v = src[j];               // zero padding for columns >= 2*T
                    hr += wj * v.x;
                    hi += wj * v.y;
                }
#pragma unroll
                for (int off = NG / 2; off > 0; off >>= 1) {
                    hr += __shfl_xor_sync(0xffffffffu, hr, off);
                    hi += __shfl_xor_sync(0xffffffffu, hi, off);
                }
                if (lq == 0)
                    sout[line * g.BY + i] = C{hr, hi};
            }

            __syncthreads();

            // Write (or accumulate, if the window of k'y started in an earlier chunk) to global memory.
            for (int l = 0; l < nl; l++) {
                C *dst = outb + size_t(o_first + l) * My;
                for (int i = tid; i < nby; i += T) {
                    C v = sout[l * g.BY + i];
                    if (g.nch > 1) {
                        const int ch_first = (g.y.window_start(Y0 + i) - ys) / (2 * T);
                        if (ch < ch_first)
                            continue;
                        if (ch > ch_first) {
                            C u = dst[i];
                            v.x += u.x;
                            v.y += u.y;
                        }
                    }
                    dst[i] = v;
                }
            }

            o_first += nl;
            nl = 0;
        };

        // Stage 1: scatter the modes v[4] from the current row (with phase index p) into the
        // accumulators, then advance to the next row. If the row completes group q, then output
        // k'x = q + dmin_x is complete, and is written to the transpose buffer.
        auto process_row = [&](const Real v[4]) {
            if (active) {
                const Real *wrow = ws_x + p * Kx;
#pragma unroll
                for (int k = 0; k < KMAX; k++) {
                    const Real w = (k < Kx) ? __ldg(wrow + k) : Real(0);
                    a0r[k] += w * v[0];
                    a0i[k] += w * v[1];
                    a1r[k] += w * v[2];
                    a1i[k] += w * v[3];
                }
            }

            p += Ax;
            if (p < Px)
                return;

            p -= Px;
            if (q + g.x.dmin >= X0) {
                store2(buf + nl * W + 2 * tid, a0r[0], a0i[0], a1r[0], a1i[0]);
                nl++;
            }
#pragma unroll
            for (int k = 0; k < KMAX - 1; k++) {
                a0r[k] = a0r[k+1];
                a0i[k] = a0i[k+1];
                a1r[k] = a1r[k+1];
                a1i[k] = a1i[k+1];
            }
            a0r[KMAX-1] = a0i[KMAX-1] = a1r[KMAX-1] = a1i[KMAX-1] = Real(0);
            q++;

            if ((nl == G) || (q > q_end))
                flush();
        };

        int kx = kx_start;
        for (int row = 0; row < nrows; row++) {
            Real v[4];
            if (active) {
                const int nx = signed_wavenumber(kx, N);
                const U ph_row = phx[kx] + phz_kz;
                if (kz > 0)
                    gen.pair(nx, ny0, ny1, kz, ph_row + phy0, ph_row + phy1, v);
                else {
                    gen.single_kz0(nx, ny0, ph_row + phy0, v[0], v[1]);
                    gen.single_kz0(nx, ny1, ph_row + phy1, v[2], v[3]);
                }
            }
            process_row(v);
            kx = (kx == N - 1) ? 0 : (kx + 1);
        }
    }
}


struct LaunchArgs {
    cudaStream_t stream;
    int Bz;
    const int32_t *kz0;
    const void *ws_x, *gw_y, *phx, *phy, *phz;
    void *out;
};


template <typename Real>
size_t smem_bytes(int T, int BY)
{
    constexpr int G = Tuning<Real>::G;
    return size_t(G) * (2 * T + 32 / G + BY) * sizeof(typename Cplx<Real>::T);
}


struct Plan {
    int BY, nch, T, seg_len, nstrips, nseg;
    size_t smem;
};

// Chooses the thread block shape, to minimize an estimate of the run time. We assume that thread
// blocks run in waves of (blocks/SM) * (#SMs), and that the time per wave is proportional to the
// number of threads per SM, times the number of column chunks per block, times the number of rows per
// block (proportional to seg_len + Kx - 1, including the halo), divided by an efficiency factor which
// depends on the number of warps per SM (empirically, about sqrt(warps/16) on an L40S). The number of
// threads per block T is at least (number of columns needed by a strip) / (2 * nch), so the estimate
// accounts for the regridding halo in y (columns needed by two adjacent strips), idle threads, and
// partial waves.
template <typename Real>
Plan make_plan(const Geometry &g, int Bz, const void *kernel, int nsm, int smem_max)
{
    const int My = g.y.M, Mx = g.x.M, Kx = g.x.K;
    int tmax = Tuning<Real>::TMAX;
    while ((tmax > 64) && (smem_bytes<Real>(tmax, kMaxStrip) > size_t(smem_max)))
        tmax -= 32;

    Plan best{};
    double best_cost = 1e300;
    const int ns_min = (My + kMaxStrip - 1) / kMaxStrip;
    int prev_BY = 0;

    for (int ns = ns_min; ns <= std::min(My, ns_min + 64); ns++) {
        const int BY = (My + ns - 1) / ns;
        const int nstrips = (My + BY - 1) / BY;
        if (BY == prev_BY)
            continue;
        prev_BY = BY;

        int64_t cmax = 0;
        for (int s = 0; s < nstrips; s++) {
            int Y0 = s * BY;
            int Y1 = std::min(Y0 + BY, My);
            int64_t ys = g.y.window_start_host(Y0) & ~int64_t(1);
            cmax = std::max(cmax, g.y.window_start_host(Y1 - 1 + g.y.K) - ys);
        }
        const int nch = int((cmax + 2 * tmax - 1) / (2 * tmax));
        const int T = int(std::max<int64_t>(64, ((cmax + 2 * nch - 1) / (2 * nch) + 31) / 32 * 32));
        const size_t smem = smem_bytes<Real>(T, BY);
        if (smem > size_t(smem_max))
            continue;

        int bps = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, T, smem);
        bps = std::max(bps, 1);
        const long wave = long(bps) * nsm;

        for (int nsx = 1; nsx <= std::max(1, Mx / (2 * Kx)); nsx++) {
            const int len = (Mx + nsx - 1) / nsx;
            const int nseg = (Mx + len - 1) / len;
            const long nblocks = long(nstrips) * nseg * Bz;
            const double eff = std::sqrt(std::min(1.0, bps * T / 512.0));
            const double cost = double((nblocks + wave - 1) / wave) * bps * T * nch * (len + Kx - 1) / eff;
            if (cost < best_cost * (1 - 1e-9)) {
                best_cost = cost;
                best = Plan{BY, nch, T, len, nstrips, nseg, smem};
            }
        }
    }
    return best;
}


// Plans are cached, since make_plan() calls the CUDA occupancy API many times.
template <typename Real>
Plan get_plan(const Geometry &g, int Bz, const void *kernel, int kmax, cudaError_t &err)
{
    static std::mutex mutex;
    static std::map<std::vector<int64_t>, Plan> cache;

    int dev = 0, nsm = 0, smem_max = 0;
    cudaGetDevice(&dev);
    std::vector<int64_t> key = {dev, int64_t(sizeof(Real)), kmax, Bz, g.nside,
                                g.x.M, g.x.A, g.x.P, g.x.K, g.x.dmin, g.y.M, g.y.A, g.y.P, g.y.K, g.y.dmin};

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;

    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
    if (err != cudaSuccess)
        return Plan{};

    Plan plan = make_plan<Real>(g, Bz, kernel, nsm, smem_max);
    cache[key] = plan;
    if (getenv("ICTILE_DEBUG")) {   // print the plan (for tuning)
        int bps = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, plan.T, plan.smem);
        printf("ictile plan: My=%d BY=%d nstrips=%d nch=%d T=%d smem=%zu bps=%d | Mx=%d seg_len=%d nseg=%d Bz=%d\n",
               g.y.M, plan.BY, plan.nstrips, plan.nch, plan.T, plan.smem, bps, g.x.M, plan.seg_len, plan.nseg, Bz);
    }
    return plan;
}


template <typename Real, int KMAX>
ffi::Error launch(const ModeGenerator<Real> &gen, Geometry g, const LaunchArgs &a)
{
    using U = typename Fixed<Real>::U;
    using C = typename Cplx<Real>::T;

    auto kernel = h2_kernel<Real, KMAX>;
    cudaError_t err = cudaSuccess;
    const Plan plan = get_plan<Real>(g, a.Bz, reinterpret_cast<const void *>(kernel), KMAX, err);
    if (err != cudaSuccess)
        return ffi::Error::Internal(cuda_error("cudaFuncSetAttribute failed", err));
    if (plan.T == 0)
        return ffi::Error::Internal("ictile: shared memory requirement too large");
    g.BY = plan.BY;
    g.nch = plan.nch;
    g.seg_len = plan.seg_len;

    dim3 grid(plan.nstrips, plan.nseg, a.Bz);
    kernel<<<grid, plan.T, plan.smem, a.stream>>>(gen, g, a.kz0,
                                                  static_cast<const Real *>(a.ws_x),
                                                  static_cast<const Real *>(a.gw_y),
                                                  static_cast<const U *>(a.phx),
                                                  static_cast<const U *>(a.phy),
                                                  static_cast<const U *>(a.phz),
                                                  static_cast<C *>(a.out));

    err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(cuda_error("kernel launch failed", err));
    return ffi::Error::Success();
}


template <typename Real>
ffi::Error dispatch_kmax(const ModeGenerator<Real> &gen, const Geometry &g, const LaunchArgs &a)
{
    if (g.x.K <= 8)
        return launch<Real, 8>(gen, g, a);
    if (g.x.K <= 12)
        return launch<Real, 12>(gen, g, a);
    if (g.x.K <= 16)
        return launch<Real, 16>(gen, g, a);
    return ffi::Error::InvalidArgument("ictile: kernel width too large (need K <= 16)");
}


bool init_axis(AxisMap &ax, int nside, int64_t M, int64_t A, int64_t P, int64_t K, int64_t dmin, const int32_t *so)
{
    ax.M = int(M);
    ax.A = int(A);
    ax.P = int(P);
    ax.K = int(K);
    ax.dmin = int(dmin);
    ax.divA = FastDiv(uint32_t(std::max<int64_t>(A, 1)));
    ax.so = so;
    return (M >= 1) && (A >= 1) && (A <= P) && (K >= 1) && (M * P == int64_t(nside) * A);
}


template <typename Real, ffi::DataType RT, ffi::DataType UT, ffi::DataType CT>
ffi::Error h2_cuda(cudaStream_t stream,
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

    Geometry g;
    g.nside = int(nside);
    if ((g.nside % 2) || (phx.element_count() != size_t(g.nside)) ||
        !init_axis(g.x, g.nside, odims[1], so_x.element_count(), xdims[0], xdims[1], dmin_x, so_x.typed_data()) ||
        !init_axis(g.y, g.nside, odims[2], so_y.element_count(), py, ky, dmin_y, so_y.typed_data()) ||
        (pf_x.element_count() != so_x.element_count()) || (int64_t(gdims[0]) != g.y.A) ||
        (gdims[1] < ly + 32 / Tuning<Real>::G))
        return ffi::Error::InvalidArgument("ictile: inconsistent nside/ntile/tables");
    if ((deriv < -1) || (deriv > 2))
        return ffi::Error::InvalidArgument("ictile: deriv must be -1, 0, 1, or 2");
    g.pf_x = pf_x.typed_data();
    g.Ly = int(ly);
    g.LWy = int(gdims[1]);

    ModeGenerator<Real> gen;
    gen.nside = g.nside;
    gen.key0 = uint32_t(key0);
    gen.key1 = uint32_t(key1);
    gen.stream = uint32_t(stream_id);
    gen.sqrtp = sqrtp.typed_data();
    gen.ntab = int(sqrtp.element_count());
    gen.tab_scale = Real(tab_scale);
    gen.fixed_amplitude = int(fixed_amplitude);
    gen.deriv = int(deriv);

    LaunchArgs a;
    a.stream = stream;
    a.Bz = int(odims[0]);
    a.kz0 = kz0.typed_data();
    a.ws_x = ws_x.typed_data();
    a.gw_y = gw_y.typed_data();
    a.phx = phx.typed_data();
    a.phy = phy.typed_data();
    a.phz = phz.typed_data();
    a.out = out->untyped_data();

    return dispatch_kmax<Real>(gen, g, a);
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

ffi::Error h2_cuda_f32(cudaStream_t stream, ICTILE_H2_ARGS(ffi::F32, ffi::U32, ffi::C64))
{
    return h2_cuda<float, ffi::F32, ffi::U32, ffi::C64>(stream, ICTILE_H2_FORWARD);
}

ffi::Error h2_cuda_f64(cudaStream_t stream, ICTILE_H2_ARGS(ffi::F64, ffi::U64, ffi::C128))
{
    return h2_cuda<double, ffi::F64, ffi::U64, ffi::C128>(stream, ICTILE_H2_FORWARD);
}

}  // anonymous namespace
}  // namespace ictile


#define ICTILE_H2_BINDING(RT, UT, CT)                                   \
    ffi::Ffi::Bind()                                                    \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()                       \
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
        .Ret<ffi::Buffer<RT>>()         /* workspace (unused on GPU) */ \
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(IctileH2F32Cuda, ictile::h2_cuda_f32,
                              ICTILE_H2_BINDING(ffi::F32, ffi::U32, ffi::C64));

XLA_FFI_DEFINE_HANDLER_SYMBOL(IctileH2F64Cuda, ictile::h2_cuda_f64,
                              ICTILE_H2_BINDING(ffi::F64, ffi::U64, ffi::C128));
