// Code shared by the CPU and CUDA kernels in ictile.
//
// This header defines the Gaussian random field realization (Philox4x32-10 counter-based RNG,
// Box-Muller transform, Hermitian symmetry, power spectrum table), which must agree exactly
// with the pure-JAX reference in ictile/_modes.py. See Appendix A of the notes.

#pragma once
#include <cstdint>
#include <cmath>

#if defined(__CUDACC__)
#define IC_HD __host__ __device__ __forceinline__
#else
#define IC_HD inline
#endif

namespace ictile {


// Philox4x32-10 (Salmon et al. 2011). Agrees with the Random123 known-answer vectors.
IC_HD void philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                         uint32_t k0, uint32_t k1, uint32_t out[4])
{
#if defined(__CUDACC__)
#pragma unroll
#endif
    for (int r = 0; r < 10; r++) {
        if (r > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
#if defined(__CUDA_ARCH__)
        uint32_t lo0 = 0xD2511F53u * c0, hi0 = __umulhi(0xD2511F53u, c0);
        uint32_t lo1 = 0xCD9E8D57u * c2, hi1 = __umulhi(0xCD9E8D57u, c2);
#else
        uint64_t p0 = uint64_t(0xD2511F53u) * c0;
        uint64_t p1 = uint64_t(0xCD9E8D57u) * c2;
        uint32_t lo0 = uint32_t(p0), hi0 = uint32_t(p0 >> 32);
        uint32_t lo1 = uint32_t(p1), hi1 = uint32_t(p1 >> 32);
#endif
        uint32_t n0 = hi1 ^ c1 ^ k0;
        uint32_t n2 = hi0 ^ c3 ^ k1;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}


// Phases are fixed-point fractions of a turn (uint32 in float32, uint64 in float64), so that
// adding phases mod 2pi is exact integer addition with wraparound.
template <typename Real> struct Fixed;

template <> struct Fixed<float> {
    using U = uint32_t;
    // Fixed-point representation of u2 = (x + 1/2) / 2^32 (the 1/2 is below 32-bit resolution).
    IC_HD static U from_u32(uint32_t x) { return x; }
    // Signed fraction of a turn, in [-1/2, 1/2).
    IC_HD static float turns(U t) { return float(int32_t(t)) * 0x1p-32f; }
    static constexpr U quarter = U(1) << 30;
};

template <> struct Fixed<double> {
    using U = uint64_t;
    IC_HD static U from_u32(uint32_t x) { return (uint64_t(x) << 32) | 0x80000000ull; }
    IC_HD static double turns(U t) { return double(int64_t(t)) * 0x1p-64; }
    static constexpr U quarter = U(1) << 62;
};


// Math wrappers. On the GPU, log2() of n^2 (for the power spectrum table index) uses the fast
// intrinsic in float32, since the index doesn't need full precision.
IC_HD float ic_log2_index(float x)
{
#if defined(__CUDA_ARCH__)
    return __log2f(x);
#else
    return std::log2(x);
#endif
}

IC_HD double ic_log2_index(double x) { return log2(x); }

// Division a/b (for the displacement factor n_j/n^2). On the GPU, float32 uses the fast
// intrinsic (2 ulp), since IEEE division is a significant extra cost in the kernel.
IC_HD float ic_div(float a, float b)
{
#if defined(__CUDA_ARCH__)
    return __fdividef(a, b);
#else
    return a / b;
#endif
}

IC_HD double ic_div(double a, double b) { return a / b; }

IC_HD void ic_sincos_turns(float t, float &s, float &c)
{
#if defined(__CUDA_ARCH__)
    sincospif(2.0f * t, &s, &c);
#else
    double a = 6.283185307179586 * double(t);
    s = float(std::sin(a));
    c = float(std::cos(a));
#endif
}

IC_HD void ic_sincos_turns(double t, double &s, double &c)
{
#if defined(__CUDA_ARCH__)
    sincospi(2.0 * t, &s, &c);
#else
    double a = 6.283185307179586 * t;
    s = std::sin(a);
    c = std::cos(a);
#endif
}

// sqrt(-ln u), where u = (x + 1/2) / 2^32.
template <typename Real> IC_HD Real sqrt_neg_log_u(uint32_t x);

template <> IC_HD float sqrt_neg_log_u<float>(uint32_t x)
{
    float u = fmaf(float(x), 0x1p-32f, 0x1p-33f);
    return sqrtf(-logf(u));
}

template <> IC_HD double sqrt_neg_log_u<double>(uint32_t x)
{
    double u = (double(x) + 0.5) * 0x1p-32;
    return sqrt(-log(u));
}

template <typename T>
IC_HD T ic_load(const T *p)
{
#if defined(__CUDA_ARCH__)
    return __ldg(p);
#else
    return *p;
#endif
}


// Generates Fourier modes a(k) of the Gaussian random field, in the convention
//   delta(r) = sum_k a(k) e^{i k.r},   <|a(k)|^2> = P(k) / V.
//
// Wavenumbers are signed integers (nx, ny, nz) with -nside/2 <= n_i < nside/2, and 0 <= nz < nside/2.
// For nz > 0:
//   a(k) = sqrtP(n^2) * sqrt(-ln u1) * exp(2 pi i u2),   u_j = (x_j + 1/2) / 2^32
// where (x0,x1,x2,x3) = Philox4x32-10(counter = (nx, floor(ny/2), nz, stream), key = seed),
// and (u1,u2) are taken from (x0,x1) if ny is even, (x2,x3) if ny is odd.
// For nz = 0, a(k) is given by the formula above if nx > 0 or (nx == 0 and ny > 0), and otherwise
// by a(k) = a(-k)^*. Nyquist modes (any n_i = -nside/2) and k=0 are zero.
//
// The caller supplies an extra phase (fixed-point turns) which multiplies a(k); this is how the
// tile phase exp(i k.R_c) is applied.
//
// If deriv = j (0 <= j < 3), the modes are multiplied by i n_j / n^2. This gives the Fourier modes of
// the linear displacement psi_j = -d_j del^{-2} delta, up to a factor box_size / (2 pi).

template <typename Real>
struct ModeGenerator {
    using U = typename Fixed<Real>::U;

    int nside;
    uint32_t key0, key1, stream;
    const Real *sqrtp;   // sqrt(P/V), tabulated uniformly in t = log2(n^2), on [0, (ntab-1)/tab_scale]
    int ntab;
    Real tab_scale;
    int fixed_amplitude;
    int deriv;    // -1 (delta), or 0 <= j < 3 (displacement psi_j)

    IC_HD Real sqrt_power(uint32_t n2) const
    {
        Real t = ic_log2_index(Real(n2)) * tab_scale;
        int i = int(t);
        i = (i < ntab-2) ? i : (ntab-2);
        Real f = t - Real(i);
        Real a = ic_load(sqrtp + i);
        Real b = ic_load(sqrtp + i + 1);
        return a + f * (b - a);
    }

    // Mode with given n^2 > 0, Philox outputs (x1, x2), conjugation flag, and extra phase.
    // Here nj is the component n_j of the wavenumber, if deriv = j >= 0.
    IC_HD void value(uint32_t n2, int nj, uint32_t x1, uint32_t x2, bool conj, U phase, Real &re, Real &im) const
    {
        Real amp = sqrt_power(n2);
        if (!fixed_amplitude)
            amp *= sqrt_neg_log_u<Real>(x1);
        U t = Fixed<Real>::from_u32(x2);
        t = (conj ? (U(0) - t) : t) + phase;
        if (deriv >= 0) {
            amp *= ic_div(Real(nj), Real(n2));
            t += Fixed<Real>::quarter;
        }
        Real s, c;
        ic_sincos_turns(Fixed<Real>::turns(t), s, c);
        re = amp * c;
        im = amp * s;
    }

    // Pair of modes (nx, ny0, nz) and (nx, ny1, nz), where ny0 is even, nz > 0, and ny1 = ny0+1
    // (or ny1 = -nside/2, if ny0+1 = nside/2). One Philox call. Output is (re0, im0, re1, im1).
    IC_HD void pair(int nx, int ny0, int ny1, int nz, U ph0, U ph1, Real out[4]) const
    {
        const int h = nside / 2;
        uint32_t x[4];
        philox4x32_10(uint32_t(nx), uint32_t(ny0 >> 1), uint32_t(nz), stream, key0, key1, x);

        out[0] = out[1] = out[2] = out[3] = Real(0);
        if (nx == -h)
            return;

        uint32_t nxz2 = uint32_t(nx*nx) + uint32_t(nz*nz);
        int nj0 = (deriv == 0) ? nx : ((deriv == 1) ? ny0 : nz);
        int nj1 = (deriv == 0) ? nx : ((deriv == 1) ? ny1 : nz);
        if (ny0 != -h)
            value(nxz2 + uint32_t(ny0*ny0), nj0, x[0], x[1], false, ph0, out[0], out[1]);
        if (ny1 != -h)
            value(nxz2 + uint32_t(ny1*ny1), nj1, x[2], x[3], false, ph1, out[2], out[3]);
    }

    // Single mode in the nz=0 plane (Hermitian symmetry applied).
    IC_HD void single_kz0(int nx, int ny, U ph, Real &re, Real &im) const
    {
        const int h = nside / 2;
        re = im = Real(0);
        if ((nx == -h) || (ny == -h) || ((nx == 0) && (ny == 0)))
            return;

        bool canonical = (nx > 0) || ((nx == 0) && (ny > 0));
        int cx = canonical ? nx : -nx;
        int cy = canonical ? ny : -ny;

        uint32_t x[4];
        philox4x32_10(uint32_t(cx), uint32_t(cy >> 1), 0u, stream, key0, key1, x);
        uint32_t n2 = uint32_t(nx*nx) + uint32_t(ny*ny);
        int nj = (deriv == 0) ? nx : ((deriv == 1) ? ny : 0);
        if (cy & 1)
            value(n2, nj, x[2], x[3], !canonical, ph, re, im);
        else
            value(n2, nj, x[0], x[1], !canonical, ph, re, im);
    }
};


// Integer helpers. Wavenumber indices are "unsigned" (FFT order, 0 <= k < nside) or "signed"
// (-nside/2 <= n < nside/2). Unwrapped indices are assumed to be in [-nside, 2*nside).
IC_HD int wrap_index(int k, int nside)
{
    return (k < 0) ? (k + nside) : ((k >= nside) ? (k - nside) : k);
}

IC_HD int signed_wavenumber(int k, int nside)
{
    return (k >= nside/2) ? (k - nside) : k;
}

// Division by a runtime-constant divisor d >= 1, for numerators 0 <= n < 2^31
// (multiply-and-shift, as in Granlund & Montgomery 1994).
struct FastDiv {
    uint32_t d, m, s;

    FastDiv() : d(1), m(1), s(0) { }

    explicit FastDiv(uint32_t d_) : d(d_), s(0)
    {
        while ((uint64_t(1) << s) < d)
            s++;
        m = uint32_t(((uint64_t(1) << 32) * ((uint64_t(1) << s) - d)) / d + 1);
    }

    IC_HD int div(int n) const
    {
#if defined(__CUDA_ARCH__)
        uint32_t t = __umulhi(uint32_t(n), m);
#else
        uint32_t t = uint32_t((uint64_t(uint32_t(n)) * m) >> 32);
#endif
        return int((t + uint32_t(n)) >> s);
    }
};


}  // namespace ictile
