"""Precomputed tables for tiled Gaussian random fields (numpy, float64).

Conventions follow Appendix A of the notes. Along each axis, M = 2*ntile is the bounding-box
size, and an ambient wavenumber index n (unsigned, 0 <= n < nside) sits at position n*M/nside
on the bounding-box grid. Writing this position as Q + p/P, where P = nside/gcd(nside,M) is the
number of distinct fractional offsets and 0 <= p < P, wavenumber n contributes to the bounding-box
wavenumbers n' = (Q + dmin + k) mod M, for 0 <= k < K, with weight ws[p,k]. If M divides nside,
then P = nside/M, Q = n // P, and p = n % P (the "polyphase" case).
"""

import math
import numpy as np


def es_kernel(z, beta):
    """Exponential-of-semicircle kernel phi(z) = exp(beta*(sqrt(1-z^2)-1)) for |z| < 1, else 0."""
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    m = np.abs(z) < 1
    out[m] = np.exp(beta * (np.sqrt(1 - z[m]**2) - 1))
    return out


def axis_phases(nside, M):
    """Returns (A, P), where P = nside/gcd(nside,M) and A = M/gcd(nside,M), so that M/nside = A/P."""
    g = math.gcd(nside, M)
    return M // g, nside // g


def axis_index(n, nside, M):
    """Returns (Q, p), where n*M/nside = Q + p/P (see module docstring). Here n is an integer array."""
    A, P = axis_phases(nside, M)
    return np.divmod(np.asarray(n, dtype=np.int64) * A, P)


def polyphase_weights(P, w, beta=None):
    """Returns (dmin, ws), where ws[p,k] = phi((dmin + k - p/P) / (w/2)) has shape (P, K).

    The taps are all integers d = dmin + k such that |d - p/P| < w/2 for some p.
    """
    if beta is None:
        beta = 2.30 * w
    delta = np.arange(P) / P
    dlo = np.floor(delta - w/2).astype(int) + 1
    dhi = np.ceil(delta + w/2).astype(int) - 1
    dmin, dmax = int(dlo.min()), int(dhi.max())
    K = dmax - dmin + 1
    z = (dmin + np.arange(K)[None, :] - delta[:, None]) / (w/2)
    return dmin, es_kernel(z, beta)


def window_start(o, nside, M, dmin, K):
    """First (unwrapped) ambient index n which contributes to bounding-box index o (an integer array).

    Output o receives contributions from the window n in [window_start(o), window_start(o+K)).
    """
    A, P = axis_phases(nside, M)
    q = np.asarray(o, dtype=np.int64) - dmin - K + 1
    return -((-q * P) // A)


def gather_tables(nside, M, dmin, ws, pad=8):
    """Tables used by the custom kernels, which depend on bounding-box index o through r = o % A.

    Returns (so, pf, L, gw), where:
      so[r] = window_start(r), so that window_start(o) = (o // A) * P + so[o % A]
      pf[r] = phase index p of ambient index so[r] (see module docstring)
      L = maximum window length
      gw[r,j] = weight of ambient index so[r]+j in output r, with shape (A, L+pad), zero-padded
    """
    A, P = axis_phases(nside, M)
    K = ws.shape[1]
    r = np.arange(A)
    so = window_start(r, nside, M, dmin, K)
    lw = window_start(r + K, nside, M, dmin, K) - so
    pf = so * A - (r - dmin - K + 1) * P
    L = int(lw.max())

    j = np.arange(L + pad)
    Q, p = np.divmod((so[:, None] + j[None, :]) * A, P)
    k = r[:, None] - dmin - Q
    valid = j[None, :] < lw[:, None]
    assert np.all((k[valid] >= 0) & (k[valid] < K)) and np.all((pf >= 0) & (pf < A))
    gw = np.where(valid, ws[p, np.clip(k, 0, K-1)], 0.0)
    return so.astype(np.int32), pf.astype(np.int32), L, gw


def deapodization(t, M, dmin, ws):
    """De-apodization factor psi(t), averaged over the P offsets (Eq. (psi_avg) in the notes).

    Here t is the offset from the tile center, in pixels.
    """
    P, K = ws.shape
    t = np.asarray(t, dtype=float)
    psi = np.zeros(t.shape)
    chunk = max(1, 2**22 // (K * max(t.size, 1)))    # P can be large if M doesn't divide nside
    for p0 in range(0, P, chunk):
        p = np.arange(p0, min(P, p0 + chunk))
        z = dmin + np.arange(K)[None, :] - p[:, None] / P               # (chunk, K)
        c = np.cos(2*np.pi * z[:, :, None] * t[None, None, :] / M)        # (chunk, K, nt)
        psi += np.einsum('pk,pkt->t', ws[p], c)
    return psi / P


def zregrid_windows(nside, ntile, dmin, ws, kz_batch):
    """Regridding matrices for kz -> k'z, in blocks of kz_batch consecutive kz values.

    We loop over 0 <= kz < nside/2 (the Nyquist plane is zero), and accumulate into
    G(x, y, k'z) for 0 <= k'z <= ntile. Mode kz contributes F at +kz and F^* at -kz, with
    weight 1/2 if kz=0. Writing F = X + iY, the contribution is X*Ap + iY*Am, where
    Ap = A+B and Am = A-B, and A,B are the (real) regridding weights from +kz and -kz.

    Returns (k0, Ap, Am), where k0 has shape (nbatch,) and Ap, Am have shape (nbatch, kz_batch, Wn).
    Batch b updates G[:, :, k0[b]:k0[b]+Wn]. Rows with kz >= nside/2 (padding) are zero.
    """
    K = ws.shape[1]
    M = 2 * ntile
    nkz = nside // 2
    nbatch = (nkz + kz_batch - 1) // kz_batch

    A = np.zeros((nbatch * kz_batch, ntile + 1))
    B = np.zeros((nbatch * kz_batch, ntile + 1))
    kz = np.arange(nkz)
    c = np.where(kz == 0, 0.5, 1.0)

    for sgn, T in ((1, A), (-1, B)):
        Q, p = axis_index((sgn * kz) % nside, nside, M)
        for k in range(K):
            kp = (Q + dmin + k) % M
            m = kp <= ntile
            np.add.at(T, (kz[m], kp[m]), c[m] * ws[p[m], k])

    Ap = (A + B).reshape(nbatch, kz_batch, ntile + 1)
    Am = (A - B).reshape(nbatch, kz_batch, ntile + 1)

    nz = np.any((Ap != 0) | (Am != 0), axis=1)    # (nbatch, ntile+1)
    lo = np.array([np.flatnonzero(r)[0] if r.any() else 0 for r in nz])
    hi = np.array([np.flatnonzero(r)[-1] if r.any() else 0 for r in nz])
    Wn = min(int(np.max(hi - lo)) + 1, ntile + 1)
    k0 = np.minimum(lo, ntile + 1 - Wn)

    Ap_w = np.stack([Ap[b, :, k0[b]:k0[b]+Wn] for b in range(nbatch)])
    Am_w = np.stack([Am[b, :, k0[b]:k0[b]+Wn] for b in range(nbatch)])
    assert np.allclose(np.abs(Ap_w).sum(), np.abs(Ap).sum()) and np.allclose(np.abs(Am_w).sum(), np.abs(Am).sum())
    return k0.astype(np.int32), Ap_w, Am_w


def sqrtp_table(pk, nside, box_size, ntab=4096):
    """Table of sqrt(P(k)/V), uniform in t = log2(n^2) on [0, log2(3*(nside/2)^2)].

    Here k = (2pi/box_size) * sqrt(n^2), and V = box_size^3. Returns (table, tab_scale), where
    the table index is t * tab_scale.
    """
    tmax = np.log2(3 * (nside//2)**2)
    t = np.linspace(0, tmax, ntab)
    k = (2*np.pi / box_size) * 2**(t/2)
    p = np.asarray(pk(k), dtype=float)
    if (p.shape != k.shape) or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError('ictile: power spectrum must return finite, non-negative values with the same shape as k')
    return np.sqrt(p / box_size**3), (ntab - 1) / tmax
