"""Tests for ictile. Run with 'pytest' from the repo root."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from conftest import PLATFORMS, device
from ictile import TiledGRF, philox4x32_10, mode_values, custom_kernel_available


def pk_test(k):
    return 1e3 * k**-1.5 * np.exp(-(k / 2.0)**2)


def tile_of(full, start, ntile):
    N = full.shape[0]
    ntile = np.broadcast_to(ntile, (3,))
    idx = [(start[i] + np.arange(ntile[i])) % N for i in range(3)]
    return full[np.ix_(*idx)]


def test_philox_kat():
    # Random123 known-answer vectors for Philox4x32-10.
    M = 0xFFFFFFFF
    kat = [
        ([0, 0, 0, 0], 0, [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]),
        ([M, M, M, M], (M << 32) | M, [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]),
        ([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], (0x299f31d0 << 32) | 0xa4093822,
         [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]),
    ]
    for ctr, key, expected in kat:
        got = [int(x) for x in philox4x32_10(*[jnp.uint32(c) for c in ctr], key)]
        assert got == expected


def _modes(N, pk=lambda k: np.ones_like(k), fixed_amplitude=False, seed=1):
    g = TiledGRF(N, float(N), pk, seed=seed, ntile=N//2, dtype=np.float64, backend="jax", w=4,
                 device=jax.devices('cpu')[0], fixed_amplitude=fixed_amplitude)
    k = np.arange(N)
    n = np.where(k >= N//2, k - N, k)
    a = mode_values(n[:, None, None], n[None, :, None], np.arange(N//2)[None, None, :],
                    nside=N, **g._gen_kwargs(g._tables))
    return np.asarray(a), n


def test_mode_symmetries():
    N = 64
    a, n = _modes(N)
    # Hermitian symmetry in the kz=0 plane: a(-kx,-ky,0) = a(kx,ky,0)^*.
    a0 = a[:, :, 0]
    neg = (-np.arange(N)) % N
    assert np.array_equal(a0[np.ix_(neg, neg)], np.conj(a0))
    # Nyquist modes and k=0 are zero.
    assert np.all(a[N//2, :, :] == 0) and np.all(a[:, N//2, :] == 0) and (a[0, 0, 0] == 0)
    # Unit variance (white spectrum with P/V = 1/N^3). The statistical error is 0.3%.
    nonzero = np.abs(a) > 0
    var = np.mean(np.abs(a[nonzero])**2) * N**3
    assert abs(var - 1) < 0.01


def test_resolution_independence():
    # With a white spectrum, modes with |n_i| < N/2 don't depend on N (up to the overall
    # normalization 1/sqrt(V), since box_size = N in _modes()).
    a1, _ = _modes(16)
    a2, _ = _modes(32)
    n = np.r_[0:8, -7:0]
    sub = a2[np.ix_(n % 32, n % 32, np.arange(8))] * np.sqrt(32.0**3 / 16.0**3)
    ref = a1[np.ix_(n % 16, n % 16, np.arange(8))]
    assert np.allclose(sub, ref, rtol=1e-12, atol=0)


def test_fixed_amplitude():
    a, _ = _modes(16, fixed_amplitude=True)
    m = np.abs(a) > 0
    assert np.allclose(np.abs(a[m]), 16.0**-1.5, rtol=1e-12)


@pytest.mark.parametrize('platform', PLATFORMS)
@pytest.mark.parametrize('dtype,w,tol', [(np.float32, 8, 2e-6), (np.float32, 9, 2e-6), (np.float64, 13, 1e-14), (np.float64, 7, 1e-14)])
def test_h2_custom_vs_jax(platform, dtype, w, tol):
    # Custom C++/CUDA kernel vs pure-JAX reference, for the first part of step 2(a) of algorithm 2.
    if not custom_kernel_available(platform):
        pytest.skip(f'custom kernel not built for {platform}')
    # Includes non-cubic tiles, tiles whose bounding box M = 2*ntile doesn't divide N, and tiles
    # smaller than the kernel width.
    cases = [(64, 16), (64, 32), (96, 16), (64, (16, 32, 16)), (96, (16, 24, 48)), (96, (48, 16, 24)),
             (64, (10, 10, 10)), (120, (7, 13, 11)), (96, (48, 47, 9)), (64, (3, 5, 2))]
    if w not in (8, 13):    # non-default kernel widths: fewer cases, to save time
        cases = [(64, 16), (96, (16, 24, 48)), (120, (7, 13, 11))]
    for N, nt in cases:
        g = TiledGRF(N, 2.0*N, pk_test, seed=12345, ntile=nt, dtype=dtype, w=w, device=device(platform), kz_batch=12)
        for kz0 in [0, 24]:
            for start in [(0, 0, 0), (5, N-3, 33)]:
                a = np.asarray(g.h2(kz0, start, backend='custom'))
                b = np.asarray(g.h2(kz0, start, backend='jax'))
                err = np.abs(a - b).max() / np.abs(b).max()
                assert err < tol, (N, nt, kz0, start, err)


@pytest.mark.parametrize('platform', PLATFORMS)
@pytest.mark.parametrize('dtype,tol', [(np.float32, 2e-6), (np.float64, 1e-14)])
def test_h2_displacement_custom_vs_jax(platform, dtype, tol):
    # As test_h2_custom_vs_jax, with modes multiplied by i n_j/n^2 (displacement psi_j).
    if not custom_kernel_available(platform):
        pytest.skip(f'custom kernel not built for {platform}')
    for N, nt in [(64, 16), (120, (7, 13, 11)), (64, (3, 5, 2))]:
        g = TiledGRF(N, 2.0*N, pk_test, seed=12345, ntile=nt, dtype=dtype, device=device(platform), kz_batch=12)
        for deriv in range(3):
            for kz0, start in [(0, (0, 0, 0)), (24, (5, N-3, 33))]:
                a = np.asarray(g.h2(kz0, start, backend='custom', deriv=deriv))
                b = np.asarray(g.h2(kz0, start, backend='jax', deriv=deriv))
                err = np.abs(a - b).max() / np.abs(b).max()
                assert err < tol, (N, nt, deriv, kz0, start, err)


@pytest.mark.parametrize('platform', PLATFORMS)
@pytest.mark.parametrize('dtype,tol', [(np.float32, 1e-5), (np.float64, 1e-14)])
def test_h2_large_window(platform, dtype, tol):
    # Small tiles in a large box: on the GPU, the ky window of each strip of k'y outputs is wider
    # than a thread block, and is processed in several chunks. (Each output is a sum over ~512^2
    # modes, so float32 roundoff is larger than in test_h2_custom_vs_jax.)
    if not custom_kernel_available(platform):
        pytest.skip(f'custom kernel not built for {platform}')
    N = 1024
    for nt in [(8, 8, 8), (9, 7, 4)]:
        g = TiledGRF(N, 2.0*N, pk_test, seed=99, ntile=nt, dtype=dtype, device=device(platform), kz_batch=4)
        for kz0, start in [(0, (0, 0, 0)), (300, (17, 1000, 5))]:
            a = np.asarray(g.h2(kz0, start, backend='custom'))
            b = np.asarray(g.h2(kz0, start, backend='jax'))
            err = np.abs(a - b).max() / np.abs(b).max()
            assert err < tol, (nt, kz0, start, err)


@pytest.mark.parametrize('platform', PLATFORMS)
@pytest.mark.parametrize('N,nt', [(64, 16), (128, 32), (128, 64), (128, (32, 64, 16)), (120, (15, 20, 30)),
                                  (64, (16, 16, 10)), (128, (32, 16, 21)), (120, (15, 20, 7)), (64, (16, 16, 3)),
                                  (64, (10, 12, 7)), (96, (47, 9, 30)), (64, (3, 2, 5))])
def test_tile_vs_full_box(platform, N, nt):
    # Tiles vs an ordinary FFT of the full box (float64), including tiles which wrap around.
    # (The float32 tiles are compared to a float64 reference, since the float32 full-box FFT
    # has larger roundoff error than the tiled algorithm.)
    dev = device(platform)
    full = np.asarray(TiledGRF(N, 2.0*N, pk_test, seed=777, ntile=nt, dtype=np.float64, device=dev).full_box())
    rms = full.std()
    for dtype, tol_rms, tol_max in [(np.float32, 2e-6, 1e-5), (np.float64, 1e-11, 1e-10)]:
        g = TiledGRF(N, 2.0*N, pk_test, seed=777, ntile=nt, dtype=dtype, device=dev, kz_batch=12)
        for start in [(0, 0, 0), (N-5, 17, N//2 + 3)]:
            t = np.asarray(g.tile(start))
            assert t.shape == tuple(np.broadcast_to(nt, (3,)))
            e = np.abs(t - tile_of(full, start, nt)) / rms
            msg = (np.dtype(dtype).name, start)
            assert np.sqrt(np.mean(e**2)) < tol_rms, msg
            assert e.max() < tol_max, msg


@pytest.mark.parametrize('platform', PLATFORMS)
@pytest.mark.parametrize('N,nt', [(64, 16), (120, (15, 20, 7)), (64, (10, 12, 7)), (64, (3, 2, 5))])
def test_displacement_vs_full_box(platform, N, nt):
    # Displacement tiles vs an ordinary FFT of the full box, as in test_tile_vs_full_box.
    dev = device(platform)
    full = np.asarray(TiledGRF(N, 2.0*N, pk_test, seed=777, ntile=nt, dtype=np.float64, device=dev).full_box_displacement())
    rms = full.std(axis=(1, 2, 3))[:, None, None, None]
    for dtype, tol_rms, tol_max in [(np.float32, 2e-6, 1e-5), (np.float64, 1e-11, 1e-10)]:
        g = TiledGRF(N, 2.0*N, pk_test, seed=777, ntile=nt, dtype=dtype, device=dev, kz_batch=12)
        for start in [(0, 0, 0), (N-5, 17, N//2 + 3)]:
            t = np.asarray(g.displacement_tile(start))
            assert t.shape == (3,) + tuple(np.broadcast_to(nt, (3,)))
            e = np.abs(t - np.stack([tile_of(full[j], start, nt) for j in range(3)])) / rms
            msg = (np.dtype(dtype).name, start)
            assert np.sqrt(np.mean(e**2)) < tol_rms, msg
            assert e.max() < tol_max, msg


@pytest.mark.parametrize('platform', PLATFORMS)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_whole_box_tiles(platform, dtype):
    # ntile = nside: tile() and displacement_tile() are the full box, rolled so that 'start' is at index 0.
    N = 32
    dev = device(platform)
    g = TiledGRF(N, 2.0*N, pk_test, seed=777, ntile=N, dtype=dtype, device=dev)
    assert g.full and (g.backend == 'jax')
    full, psi = np.asarray(g.full_box()), np.asarray(g.full_box_displacement())
    tol = 1e-6 if (dtype == np.float32) else 1e-13
    for start in [(0, 0, 0), (N-5, 17, N//2 + 3), (2*N + 1, -3, 7)]:
        t, d = np.asarray(g.tile(start)), np.asarray(g.displacement_tile(start))
        assert t.shape == (N, N, N) and d.shape == (3, N, N, N)
        assert np.abs(t - tile_of(full, start, N)).max() < tol * full.std(), start
        assert np.abs(d - np.stack([tile_of(psi[j], start, N) for j in range(3)])).max() < tol * psi.std(), start


@pytest.mark.parametrize('platform', PLATFORMS)
def test_whole_box_vs_tiles(platform):
    # In float64, whole-box tiles (an ordinary FFT) agree with ordinary tiles of the same field, to the accuracy of
    # the tiled algorithm (see test_tile_vs_full_box).
    N, nt, start = 64, (16, 32, 10), (59, 17, 35)
    dev = device(platform)
    g = TiledGRF(N, 2.0*N, pk_test, seed=777, ntile=N, dtype=np.float64, device=dev)
    h = TiledGRF(N, 2.0*N, pk_test, seed=777, ntile=nt, dtype=np.float64, device=dev, kz_batch=12)
    a = np.asarray(g.displacement_tile(start))[:, :nt[0], :nt[1], :nt[2]]
    b = np.asarray(h.displacement_tile(start))
    e = np.abs(a - b) / a.std(axis=(1, 2, 3))[:, None, None, None]
    assert np.sqrt(np.mean(e**2)) < 1e-11 and e.max() < 1e-10


def test_ntile_validation():
    # ntile must be <= nside/2 on every axis, or nside on every axis.
    cpu = jax.devices('cpu')[0]
    for nt in [33, 63, (64, 64, 16), (64, 40, 64), 0]:
        with pytest.raises(ValueError, match='ntile'):
            TiledGRF(64, 64.0, pk_test, seed=1, ntile=nt, device=cpu)
    assert not TiledGRF(64, 64.0, pk_test, seed=1, ntile=(32, 1, 32), device=cpu).full
    g = TiledGRF(64, 64.0, pk_test, seed=1, ntile=(64, 64, 64), device=cpu)
    assert g.full and (g.ntile == (64, 64, 64))
    with pytest.raises(RuntimeError, match='ntile = nside'):
        g.h2(0)
    with pytest.raises(ValueError, match='backend'):
        TiledGRF(64, 64.0, pk_test, seed=1, ntile=64, backend='custom', device=cpu)


def test_displacement_divergence():
    # The displacement psi = -grad(del^{-2} delta) satisfies div(psi) = -delta and curl(psi) = 0.
    N, L = 32, 50.0
    g = TiledGRF(N, L, pk_test, seed=3, ntile=8, dtype=np.float64, device=jax.devices('cpu')[0])
    delta = np.asarray(g.full_box())
    psi_k = np.fft.fftn(np.asarray(g.full_box_displacement()), axes=(1, 2, 3))
    k = 2*np.pi/L * np.fft.fftfreq(N, 1.0/N)
    ik = [1j * k[:, None, None], 1j * k[None, :, None], 1j * k[None, None, :]]
    div = np.fft.ifftn(sum(ik[j] * psi_k[j] for j in range(3))).real
    assert np.abs(div + delta).max() < 1e-12 * delta.std()
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        curl = np.fft.ifftn(ik[i] * psi_k[j] - ik[j] * psi_k[i]).real
        assert np.abs(curl).max() < 1e-12 * delta.std()


def test_variance():
    # White spectrum P(k) = P0: var(delta) = P0 * (number of nonzero modes) / V.
    N, L, P0 = 64, 100.0, 3.0
    g = TiledGRF(N, L, lambda k: P0 * np.ones_like(k), seed=5, ntile=16, dtype=np.float64, device=jax.devices('cpu')[0])
    full = np.asarray(g.full_box())
    nmodes = N**3 - (3*N**2 - 3*N + 1) - 1     # exclude Nyquist planes and k=0
    expected = P0 * nmodes / L**3
    assert abs(full.var() / expected - 1) < 0.01

