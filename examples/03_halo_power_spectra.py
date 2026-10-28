"""Power spectra of many halo catalogs, from the N-body emulator (02_make_halo_catalog.py) and from the Quijote N-body
simulations, for 04_monte_carlo_comparison.ipynb. Runs on CPUs, with kszx (https://github.com/kmsmith137/kszx).

    python 03_halo_power_spectra.py match --emulator-dir EMU_DIR [--nboxes 300] [--out matching.json]
    python 03_halo_power_spectra.py compute {emulator,quijote} START [END] --outdir DIR [--emulator-dir EMU_DIR]
                                            [--matching matching.json]

Catalogs. The emulator's are EMU_DIR/halo_catalog_{i}.h5, made by 02_make_halo_catalog.py with its default parameters,
which match Quijote's. Quijote's are the FoF catalogs of the fiducial realizations at z = 0, read with
kszx.quijote.read_halos(). Download them first (34 MB each; needs Globus, see kszx/globus_utils.py):

    python -c "import kszx; kszx.quijote.download('fiducial', 1000, products=('halos',))"

Both kinds of catalogs have positions in Mpc and velocities in units of c, in a box of side L = 1000/h Mpc
(h = 0.6711). Both have the halos with at least 20 particles.

'match' chooses subsamples of the two kinds of catalogs with the same large-scale halo bias b and the same number
density, using the first NBOXES catalogs of each kind (~10 minutes for 300):

  - b minimizes the power of the residual r = h - b v (see below) at k < KFIT: b = sum Re(h v*) / sum |v|^2, summed
    over the modes and the catalogs. Sample variance cancels mode by mode, so 300 boxes give b to ~0.1%.
  - At the same minimum particle number, the emulator's halos are less biased than Quijote's. The emulator's minimum
    particle number N is raised until its b matches Quijote's (all halos), and made continuous by also keeping a random
    fraction of the halos with N - 1 particles.
  - The more abundant of the two samples is then diluted at random to the other's number density. This doesn't change
    b; it only adds shot noise.

It writes the two selections, and a table of b and the number of halos against N, to a JSON file.

'compute' writes DIR/pk_{kind}_{i}.npz for each catalog i in range(START, END): the power spectra of all the halos
and, with --matching, of the matched subsample, and the number of halos with each particle number (for the mass
function). Files that exist are skipped (unless they lack the matched sample, or
it was selected differently), so an interrupted job finishes when rerun. 15-30 s per catalog with 16 threads.

Fields, in the Fourier conventions of kszx (f(k) = sum_x V_pix f(x) e^{-ikx}, <f(k) g(k')^*> = V P_fg(k) delta_kk'):

    h(k) = sum_i e^{-ik.x_i} / nbar          halo number density contrast
    p(k) = sum_i v_i e^{-ik.x_i} / nbar      halo momentum density (1 + delta_h) v, in units of c
    q(k) = -i khat . p(k)                    gradient (longitudinal) mode; in linear theory, q = (faH/c) delta / k
    c(k) = khat x p(k)                       curl (transverse) mode, 2 components
    v(k) = k q(k) / (faH/c)                  velocity-inferred density: equal to delta_matter in linear theory

f = 0.5295 is the growth rate at z = 0 for Omega_m = 0.3175 (as computed by jax_nbody_emulator; CAMB gives 0.5287,
with radiation). It only sets the normalization of v, and so of b.

Spectra: P[s, j] for s in SPECTRA = (hh, hq, qq, cc, hv, vv), where P_cc = P_pxpx + P_pypy + P_pzpz - P_qq is the
power of both curl components. Units: Mpc^3 times (v/c)^n, with n = 0, 1, 2, 2, 0, 0. The halos are gridded on an
NPIX^3 grid with kszx's cubic kernel, which is then divided out. The bins are [(j + 1/2) k_f, (j + 3/2) k_f) with
k_f = 2 pi / L, up to k = 0.8 Mpc^-1 (189 bins). Shot noise is not subtracted.

Contents of pk_{kind}_{i}.npz:

    k, k2, counts, kbin_edges     mean k and k^2 [Mpc^-1, Mpc^-2] and number of modes in each bin, and the bin edges
    spectra, box_size, npix, faH  the names of the spectra, L [Mpc], NPIX, f a H / c [Mpc^-1]
    kind, catalog_index           'emulator' or 'quijote', and i
    samples                       ['all'], or ['all', 'matched'] with --matching
    npart_values, npart_counts    the number of halos (npart_counts) with each particle number (npart_values)
    particle_mass                 the particle mass [Msun], so a halo's mass is npart * particle_mass
    for each sample s:
      P_s                         the spectra, shape (6, nbins)
      nhalos_s, nbar_s, v2_s      number of halos, their number density [Mpc^-3], and the mean of |v|^2 [(v/c)^2]
      shot_noise_s                expected shot noise of each spectrum: 1/nbar, 0, <v^2>/(3 nbar), 2<v^2>/(3 nbar),
                                  0, and nan for P_vv, whose noise depends on k:
      noise_vv_s                  k^2 <v^2> / (3 nbar) / (faH/c)^2 in each bin
      selection_s                 the selection (JSON): min_npart, boundary_fraction, keep_fraction
"""

import argparse
import datetime
import json
import os
import signal
import sys
import time

import numpy as np
import h5py
import kszx

# Quijote's fiducial cosmology (and the defaults of 02_make_halo_catalog.py).
QUIJOTE = dict(omega_m=0.3175, h=0.6711, box_size=1000.0)       # box size in Mpc/h
L = QUIJOTE['box_size'] / QUIJOTE['h']                          # box size [Mpc]
F_GROWTH = 0.5295                                               # growth rate f at z = 0
C_KMS = 299792.458
FAH = F_GROWTH * 100 * QUIJOTE['h'] / C_KMS                     # f a H / c at z = 0 [Mpc^-1]

SPECTRA = ('hh', 'hq', 'qq', 'cc', 'hv', 'vv')
MIN_NPART = 20          # smallest halos in both kinds of catalogs
KMAX = 0.8              # Mpc^-1
KFIT = 0.02             # Mpc^-1: b is fit at k < KFIT
MATCH_NPIX = 128        # 'match' only needs large scales
MATCH_THRESHOLDS = range(MIN_NPART, MIN_NPART + 7)              # emulator minimum particle numbers tried by 'match'


def default_threads():
    # Slurm sets the CPU affinity; os.cpu_count() would count the whole node.
    return int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or len(os.sched_getaffinity(0))


def read_catalog(kind, i, emulator_dir=None):
    """Returns positions (N,3) [Mpc] in [0, L), velocities (N,3) [v/c], and npart (N,) of all halos in a catalog,
    and the particle mass [Msun]."""
    if kind == 'emulator':
        filename = os.path.join(emulator_dir, f'halo_catalog_{i}.h5')
        with h5py.File(filename, 'r') as f:
            a = f.attrs
            if (abs(a['box_size'] / L - 1) > 1e-6) or (a['redshift'] != 0) or (a['nmin'] != MIN_NPART) or \
                    any(abs(a[k] - QUIJOTE[k]) > 1e-6 for k in ('omega_m', 'h')):
                raise RuntimeError(f'{filename}: expected a catalog made with the default (Quijote-like) parameters '
                                   'of 02_make_halo_catalog.py')
            c = {k: f[k][:] for k in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'npart')}
            particle_mass = float(a['particle_mass'])
    elif kind == 'quijote':
        cat = kszx.quijote.read_halos('fiducial', i, 0.0)
        c = {k: getattr(cat, k) for k in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'npart')}
        particle_mass = float(np.median(cat.mass / cat.npart))
    else:
        raise ValueError(kind)
    pos = np.stack([c['x'], c['y'], c['z']], axis=1).astype(np.float64) % L
    vel = np.stack([c['vx'], c['vy'], c['vz']], axis=1).astype(np.float64)
    return pos, vel, np.asarray(c['npart']), particle_mass


def select(kind, i, npart, min_npart=MIN_NPART, boundary_fraction=0.0, keep_fraction=1.0):
    """Boolean mask: the halos with npart >= min_npart, plus a random fraction boundary_fraction of those with
    npart = min_npart - 1, then a random fraction keep_fraction of these. The random numbers are seeded by (kind, i)."""
    rng = np.random.default_rng([i, ('emulator', 'quijote').index(kind), 20261027])
    u1, u2 = rng.random(len(npart)), rng.random(len(npart))
    s = (npart >= min_npart) | ((npart == min_npart - 1) & (u1 < boundary_fraction))
    return s & (u2 < keep_fraction)


def fourier_fields(box, pos, vel, kernel, threads):
    """h, px, py, pz (see the docstring), with the gridding kernel divided out."""
    nbar = len(pos) / L**3
    maps = []
    for w in (None, vel[:, 0], vel[:, 1], vel[:, 2]):
        m = kszx.grid_points(box, pos, weights=w, kernel=kernel, periodic=True)
        m = kszx.fft_r2c(box, m, threads=threads)
        kszx.apply_kernel_compensation(box, m, kernel)
        m /= nbar
        maps.append(m)
    return maps


def gradient_mode(box, px, py, pz):
    """q = -i khat . p"""
    q = sum(box.get_k_component(a) * m for a, m in enumerate((px, py, pz))) * (-1j / box.get_k(regulate=True))
    q[0, 0, 0] = 0
    return q


def power_spectra(pos, vel, npix=512, threads=None):
    """Returns a dict with P (6, nbins) for SPECTRA, and the shot noise, for halos at pos [Mpc] with velocities vel."""
    box = kszx.Box((npix,) * 3, L / npix)
    kf = 2 * np.pi / L
    kbin_edges = kf * (np.arange(int(KMAX / kf) + 1) + 0.5)
    h, px, py, pz = fourier_fields(box, pos, vel, 'cubic', threads)
    q = gradient_mode(box, px, py, pz)
    v = q * (box.get_k() / FAH)
    pk = kszx.estimate_power_spectrum(box, [h, q, px, py, pz, v], kbin_edges)
    P = np.array([pk[0, 0], pk[0, 1], pk[1, 1], pk[2, 2] + pk[3, 3] + pk[4, 4] - pk[1, 1], pk[0, 5], pk[5, 5]])
    k, counts = kszx.kbin_average(box, box.get_k(), kbin_edges, return_counts=True)
    k2 = kszx.kbin_average(box, box.get_k(2), kbin_edges)
    nbar = len(pos) / L**3
    v2 = np.mean(np.sum(vel**2, axis=1))
    return dict(P=P, nhalos=len(pos), nbar=nbar, v2=v2,
                shot_noise=np.array([1 / nbar, 0.0, v2 / (3 * nbar), 2 * v2 / (3 * nbar), 0.0, np.nan]),
                noise_vv=k2 * v2 / (3 * nbar) / FAH**2,
                bins=dict(k=k, k2=k2, counts=counts, kbin_edges=kbin_edges))


def large_scale_sums(pos, vel, threads=None):
    """Sum over the modes with k < KFIT of Re(h v*) and |v|^2, divided by the volume (on a coarse grid)."""
    box = kszx.Box((MATCH_NPIX,) * 3, L / MATCH_NPIX)
    h, px, py, pz = fourier_fields(box, pos, vel, 'cic', threads)
    v = gradient_mode(box, px, py, pz) * (box.get_k() / FAH)
    pk, counts = kszx.estimate_power_spectrum(box, [h, v], np.array([np.pi / L, KFIT]), return_counts=True)
    return pk[0, 1, 0] * counts[0], pk[1, 1, 0] * counts[0]


def match(args):
    t0 = time.time()
    table = {}
    for kind, thresholds in (('emulator', MATCH_THRESHOLDS), ('quijote', [MIN_NPART])):
        hv, vv, nh = (np.zeros((args.nboxes, len(thresholds))) for _ in range(3))
        for i in range(args.nboxes):
            pos, vel, npart, _ = read_catalog(kind, i, args.emulator_dir)
            for t, n in enumerate(thresholds):
                s = npart >= n
                hv[i, t], vv[i, t] = large_scale_sums(pos[s], vel[s], args.threads)
                nh[i, t] = s.sum()
            if (i + 1) % 50 == 0:
                print(f'{kind}: {i + 1} of {args.nboxes} catalogs, {time.time() - t0:.0f} s', flush=True)
        b = hv.sum(0) / vv.sum(0)
        b_err = (hv / vv).std(0, ddof=1) / np.sqrt(args.nboxes)       # from the box-to-box spread
        table[kind] = [dict(min_npart=int(n), b=float(b[t]), b_err=float(b_err[t]), nhalos=float(nh[:, t].mean()))
                       for t, n in enumerate(thresholds)]

    # The emulator's threshold N, plus a fraction phi of the halos with N - 1 particles: b and the number of halos are
    # interpolated linearly in phi.
    target = table['quijote'][0]['b']
    emu = table['emulator']
    j = next((j for j, row in enumerate(emu) if row['b'] >= target), None)
    if j is None:
        raise RuntimeError(f'the emulator halos with >= {emu[-1]["min_npart"]} particles are still less biased than '
                           "Quijote's: extend MATCH_THRESHOLDS")
    if j == 0:
        raise RuntimeError("the emulator's halos are at least as biased as Quijote's: this script assumes the opposite")
    phi = (emu[j]['b'] - target) / (emu[j]['b'] - emu[j - 1]['b'])
    n_emu = emu[j]['nhalos'] + phi * (emu[j - 1]['nhalos'] - emu[j]['nhalos'])
    n_qui = table['quijote'][0]['nhalos']
    n = min(n_emu, n_qui)
    out = dict(emulator=dict(min_npart=emu[j]['min_npart'], boundary_fraction=round(phi, 4),
                             keep_fraction=round(min(1.0, n / n_emu), 4)),
               quijote=dict(min_npart=MIN_NPART, boundary_fraction=0.0, keep_fraction=round(min(1.0, n / n_qui), 4)),
               b=target, nhalos=n, table=table, nboxes=args.nboxes, kfit=KFIT, npix=MATCH_NPIX,
               emulator_dir=os.path.abspath(args.emulator_dir), created=datetime.datetime.now().isoformat(timespec='seconds'))

    for kind, rows in table.items():
        for row in rows:
            print(f'{kind:8s} npart >= {row["min_npart"]}: b = {row["b"]:.4f} ± {row["b_err"]:.4f}, '
                  f'{row["nhalos"]:.0f} halos per box')
    print(f'matched samples: b = {target:.4f}, {n:.0f} halos per box; emulator {out["emulator"]}, quijote {out["quijote"]}')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'wrote {args.out} ({time.time() - t0:.0f} s)')


def is_done(filename, selections):
    """True if filename has the spectra of these samples, selected in the same way, and the halo counts."""
    try:
        with np.load(filename) as f:
            return ('npart_counts' in f) and \
                all((f'P_{s}' in f) and (json.loads(str(f[f'selection_{s}'])) == sel) for s, sel in selections.items())
    except Exception:          # missing or unreadable
        return False


def compute(args):
    selections = {'all': dict(min_npart=MIN_NPART, boundary_fraction=0.0, keep_fraction=1.0)}
    if args.matching:
        with open(args.matching) as f:
            m = json.load(f)[args.kind]
        selections['matched'] = {k: m[k] for k in ('min_npart', 'boundary_fraction', 'keep_fraction')}
    os.makedirs(args.outdir, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))     # e.g. preemption: remove the temporary file

    for i in range(args.start, args.end if args.end is not None else args.start + 1):
        filename = os.path.join(args.outdir, f'pk_{args.kind}_{i}.npz')
        if is_done(filename, selections):
            continue
        t0 = time.time()
        pos, vel, npart, particle_mass = read_catalog(args.kind, i, args.emulator_dir)
        out = dict(kind=args.kind, catalog_index=i, spectra=np.array(SPECTRA), box_size=L, npix=args.npix, faH=FAH,
                   samples=np.array(list(selections)), particle_mass=particle_mass)
        out['npart_values'], out['npart_counts'] = np.unique(npart, return_counts=True)
        for s, sel in selections.items():
            m = select(args.kind, i, npart, **sel)
            res = power_spectra(pos[m], vel[m], args.npix, args.threads)
            out.update(res.pop('bins'))
            out.update({f'{key}_{s}': val for key, val in res.items()})
            out[f'selection_{s}'] = json.dumps(sel)
        tmp = os.path.join(args.outdir, f'.pk_{args.kind}_{i}.tmp{os.getpid()}.npz')
        try:
            np.savez(tmp, **out)
            os.replace(tmp, filename)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        print(f'{filename}: ' + ', '.join(f'{s} {out[f"nhalos_{s}"]} halos' for s in selections)
              + f', {time.time() - t0:.1f} s', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('match', help='choose emulator and Quijote subsamples with the same large-scale bias and density',
                       description="Chooses the matched subsamples, and writes them to a JSON file for 'compute'. "
                                   'See 03_halo_power_spectra.py --help for details.')
    p.add_argument('--emulator-dir', required=True, help='directory of halo_catalog_{i}.h5 from 02_make_halo_catalog.py')
    p.add_argument('--nboxes', type=int, default=300, help='number of catalogs of each kind to use (default 300)')
    p.add_argument('--out', default='matching.json', help='output JSON file (default matching.json)')
    p.add_argument('--threads', type=int, default=default_threads(), help='FFT threads (default: the CPUs available)')

    p = sub.add_parser('compute', help='power spectra of catalogs START..END-1, one file each',
                       description='Writes OUTDIR/pk_{kind}_{i}.npz for i in range(START, END), skipping files that exist. '
                                   'See 03_halo_power_spectra.py --help for the fields and the file format.')
    p.add_argument('kind', choices=('emulator', 'quijote'))
    p.add_argument('start', type=int, help='first catalog')
    p.add_argument('end', type=int, nargs='?', help='last catalog + 1 (default START + 1)')
    p.add_argument('--outdir', required=True, help='directory for the output files')
    p.add_argument('--emulator-dir', help='directory of halo_catalog_{i}.h5 from 02_make_halo_catalog.py')
    p.add_argument('--matching', help="JSON file from 'match': also compute the spectra of the matched subsample")
    p.add_argument('--npix', type=int, default=512, help='grid size (default 512)')
    p.add_argument('--threads', type=int, default=default_threads(), help='FFT threads (default: the CPUs available)')

    args = ap.parse_args()
    if (getattr(args, 'kind', 'emulator') == 'emulator') and not args.emulator_dir:
        ap.error('--emulator-dir is needed for the emulator catalogs')
    if args.command == 'match':
        match(args)
    else:
        compute(args)


if __name__ == '__main__':
    main()
