"""Makes halo catalogs: tiled initial conditions (ictile) -> N-body emulator -> friends-of-friends halos (jz-tree).

This is the pipeline of 01_halo_catalog.ipynb, as a script for batch jobs. See that notebook for how it works,
and README.md for installation (this script also needs h5py).

    python 02_make_halo_catalog.py START [END] [--outdir DIR] [options]

writes DIR/halo_catalog_{i}.h5 for i in range(START, END) (END defaults to START + 1). Each catalog has its own
random seed (from os.urandom), which is saved in the file. Catalogs that already exist are skipped, so rerunning an
interrupted job finishes it. The defaults match the fiducial simulations of Quijote: 512^3 particles in a box of
1 Gpc/h, in the same cosmology. The particle spacing is fixed by the emulator (1000/512 Mpc/h), so --nside sets the
box size.

The files have the HDF5 layout of kszx.Catalog (kszx.Catalog.from_h5() reads them): one 1-d dataset per column,
with the units of kszx.quijote.read_halos():

    x, y, z       halo position (center of mass) in Mpc (not Mpc/h), in [0, box_size)
    vx, vy, vz    halo peculiar velocity (mean over its particles) / c   (not written with --no-velocities)
    mass          halo mass, npart * particle_mass, in Msun (not Msun/h)
    npart         number of particles

Halos are sorted by npart, largest first. File attributes: 'size' (the number of halos), 'name', and the parameters
of the run ('seed', 'nside', 'box_size' in Mpc, 'redshift', the cosmology, 'linking_length', ...).
"""

import argparse
import contextlib
import datetime
import io
import itertools
import os
import signal
import sys
import time

# JAX preallocates 75% of GPU memory by default. Compiling the emulator temporarily needs ~1.7x the memory it needs
# to run (XLA times several convolution algorithms), so let JAX use more. Must be set before importing jax.
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

import numpy as np
import h5py
import jax
import jax.numpy as jnp
import camb
import ictile
import jax_nbody_emulator as jne
import jztree as jz

GRID = 1000 / 512             # particle spacing [Mpc/h], fixed by the emulator's training
RHO_CRIT = 2.77536627e11      # critical density [h^2 Msun / Mpc^3]
C_KMS = 299792.458            # speed of light [km/s]
EMU_PAD = 48                  # the emulator's output depends on its input within 48 cells
FOF_PAD = 24                  # FoF padding [cells per side], so that halos near the edge of a tile are complete
MAX_TILE = 176                # above this, some of the emulator's arrays exceed 2^31 elements, and it gets 10-20x slower


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('start', type=int, help='index of the first catalog')
    ap.add_argument('end', type=int, nargs='?', help='end of the range of catalog indices (default: START + 1)')
    ap.add_argument('--outdir', default='.', help='output directory (default: the current directory)')
    ap.add_argument('--nside', type=int, default=512, help='particles per side; the box is nside * 1000/512 Mpc/h (default 512)')
    ap.add_argument('--redshift', type=float, default=0.0, help='output redshift, 0 <= z <= 3 (default 0)')
    ap.add_argument('--omega-m', type=float, default=0.3175, help='Omega_m, 0.1 to 0.5 (default 0.3175)')
    ap.add_argument('--omega-b', type=float, default=0.049, help='Omega_b (default 0.049)')
    ap.add_argument('--h', type=float, default=0.6711, help='H0 / (100 km/s/Mpc) (default 0.6711)')
    ap.add_argument('--n-s', type=float, default=0.9624, help='spectral index (default 0.9624)')
    ap.add_argument('--sigma-8', type=float, default=0.834, help='sigma_8 at z=0 (default 0.834)')
    ap.add_argument('--linking-length', type=float, default=0.18,
                    help='FoF linking length, in units of the mean interparticle spacing (default 0.18, which reproduces '
                         "the masses of Quijote's FoF b = 0.2 halos: see 01_halo_catalog.ipynb)")
    ap.add_argument('--nmin', type=int, default=20, help='minimum number of particles per halo (default 20)')
    ap.add_argument('--tile', type=int, default=128,
                    help='each tile makes the halos of a TILE^3 region; a multiple of 8 which divides nside (default 128)')
    ap.add_argument('--no-velocities', action='store_true', help="don't compute velocities (the emulator is ~2.3x faster)")
    ap.add_argument('--ics', choices=('auto', 'box', 'tiles'), default='auto',
                    help="initial conditions: 'box' generates the whole box once per catalog and cuts the tiles from it; "
                         "'tiles' generates each tile separately (for boxes that don't fit in GPU memory). The default "
                         "'auto' uses 'box' if it fits.")
    args = ap.parse_args()

    if args.end is None:
        args.end = args.start + 1
    if args.end <= args.start:
        ap.error(f'END ({args.end}) must be larger than START ({args.start})')
    if (args.tile < 8) or (args.tile % 8) or (args.nside % args.tile):
        ap.error(f'--tile={args.tile} must be a multiple of 8 which divides --nside={args.nside}')
    if not (0 <= args.redshift <= 3) or not (0.1 <= args.omega_m <= 0.5):
        ap.error('the emulator was trained for 0 <= redshift <= 3 and 0.1 <= Omega_m <= 0.5')
    return args


def linear_pk(args):
    """Returns P(k) [(Mpc/h)^3] at z=0, a vectorized function of k [h/Mpc], from CAMB, normalized to sigma_8."""
    h = args.h
    pars = camb.set_params(H0=100 * h, ombh2=args.omega_b * h**2, omch2=(args.omega_m - args.omega_b) * h**2,
                           mnu=0, ns=args.n_s, As=2e-9)
    pars.set_matter_power(redshifts=[0.0], kmax=20.0)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    k_tab, _, pk_tab = results.get_matter_power_spectrum(minkh=1e-4, maxkh=20, npoints=1000)
    pk_tab = pk_tab[0] * (args.sigma_8 / results.get_sigma8_0())**2
    lk0, lp0 = np.log(k_tab), np.log(pk_tab)

    def pk(k):
        # log-log interpolation, power-law extrapolation, P(0) = 0
        k = np.asarray(k, dtype=np.float64)
        lk = np.log(np.maximum(k, 1e-30))
        lp = np.interp(lk, lk0, lp0)
        lp = np.where(lk < lk0[0], lp0[0] + (lk - lk0[0]) * (lp0[1] - lp0[0]) / (lk0[1] - lk0[0]), lp)
        lp = np.where(lk > lk0[-1], lp0[-1] + (lk - lk0[-1]) * (lp0[-1] - lp0[-2]) / (lk0[-1] - lk0[-2]), lp)
        return np.where(k > 0, np.exp(lp), 0.0)

    return pk


class CatalogMaker:
    """Makes halo catalogs. Each tile has three nested regions (see 01_halo_catalog.ipynb):

       |<- 48 ->|<- 24 ->|<----- TILE ----->|<- 24 ->|<- 48 ->|
       |<----------- IC tile = emulator input ---------------->|      TILE + 144 cells per side
                |<------ emulator output = FoF region ->|              TILE +  48
                         |<-- halos kept -->|                          TILE
    """

    def __init__(self, args):
        self.args = args
        self.nside, self.tile = args.nside, args.tile
        self.nfof = self.tile + 2 * FOF_PAD                 # side of the FoF region (= emulator output)
        self.nin = self.nfof + 2 * EMU_PAD                  # side of the IC tile (= emulator input)
        self.ntiles = self.nside // self.tile               # tiles per side
        self.box = self.nside * GRID                        # box size [Mpc/h]
        self.m_part = args.omega_m * RHO_CRIT * GRID**3     # particle mass [Msun/h]
        self.max_halos = self.nfof**3 // 100                # tiles have ~0.003 halos per particle: a wide margin
        self.velocities = not args.no_velocities
        self.ics = self._choose_ics()
        self.pk = linear_pk(args)

        with contextlib.redirect_stdout(io.StringIO()):     # modulating the parameters prints a lot
            self.emulator = jne.create_emulator(premodulate=True, premodulate_z=args.redshift,
                                                premodulate_Om=args.omega_m, compute_vel=self.velocities)

        z, om = jnp.array([args.redshift]), jnp.array([args.omega_m])
        Dz, vel_fac = jne.growth_factor(z, om), jne.vel_norm(z, om)
        model, nin = self.emulator.model, self.nin

        # The emulator's parameters are an argument, rather than captured, so that they aren't baked into the
        # compiled program as constants.
        @jax.jit
        def emulate(params, psi_lin):
            # Linear z=0 displacement [Mpc/h], shape (3, n+96, n+96, n+96) -> nonlinear displacement [Mpc/h] and
            # velocity [km/s] (or None) at the redshift, shape (3, n, n, n).
            if self.velocities:
                psi, vel = model.apply(params, psi_lin[None], Dz, vel_fac)
                return psi[0], vel[0]
            return model.apply(params, psi_lin[None], Dz)[0], None

        @jax.jit
        def cut(psi_box, start):
            # The IC tile with lower corner 'start' (0 <= start < nside), cut from the periodic box.
            for axis in range(3):
                psi_box = jnp.take(psi_box, jnp.arange(nin) + start[axis], axis=axis + 1, mode='wrap')
            return psi_box

        self._emulate, self._cut = emulate, cut
        self._find_halos = jax.jit(self._find_halos_impl)

    def _choose_ics(self):
        fits = self.nin <= self.nside // 2      # ictile tiles are at most nside/2 (or the whole box)
        stats = jax.devices()[0].memory_stats() or {}
        # The whole-box displacement needs ~8 nside^3 floats while it's computed; leave room for the emulator.
        box_fits = 32 * self.nside**3 <= 0.3 * stats.get('bytes_limit', 0)
        if self.args.ics == 'auto':
            return 'box' if (box_fits or not fits) else 'tiles'
        if (self.args.ics == 'tiles') and not fits:
            raise RuntimeError(f'--ics=tiles needs IC tiles (TILE + 144 = {self.nin}) of at most nside/2 = {self.nside // 2}')
        return self.args.ics

    def _find_halos_impl(self, psi_nl, vel_nl):
        # psi_nl: nonlinear displacement (3, NFOF, NFOF, NFOF) [Mpc/h] on the FoF region of one tile; vel_nl: velocity
        # [km/s] or None. Returns a dict with nhalos and per-halo arrays of length MAX_HALOS (the first nhalos entries
        # are used): keep (True if this tile keeps the halo), npart, com (center of mass [Mpc/h], relative to the
        # corner of the FoF region), and vel (mean velocity [km/s]).
        n = self.nfof
        q = jnp.stack(jnp.meshgrid(*[jnp.arange(n, dtype=jnp.int32)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)
        x = q * GRID + psi_nl.reshape(3, -1).T                       # particle positions [Mpc/h], shape (P, 3)
        P = x.shape[0]
        _, label = jz.fof.fof_labels(x, rlink=self.args.linking_length * GRID, output_order='input')

        npart = jax.ops.segment_sum(jnp.ones(P, jnp.int32), label, num_segments=P)
        is_halo = npart >= self.args.nmin
        nhalos = jnp.sum(is_halo)
        rep = jnp.nonzero(is_halo, size=self.max_halos, fill_value=0)[0]   # one representative particle per halo

        # A halo is kept by the tile whose central TILE^3 region contains the midpoint of the halo's Lagrangian
        # bounding box. Every tile that finds the halo finds the same particles, so exactly one tile keeps it.
        qmin = jax.ops.segment_min(q, label, num_segments=P)[rep]
        qmax = jax.ops.segment_max(q, label, num_segments=P)[rep]
        mid = (qmin + qmax) // 2
        keep = jnp.all((mid >= FOF_PAD) & (mid < FOF_PAD + self.tile), axis=1) & (jnp.arange(self.max_halos) < nhalos)

        dx = jax.ops.segment_sum(x - x[label], label, num_segments=P)[rep]  # offsets from the representative,
        com = x[rep] + dx / npart[rep, None]                                # for float32 precision
        out = dict(nhalos=nhalos, keep=keep, npart=npart[rep], com=com)
        if vel_nl is not None:
            v = jax.ops.segment_sum(vel_nl.reshape(3, -1).T, label, num_segments=P)[rep]
            out['vel'] = v / npart[rep, None]
        return out

    def make(self, seed):
        """Returns the catalog for this seed, as a dict of columns in kszx units, sorted by npart (largest first)."""
        grf_ntile = self.nside if (self.ics == 'box') else self.nin
        grf = ictile.TiledGRF(nside=self.nside, box_size=self.box, pk=self.pk, seed=seed, ntile=grf_ntile)
        psi_box = grf.displacement_tile((0, 0, 0)) if (self.ics == 'box') else None

        cols = dict(pos=[], vel=[], npart=[])

        def collect(ijk, h):
            h = jax.device_get(h)
            if h['nhalos'] > self.max_halos:
                raise RuntimeError(f'tile {ijk}: {h["nhalos"]} halos, more than MAX_HALOS = {self.max_halos}')
            keep = h['keep']
            origin = (np.array(ijk) * self.tile - FOF_PAD) * GRID      # corner of the FoF region [Mpc/h]
            cols['pos'].append((h['com'][keep] + origin) % self.box)
            cols['npart'].append(h['npart'][keep])
            if self.velocities:
                cols['vel'].append(h['vel'][keep])

        # Each tile is dispatched (asynchronously) before the previous tile's halos are collected, so that the GPU
        # doesn't wait for the host.
        pending = None
        for ijk in itertools.product(range(self.ntiles), repeat=3):
            start = (np.array(ijk) * self.tile - (EMU_PAD + FOF_PAD)) % self.nside
            if self.ics == 'box':
                psi_lin = self._cut(psi_box, jnp.asarray(start, dtype=jnp.int32))
            else:
                psi_lin = grf.displacement_tile(start)
            h = self._find_halos(*self._emulate(self.emulator.params, psi_lin))
            if pending is not None:
                collect(*pending)
            pending = (ijk, h)
        collect(*pending)

        npart = np.concatenate(cols['npart'])
        order = np.argsort(-npart, kind='stable')
        L = self.box / self.args.h                                               # box size [Mpc]
        pos = (np.concatenate(cols['pos'])[order] / self.args.h).astype(np.float32)   # Mpc/h -> Mpc
        pos[pos.astype(np.float64) >= L] -= np.float32(L)    # rounding to float32 can give box_size, rather than < box_size
        out = dict(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2])
        if self.velocities:
            vel = np.concatenate(cols['vel'])[order] / C_KMS                     # km/s -> v/c
            out.update(vx=vel[:, 0], vy=vel[:, 1], vz=vel[:, 2])
        out['mass'] = (npart[order] * (self.m_part / self.args.h)).astype(np.float32)   # Msun
        out['npart'] = npart[order].astype(np.int32)
        return {k: np.ascontiguousarray(v, dtype=(np.int32 if k == 'npart' else np.float32)) for k, v in out.items()}

    def attrs(self, index, seed):
        a = self.args
        return dict(name=f'ictile + N-body emulator FoF halos, catalog {index}', catalog_index=index, seed=np.uint64(seed),
                    nside=a.nside, box_size=self.box / a.h, redshift=a.redshift, omega_m=a.omega_m, omega_b=a.omega_b,
                    h=a.h, n_s=a.n_s, sigma_8=a.sigma_8, linking_length=a.linking_length, nmin=a.nmin,
                    particle_mass=self.m_part / a.h, tile=a.tile, velocities=self.velocities,
                    units='x,y,z: Mpc; vx,vy,vz: peculiar velocity / c; mass: Msun (no factors of h); box_size: Mpc; '
                          'particle_mass: Msun',
                    ictile_version=ictile.__version__, jax_version=jax.__version__,
                    created=datetime.datetime.now().isoformat(timespec='seconds'), command=' '.join(sys.argv))


def write_catalog(filename, cols, attrs):
    """Writes a catalog in the HDF5 layout of kszx.Catalog, via a temporary file, so that an interrupted write
    doesn't leave a file that looks complete."""
    tmp = os.path.join(os.path.dirname(filename), f'.{os.path.basename(filename)}.tmp{os.getpid()}')
    try:
        with h5py.File(tmp, 'w') as f:
            f.attrs['size'] = len(cols['npart'])
            for k, v in attrs.items():
                f.attrs[k] = v
            for k, v in cols.items():
                f.create_dataset(k, data=v)
        os.replace(tmp, filename)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    args = parse_args()
    # Slurm sends SIGTERM when it preempts or cancels a job. Exiting normally removes the temporary file of an
    # unfinished catalog (see write_catalog()).
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(f'{sys.argv[0]}: SIGTERM received, exiting'))
    os.makedirs(args.outdir, exist_ok=True)
    filename = lambda i: os.path.join(args.outdir, f'halo_catalog_{i}.h5')
    todo = [i for i in range(args.start, args.end) if not os.path.exists(filename(i))]
    print(f'{len(todo)} of {args.end - args.start} catalogs to make in {args.outdir} '
          f'(halo_catalog_{{i}}.h5 for i in range({args.start}, {args.end})), the others exist', flush=True)
    if not todo:
        return

    t0 = time.time()
    maker = CatalogMaker(args)
    print(f'GPU: {jax.devices()[0].device_kind} | nside {args.nside} ({maker.box:.0f} Mpc/h box), z = {args.redshift}, '
          f'{maker.ntiles}^3 tiles of {args.tile}^3, initial conditions: {maker.ics}, velocities: {maker.velocities} '
          f'| setup {time.time() - t0:.0f} s', flush=True)
    if args.tile > MAX_TILE:
        print(f'WARNING: --tile={args.tile} > {MAX_TILE}: the emulator may be 10-20x slower', flush=True)

    for n, i in enumerate(todo):
        if os.path.exists(filename(i)):          # e.g. made by another job since we started
            continue
        t = time.time()
        seed = int.from_bytes(os.urandom(8), 'little')
        cols = maker.make(seed)
        write_catalog(filename(i), cols, maker.attrs(i, seed))
        print(f'{filename(i)}: {len(cols["npart"])} halos, most massive {cols["mass"][0]:.3g} Msun, '
              f'{time.time() - t:.1f} s' + (' (including compilation)' if n == 0 else ''), flush=True)
    print(f'done: {len(todo)} catalogs in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
