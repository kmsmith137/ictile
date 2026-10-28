"""Halo counts from the Quijote N-body simulations, for the mass-function comparison in ../01_halo_catalog.ipynb.

Quijote (Villaescusa-Navarro et al. 2020, arXiv:1909.05273, https://quijote-simulations.readthedocs.io) has many
realizations of its fiducial cosmology (Omega_m = 0.3175, Omega_b = 0.049, h = 0.6711, n_s = 0.9624, sigma_8 = 0.834),
each with 512^3 particles in a 1 Gpc/h box: the resolution the emulator was trained at. This script reads the z = 0
FoF (b = 0.2) catalogs of the first NREAL realizations, and writes the number of halos with each particle count,
summed over the realizations, to quijote_fof_z0.txt (next to this script, unless --out is given).

The catalogs are read with kszx (https://github.com/kmsmith137/kszx). Download them first (34 MB each; needs Globus,
see kszx/globus_utils.py):

  python -c "import kszx; kszx.quijote.download('fiducial', 64, products=('halos',))"
  python make_quijote_hmf.py [--nreal 64]
"""

import argparse
import datetime
import os

import numpy as np
import kszx

H = 0.6711              # Quijote's fiducial cosmology
BOX_SIZE = 1000.0       # Mpc/h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--nreal', type=int, default=64, help='number of realizations (default 64)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quijote_fof_z0.txt'))
    args = ap.parse_args()

    counts, mp = {}, []
    for i in range(args.nreal):
        cat = kszx.quijote.read_halos('fiducial', i, 0.0)
        npart = cat.npart
        for n, c in zip(*np.unique(npart, return_counts=True)):
            counts[int(n)] = counts.get(int(n), 0) + int(c)
        mp.append(np.median(cat.mass * H / npart))       # particle mass [Msun/h]
    mp = float(np.median(mp))
    n = np.array(sorted(counts))
    c = np.array([counts[k] for k in n])

    with open(args.out, 'w') as f:
        f.write(f'# Quijote FoF (b = 0.2) halos at z = 0: the number of halos with each particle count, summed over\n'
                f'# realizations 0..{args.nreal - 1} of the fiducial cosmology (Omega_m = 0.3175, Omega_b = 0.049, h = 0.6711,\n'
                f'# n_s = 0.9624, sigma_8 = 0.834), each with 512^3 particles in a box of {BOX_SIZE:g} Mpc/h.\n'
                f'# Halo mass = npart * particle_mass. Source: Quijote Globus collection {kszx.quijote.QUIJOTE_EP2},\n'
                f'# /Halos/FoF/fiducial/<i>/groups_004/group_tab_004.0. Written by make_quijote_hmf.py on {datetime.date.today()}.\n'
                f'# nreal = {args.nreal}\n'
                f'# box_size = {BOX_SIZE:g}\n'
                f'# particle_mass = {mp:.6g}\n'
                f'# npart nhalos\n')
        for k, v in zip(n, c):
            f.write(f'{k} {v}\n')
    print(f'wrote {args.out}: {c.sum()} halos with {n.min()} to {n.max()} particles, from {args.nreal} realizations; '
          f'particle mass {mp:.6g} Msun/h')


if __name__ == '__main__':
    main()
