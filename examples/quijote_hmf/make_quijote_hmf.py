"""Halo counts from the Quijote N-body simulations, for the mass-function comparison in ../01_halo_catalog.ipynb.

Quijote (Villaescusa-Navarro et al. 2020, arXiv:1909.05273, https://quijote-simulations.readthedocs.io) has many
realizations of its fiducial cosmology (Omega_m = 0.3175, Omega_b = 0.049, h = 0.6711, n_s = 0.9624, sigma_8 = 0.834),
each with 512^3 particles in a 1 Gpc/h box: the resolution the emulator was trained at. This script reads the z = 0
FoF (b = 0.2) catalogs of the first NREAL realizations, and writes the number of halos with each particle count,
summed over the realizations, to quijote_fof_z0.txt (next to this script, unless --out is given).

The catalogs are on the Quijote Globus collection, at /Halos/FoF/fiducial/<i>/groups_004/group_tab_004.0 (34 MB
each). With --download, they are fetched with the Globus CLI first, which needs:
  - pip install globus-cli, then: globus login
  - Globus Connect Personal running on this machine (https://www.globus.org/globus-connect-personal), with DIR
    inside a directory it can write to; LOCAL_ENDPOINT is its endpoint ID (globus endpoint local-id).

Usage:
  python make_quijote_hmf.py DIR --download LOCAL_ENDPOINT [--nreal 64]    # download into DIR, then count
  python make_quijote_hmf.py DIR [--nreal 64]                               # DIR/<i>/group_tab_004.0 already there
"""

import argparse
import datetime
import os
import subprocess

import numpy as np

QUIJOTE_COLLECTION = 'e0eae0aa-5bca-11ea-9683-0e56c063f437'    # "Quijote_simulations2" Globus collection
REMOTE = '/Halos/FoF/fiducial/{i}/groups_004/group_tab_004.0'
LOCAL = '{i}/group_tab_004.0'
BOX_SIZE = 1000.0       # Mpc/h

# Gadget "group_tab" format: header, then per-group arrays (see e.g. Pylians' readfof.FoF_catalog).
HEADER = np.dtype([('Ngroups', '<i4'), ('TotNgroups', '<i4'), ('Nids', '<i4'), ('TotNids', '<i8'), ('Nfiles', '<i4')])


def read_group_tab(filename):
    """Returns (number of particles, mass [Msun/h]) of each group in a single-file group_tab catalog."""
    with open(filename, 'rb') as f:
        h = np.fromfile(f, dtype=HEADER, count=1)[0]
        n = int(h['Ngroups'])
        if (h['Nfiles'] != 1) or (h['TotNgroups'] != n) or (os.path.getsize(filename) != HEADER.itemsize + 84 * n):
            raise ValueError(f'{filename}: unexpected header {h} or file size')
        npart = np.fromfile(f, dtype='<i4', count=n)
        np.fromfile(f, dtype='<i4', count=n)                                  # GroupOffset
        mass = np.fromfile(f, dtype='<f4', count=n).astype(np.float64) * 1e10  # GroupMass [1e10 Msun/h]
    return npart, mass


def download(dirname, local_endpoint, nreal):
    """Globus transfer of the catalogs that aren't in dirname yet."""
    todo = [i for i in range(nreal) if not os.path.exists(os.path.join(dirname, LOCAL.format(i=i)))]
    if not todo:
        return
    batch = ''.join(f'{REMOTE.format(i=i)} {os.path.join(os.path.abspath(dirname), LOCAL.format(i=i))}\n' for i in todo)
    print(f'Submitting a Globus transfer of {len(todo)} catalogs to {local_endpoint}:{os.path.abspath(dirname)}')
    task = subprocess.run(['globus', 'transfer', QUIJOTE_COLLECTION, local_endpoint, '--batch', '-', '--label', 'Quijote FoF catalogs',
                           '--jmespath', 'task_id', '--format', 'unix'], input=batch, text=True, capture_output=True)
    if task.returncode != 0:
        raise RuntimeError(f'globus transfer failed: {task.stderr.strip()}')
    task_id = task.stdout.strip()
    print(f'Waiting for Globus task {task_id} (globus task show {task_id})')
    subprocess.run(['globus', 'task', 'wait', task_id, '--polling-interval', '10'], check=True)
    status = subprocess.run(['globus', 'task', 'show', task_id, '--jmespath', 'status', '--format', 'unix'],
                            capture_output=True, text=True).stdout.strip()
    if status != 'SUCCEEDED':
        raise RuntimeError(f'Globus task {task_id} ended with status {status}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('dir', help='directory for the catalogs (DIR/<i>/group_tab_004.0)')
    ap.add_argument('--download', metavar='LOCAL_ENDPOINT', help='download missing catalogs with the Globus CLI to this endpoint')
    ap.add_argument('--nreal', type=int, default=64, help='number of realizations (default 64)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quijote_fof_z0.txt'))
    args = ap.parse_args()

    if args.download:
        download(args.dir, args.download, args.nreal)

    counts, mp = {}, []
    for i in range(args.nreal):
        npart, mass = read_group_tab(os.path.join(args.dir, LOCAL.format(i=i)))
        for n, c in zip(*np.unique(npart, return_counts=True)):
            counts[int(n)] = counts.get(int(n), 0) + int(c)
        mp.append(np.median(mass / npart))
    mp = float(np.median(mp))
    n = np.array(sorted(counts))
    c = np.array([counts[k] for k in n])

    with open(args.out, 'w') as f:
        f.write(f'# Quijote FoF (b = 0.2) halos at z = 0: the number of halos with each particle count, summed over\n'
                f'# realizations 0..{args.nreal - 1} of the fiducial cosmology (Omega_m = 0.3175, Omega_b = 0.049, h = 0.6711,\n'
                f'# n_s = 0.9624, sigma_8 = 0.834), each with 512^3 particles in a box of {BOX_SIZE:g} Mpc/h.\n'
                f'# Halo mass = npart * particle_mass. Source: Quijote Globus collection {QUIJOTE_COLLECTION},\n'
                f'# {REMOTE.format(i="<i>")}. Written by make_quijote_hmf.py on {datetime.date.today()}.\n'
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
