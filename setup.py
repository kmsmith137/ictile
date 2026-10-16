"""Builds the C++ and CUDA kernels as plain shared libraries, which ictile loads with ctypes
and registers as XLA FFI targets (they are not Python extension modules).

Environment variables:
  ICTILE_NVCC          path to nvcc (default: $CUDA_HOME/bin/nvcc, then nvcc on PATH, then /usr/local/cuda/bin/nvcc)
  ICTILE_CUDA_ARCH     semicolon-separated list of compute capabilities (default "89;90")
  ICTILE_NO_CUDA=1     skip the CUDA kernel
  ICTILE_CXXFLAGS      extra flags for the C++ compiler (default "-O3 -march=native")
"""

import os
import shutil
import subprocess
import sys
import tempfile

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

CSRC = os.path.join('ictile', 'csrc')


def ffi_include_dir():
    try:
        import jax.ffi
        return jax.ffi.include_dir()
    except ImportError:
        import jaxlib
        return os.path.join(os.path.dirname(jaxlib.__file__), 'include')


def find_nvcc():
    candidates = [os.environ.get('ICTILE_NVCC')]
    for var in ('CUDA_HOME', 'CUDA_PATH'):
        if os.environ.get(var):
            candidates.append(os.path.join(os.environ[var], 'bin', 'nvcc'))
    candidates += [shutil.which('nvcc'), '/usr/local/cuda/bin/nvcc']

    # Some installations (e.g. a partial conda toolkit) have an nvcc which can't compile device
    # code, so we test each candidate.
    for nvcc in candidates:
        if nvcc and os.path.exists(nvcc) and nvcc_works(nvcc):
            return nvcc
    return None


def nvcc_works(nvcc):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 't.cu')
        with open(src, 'w') as f:
            f.write('__global__ void k(float *x) { x[0] = 1.0f; }\n')
        r = subprocess.run([nvcc, '-c', src, '-o', os.path.join(d, 't.o')], capture_output=True)
        return r.returncode == 0


class SharedLibrary(Extension):
    def __init__(self, name, sources, kind):
        super().__init__(name, sources=[])
        self.lib_sources = sources
        self.kind = kind


class BuildSharedLibraries(build_ext):
    def get_ext_filename(self, fullname):
        # Plain library name (e.g. ictile/libictile_cpu.so), not a CPython ABI-tagged name.
        parts = fullname.split('.')
        return os.path.join(*parts[:-1], 'lib' + parts[-1] + '.so')

    def build_extension(self, ext):
        out = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        deps = ext.lib_sources + [os.path.join(CSRC, 'grf_common.h')]
        if (not self.force) and os.path.exists(out) and all(os.path.getmtime(d) <= os.path.getmtime(out) for d in deps):
            print(f'ictile: {out} is up to date')
            return

        inc = ['-isystem', ffi_include_dir(), '-I', CSRC]

        if ext.kind == 'cpu':
            cxx = os.environ.get('CXX', 'g++')
            flags = os.environ.get('ICTILE_CXXFLAGS', '-O3 -march=native').split()
            cmd = [cxx, *flags, '-std=c++17', '-fopenmp-simd', '-fPIC', '-shared', *inc, *ext.lib_sources, '-o', out]
        else:
            nvcc = find_nvcc()
            if nvcc is None:
                print('ictile: no working nvcc found, skipping CUDA kernel (set ICTILE_NVCC to override)', file=sys.stderr)
                return
            archs = os.environ.get('ICTILE_CUDA_ARCH', '89;90').split(';')
            gencode = []
            for a in archs:
                gencode += ['-gencode', f'arch=compute_{a},code=sm_{a}']
            gencode += ['-gencode', f'arch=compute_{archs[-1]},code=compute_{archs[-1]}']   # PTX for newer GPUs
            cmd = [nvcc, '-O3', '-std=c++17', *gencode, '-diag-suppress', '940,2473',
                   '-Xcompiler', '-fPIC', '-shared', *inc, *ext.lib_sources, '-o', out]

        print(' '.join(cmd))
        subprocess.check_call(cmd)


ext_modules = [SharedLibrary('ictile.ictile_cpu', [os.path.join(CSRC, 'grf_cpu.cc')], 'cpu')]
if os.environ.get('ICTILE_NO_CUDA', '0') != '1':
    ext_modules.append(SharedLibrary('ictile.ictile_cuda', [os.path.join(CSRC, 'grf_cuda.cu')], 'cuda'))

setup(ext_modules=ext_modules, cmdclass={'build_ext': BuildSharedLibraries})
