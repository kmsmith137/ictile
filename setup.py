"""Builds the C++ and CUDA kernels as plain shared libraries, which ictile loads with ctypes
and registers as XLA FFI targets (they are not Python extension modules).

Environment variables:
  ICTILE_CUDA          whether to build the CUDA kernel:
                         auto (default): build it if a working nvcc is found, otherwise skip it with a warning;
                         1: required, the build fails if there is no working nvcc or the CUDA build fails;
                         0: don't build it (CPU only).
  ICTILE_NVCC          path to nvcc. If set, it must work. Otherwise: $CUDA_HOME/bin/nvcc, $CUDA_PATH/bin/nvcc,
                       nvcc on PATH, /usr/local/cuda/bin/nvcc (the first that works).
  ICTILE_CUDA_ARCH     semicolon-separated list of compute capabilities, e.g. "80;90" (default "80;86;89;90;100;120",
                       minus any that nvcc doesn't support), plus PTX for the last one (for newer GPUs). Can also
                       be one of nvcc's "native", "all", "all-major".
  ICTILE_CXXFLAGS      extra flags for the C++ compiler (default "-O3 -march=native")

The build configuration is recorded in ictile/_build_info.json (see ictile.build_info()). A library is rebuilt
if its sources are newer, or if its configuration (compiler, flags, architectures) changed.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

CSRC = os.path.join('ictile', 'csrc')
DEFAULT_CUDA_ARCHS = ['80', '86', '89', '90', '100', '120']
BUILD_INFO = '_build_info.json'


def ffi_include_dir():
    try:
        import jax.ffi
        return jax.ffi.include_dir()
    except ImportError:
        import jaxlib
        return os.path.join(os.path.dirname(jaxlib.__file__), 'include')


def cuda_mode():
    mode = os.environ.get('ICTILE_CUDA', 'auto').strip().lower()
    if mode not in ('auto', '1', '0'):
        raise RuntimeError(f"ictile: ICTILE_CUDA={mode!r} must be 'auto', '1' or '0'")
    return mode


def find_nvcc():
    """Returns (path to a working nvcc, None), or (None, reason)."""
    if os.environ.get('ICTILE_NVCC'):
        nvcc = os.environ['ICTILE_NVCC']
        if not (os.path.exists(nvcc) and nvcc_works(nvcc)):
            raise RuntimeError(f'ictile: ICTILE_NVCC={nvcc} does not exist or cannot compile CUDA code')
        return nvcc, None

    candidates = [os.path.join(os.environ[var], 'bin', 'nvcc') for var in ('CUDA_HOME', 'CUDA_PATH') if os.environ.get(var)]
    candidates += [shutil.which('nvcc'), '/usr/local/cuda/bin/nvcc']

    # Some installations (e.g. a partial conda toolkit) have an nvcc which can't compile device
    # code, so we test each candidate.
    tried = []
    for nvcc in candidates:
        if nvcc and os.path.exists(nvcc):
            if nvcc_works(nvcc):
                return nvcc, None
            tried.append(nvcc)
    if tried:
        return None, f'nvcc ({", ".join(tried)}) cannot compile CUDA code'
    return None, 'no nvcc was found (set ICTILE_NVCC, or put nvcc on PATH)'


def nvcc_works(nvcc):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 't.cu')
        with open(src, 'w') as f:
            f.write('__global__ void k(float *x) { x[0] = 1.0f; }\n')
        r = subprocess.run([nvcc, '-c', src, '-o', os.path.join(d, 't.o')], capture_output=True)
        return r.returncode == 0


def nvcc_version(nvcc):
    r = subprocess.run([nvcc, '--version'], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if 'release' in l]
    return lines[-1].strip() if lines else r.stdout.strip()


def cuda_archs(nvcc):
    """Returns the list of architectures to build (or [native|all|all-major])."""
    if os.environ.get('ICTILE_CUDA_ARCH'):
        archs = [a.strip() for a in os.environ['ICTILE_CUDA_ARCH'].split(';') if a.strip()]
        if not archs:
            raise RuntimeError('ictile: ICTILE_CUDA_ARCH is empty')
        return archs

    # Default: drop architectures that this nvcc doesn't support (e.g. 100 and 120 need CUDA >= 12.8).
    r = subprocess.run([nvcc, '--list-gpu-arch'], capture_output=True, text=True)
    if r.returncode != 0:
        return DEFAULT_CUDA_ARCHS
    supported = {s.strip().removeprefix('compute_') for s in r.stdout.split()}
    archs = [a for a in DEFAULT_CUDA_ARCHS if a in supported]
    dropped = [a for a in DEFAULT_CUDA_ARCHS if a not in supported]
    if not archs:
        raise RuntimeError(f'ictile: {nvcc} supports none of the default architectures {DEFAULT_CUDA_ARCHS}; set ICTILE_CUDA_ARCH')
    if dropped:
        print(f'ictile: {nvcc} does not support compute capabilities {";".join(dropped)}, building for {";".join(archs)}',
              file=sys.stderr)
    return archs


def gencode_flags(archs):
    if len(archs) == 1 and archs[0] in ('native', 'all', 'all-major'):
        return [f'-arch={archs[0]}']
    flags = []
    for a in archs:
        flags += ['-gencode', f'arch=compute_{a},code=sm_{a}']
    return flags + ['-gencode', f'arch=compute_{archs[-1]},code=compute_{archs[-1]}']   # PTX for newer GPUs


def read_build_info(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_build_info(path, key, value):
    info = read_build_info(path)
    info[key] = value
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
        f.write('\n')


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
        info_path = os.path.join(os.path.dirname(out), BUILD_INFO)
        inc = ['-isystem', ffi_include_dir(), '-I', CSRC]

        if ext.kind == 'cpu':
            key = 'cpu'
            cxx = os.environ.get('CXX', 'g++')
            flags = os.environ.get('ICTILE_CXXFLAGS', '-O3 -march=native').split()
            cmd = [cxx, *flags, '-std=c++17', '-fopenmp-simd', '-fPIC', '-shared', *inc, *ext.lib_sources, '-o', out]
            config = dict(built=True, compiler=cxx, flags=flags)
        else:
            key = 'cuda'
            mode = cuda_mode()
            if mode == '0':
                print('ictile: ICTILE_CUDA=0, not building the CUDA kernel', file=sys.stderr)
                self.skip_cuda(out, info_path, 'ictile was built with ICTILE_CUDA=0')
                return
            nvcc, reason = find_nvcc()
            if nvcc is None:
                if mode == '1':
                    raise RuntimeError(f'ictile: ICTILE_CUDA=1, but the CUDA kernel cannot be built: {reason}')
                print('*' * 100, file=sys.stderr)
                print(f'ictile: WARNING: not building the CUDA kernel, because {reason}.', file=sys.stderr)
                print('ictile: ictile will only run on the CPU. Set ICTILE_CUDA=1 to make this an error.', file=sys.stderr)
                print('*' * 100, file=sys.stderr)
                self.skip_cuda(out, info_path, f'ictile was built without the CUDA kernel, because {reason}')
                return
            archs = cuda_archs(nvcc)
            cmd = [nvcc, '-O3', '-std=c++17', *gencode_flags(archs), '-diag-suppress', '940,2473',
                   f'-DICTILE_CUDA_ARCHS={"_".join(archs)}',     # one preprocessor token, stringified in grf_cuda.cu
                   '-Xcompiler', '-fPIC', '-shared', *inc, *ext.lib_sources, '-o', out]
            config = dict(built=True, nvcc=nvcc, nvcc_version=nvcc_version(nvcc), archs=archs)

        deps = ext.lib_sources + [os.path.join(CSRC, 'grf_common.h')]
        up_to_date = (os.path.exists(out) and all(os.path.getmtime(d) <= os.path.getmtime(out) for d in deps)
                      and read_build_info(info_path).get(key) == config)
        if up_to_date and not self.force:
            print(f'ictile: {out} is up to date')
        else:
            print(' '.join(cmd))
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError:
                if key == 'cuda':
                    print('ictile: building the CUDA kernel failed. Set ICTILE_CUDA=0 to build ictile without it (CPU only).',
                          file=sys.stderr)
                raise
        write_build_info(info_path, key, config)

    def skip_cuda(self, out, info_path, reason):
        # Remove any CUDA library left over from an earlier build, so that what is installed matches the build info.
        if os.path.exists(out):
            os.remove(out)
        write_build_info(info_path, 'cuda', dict(built=False, reason=reason))

    def copy_extensions_to_source(self):
        # In-place and editable builds compile into build/ and then copy the libraries into the source tree.
        # Unlike setuptools' version, this also copies the build info, and removes a library from the source
        # tree if this build skipped it (rather than failing to copy it).
        build_py = self.get_finalized_command('build_py')
        pkg_dir = build_py.get_package_dir('ictile')
        for ext in self.extensions:
            filename = self.get_ext_filename(self.get_ext_fullname(ext.name))
            built, inplace = os.path.join(self.build_lib, filename), os.path.join(pkg_dir, os.path.basename(filename))
            if os.path.exists(built):
                self.copy_file(built, inplace, level=self.verbose)
            elif os.path.exists(inplace):
                os.remove(inplace)
        self.copy_file(os.path.join(self.build_lib, 'ictile', BUILD_INFO), os.path.join(pkg_dir, BUILD_INFO), level=self.verbose)


ext_modules = [SharedLibrary('ictile.ictile_cpu', [os.path.join(CSRC, 'grf_cpu.cc')], 'cpu'),
               SharedLibrary('ictile.ictile_cuda', [os.path.join(CSRC, 'grf_cuda.cu')], 'cuda')]

setup(ext_modules=ext_modules, cmdclass={'build_ext': BuildSharedLibraries})
