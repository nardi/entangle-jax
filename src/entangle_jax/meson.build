jax_ffi_include_dir = run_command(python,
  ['-c', 'import jax; print(jax.ffi.include_dir())'],
  check: true,
).stdout().strip()

jax_ffi_dependency = declare_dependency(
  compile_args: ['-I' + jax_ffi_include_dir],
)

entangle_ffi_core = static_library('entangle_ffi_core',
  '_entangle_ffi.cc',
  '_testing_ffi.cc',
  dependencies: [jax_ffi_dependency],
)

python.extension_module('_ffi',
  '_ffi.pyx',
  link_with: entangle_ffi_core,
  override_options: ['cython_language=cpp'],
  install: true,
  subdir: 'entangle_jax',
)

python.install_sources(
  [
    '__init__.py',
    '_ffi.pyi',
    '_primitive.py',
    '_testing.py',
    'py.typed',
  ],
  subdir: 'entangle_jax',
)
