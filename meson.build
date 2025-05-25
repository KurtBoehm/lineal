project('lineal', 'cpp', default_options: ['cpp_std=c++20', 'warning_level=3'])

deps = [
  dependency('stado'),
  dependency('thesauros'),
  dependency('unordered_dense'),
]
args = []

lineal_dep = declare_dependency(
  include_directories: include_directories('include'),
  dependencies: deps,
  compile_args: args,
)

install_subdir(
  'include',
  install_dir: 'include',
)

pc = import('pkgconfig')
pc.generate(
  name: 'lineal',
  version: meson.project_version(),
  description: 'Generic linear algebra',
)

if get_option('test')
  subdir('test')
endif
