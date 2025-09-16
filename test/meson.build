options_sub = subproject('options')
args = (
  options_sub.get_variable('all_args')
  + options_sub.get_variable(
    'optimization_args',
  )
)
# args = args + ['-Wconversion']
if options_sub.get_variable('compiler_id') == 'gcc'
  args = args + ['-Wno-maybe-uninitialized']
endif

arches = ['native']
if get_option('test_simd')
  python3_prg = import('python').find_installation('python3')
  arches = run_command(
    python3_prg,
    files('tools/arch.py'),
    check: true,
  ).stdout().split()
  arches += ['native']
endif

test_deps = [
  lineal_dep,
  dependency('argparse'),
  dependency('pcg-cpp'),
]

tests = {
  'shared-fixed': [],
  'shared-csr': [],
  'shared-stencil-csr-compressed': [],
  'shared-stencil-csr-full': [],
  'shared-stencil-lookup': [],
  'shared-stencil-value': [],
  'shared-vector': [],
  'tensor-io': [],
}

foreach name, dummy : tests
  foreach arch : arches
    exe = executable(
      f'test-@name@-@arch@',
      [f'@name@.cpp'],
      cpp_args: args + [f'-march=@arch@'],
      dependencies: test_deps,
    )

    space_name = name.replace('-', ' ')
    test_name = f'test @space_name@ @arch@'
    test(test_name, exe)
  endforeach
endforeach

foreach name : ['UFZ_CT_02_LHS.smf', 'UFZ_CT_02_RHS.dvf']
  configure_file(copy: true, input: 'data/' + name, output: name)
endforeach
