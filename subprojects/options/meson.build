# This file is part of https://github.com/KurtBoehm/tlaxcaltin.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

project('meson-options')

# The warnings are based on those listed here:
# https://github.com/lefticus/cppbestpractices/blob/master/02-Use_the_Tools_Available.md

msvc_warnings = [
  # Enforce standards conformance.
  '/permissive-',
  # All reasonable warnings.
  '/W4',
  # 'identifier': Conversion from 'type1' to 'type1', possible loss of data.
  '/w14242',
  # 'operator': Conversion from 'type1:field_bits' to 'type2:field_bits', possible loss of data.
  '/w14254',
  # 'function': Member function does not override any base class virtual member function.
  '/w14263',
  # 'classname': Class has virtual functions, but destructor is not virtual.
  #              Instances of this class may not be destructed correctly.
  '/w14265',
  # 'operator': Unsigned/negative constant mismatch.
  '/w14287',
  # 'variable': loop control variable declared in the for-loop is used outside the for-loop scope.
  '/we4289',
  # 'operator': expression is always 'boolean_value'.
  '/w14296',
  # 'variable': pointer truncation from 'type1' to 'type2'.
  '/w14311',
  # Expression before comma evaluates to a function which is missing an argument list.
  '/w14545',
  # Function call before comma missing argument list.
  '/w14546',
  # 'operator': Operator before comma has no effect; expected operator with side-effect.
  '/w14547',
  # 'operator': Operator before comma has no effect; did you intend 'operator'?.
  '/w14549',
  # Expression has no effect; expected expression with side-effect.
  '/w14555',
  # pragma warning: There is no warning number 'number'.
  '/w14619',
  # Enable warning on thread unsafe static member initialization.
  '/w14640',
  # Conversion from 'type1' to 'type_2' is sign-extended.
  # This may cause unexpected runtime behavior.
  '/w14826',
  # Wide string literal cast to 'LPSTR'.
  '/w14905',
  # String literal cast to 'LPWSTR'.
  '/w14906',
  # Illegal copy-initialization; more than one user-defined conversion has been implicitly applied.
  '/w14928',
]
clang_warnings = [
  # Reasonable and standard.
  '-Wall',
  # Reasonable and standard.
  '-Wextra',
  # Warn if non-standard C++ is used.
  '-Wpedantic',
  # Warn the user if a variable declaration shadows one from a parent context.
  '-Wshadow',
  # Warn the user if a class with virtual functions has a non-virtual destructor.
  # This helps catch hard to track down memory errors.
  '-Wnon-virtual-dtor',
  # Warn for c-style casts.
  '-Wold-style-cast',
  # Warn for potential performance problem casts.
  '-Wcast-align',
  # Warn on anything being unused.
  '-Wunused',
  # Warn if you overload (not override) a virtual function.
  '-Woverloaded-virtual',
  # Warn on type conversions that may lose data.
  '-Wconversion',
  # Warn on sign conversions.
  '-Wsign-conversion',
  # Warn if indentation implies blocks where blocks do not exist.
  '-Wmisleading-indentation',
  # Warn if a null dereference is detected.
  '-Wnull-dereference',
  # Warn if float is implicitly promoted to double.
  '-Wdouble-promotion',
  # Warn on security issues around functions that format output (i.e. printf).
  '-Wformat=2',
  # Warn when case statements fall-through.
  '-Wimplicit-fallthrough',
]
gcc_warnings = clang_warnings + [
  # Warn if if/else chain has duplicated conditions.
  '-Wduplicated-cond',
  # Warn if if/else branches have duplicated code.
  '-Wduplicated-branches',
  # Warn about logical operations being used where bitwise were probably wanted.
  '-Wlogical-op',
  # Warn if you perform a cast to the same type.
  '-Wuseless-cast',
]

msvc_info = []
clang_info = [
  # Print all template instantiation notes.
  '-ftemplate-backtrace-limit=0',
]
gcc_info = [
  # Print more information about unfulfilled concepts.
  '-fconcepts-diagnostics-depth=3',
]

optimization = get_option('optimization')
warning_level = get_option('warning_level')

compiler = meson.get_compiler('cpp')
compiler_id = compiler.get_id()
is_gcc_compatible = ['gcc', 'clang'].contains(compiler_id)

# Determine the appropriate warnings
warning_args = []
info_args = []
if compiler_id == 'clang'
  warning_args += clang_warnings
  info_args += clang_info
endif
if compiler_id == 'gcc'
  warning_args += gcc_warnings
  info_args += gcc_info
endif
if compiler_id == 'msvc'
  warning_args += msvc_warnings
  info_args += msvc_info
endif

# Add warnings on sufficient warning levels
opt_warning_args = []
if warning_level == '3'
  opt_warning_args += warning_args
endif

experimental_args = []
# Add experimental flags if clang is used
if compiler_id == 'clang'
  major_version = compiler.version().split('.')[0].to_int()
  if major_version > 14
    experimental_args += ['-fexperimental-library']
  endif
endif

# Determine architecture-specific optimizations
optimization_args = []
if is_gcc_compatible and compiler.compiles('int dummy;', name: 'Check -march=native', args: ['-march=native'])
  optimization_args += ['-march=native']
endif

# Add architecture-specific optimizations
opt_optimization_args = []
if optimization == '3'
  opt_optimization_args += optimization_args
endif

# All arguments
all_args = experimental_args + info_args + opt_warning_args + opt_optimization_args
