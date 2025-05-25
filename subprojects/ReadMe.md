# Tlaxcaltin 🌮: A Meson Subproject Manager

_Tlaxcaltin_ (plural of Nahuatl _tlaxcalli_, “tortilla”) is a collection of wraps and corresponding package files for use in other projects, together with tools to manage the subprojects used.
It is designed to be used as the source of all subprojects which a Meson-based project uses and can be integrated into a project using the following steps:

1. Add a file named `subprojects.txt` to the root folder of the project and add the names of the required subprojects to that file. These names correspond to the wrap files in the root directory of this repository without the suffix (i.e. the _stem_ of the file path). The dependencies of these projects are determined by Tlaxcaltin and included as well.  
  If there is no file named `subprojects.txt` in the project’s root folder, all of Tlaxcaltin’s subprojects are included, which is usually not desirable.
2. Copy `update_tlaxcaltin.py` into a subfolder named `subprojects` within the project’s root folder and execute `python3 subprojects/update_tlaxcaltin.py`.

To add new subprojects to a project or to update the existing subprojects, edit `subprojects.txt` (if required) and execute `python3 subprojects/update_tlaxcaltin.py` again.

Note that this adds the selected Meson wraps and package files, as well as other required Tlaxcaltin files, to the main projects source tree.
Attempts to use Tlaxcaltin as a Git subproject have failed in the past, necessitating this suboptimal solution.

## Licence

Tlaxcaltin is licenced under the terms of the Mozilla Public Licence 2.0, which is provided in [`License`](License).
There are some wraps package files which are based on Meson wraps from the [Meson Wrap Database](https://github.com/mesonbuild/wrapdb), which are licenced under the MIT licence, which is provided as [`LicenseWrapDB`](LicenseWrapDB).
These are:

- `fmt`
- `google-benchmark`
- `liblzma`
- `nlohmann-json`
