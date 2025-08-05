#!/bin/bash
f2py=f2py
UTILS_DIR="$(dirname "$0")/utils"
cd "$UTILS_DIR" || exit 1

# Check if meson is available, if not use distutils backend
if ! command -v meson &> /dev/null; then
    echo "Meson not found, using distutils backend..."
    export NPY_DISTUTILS_APPEND_FLAGS=1
    $f2py -c --opt='-O3' --backend=distutils fill_power_spectra.f90 -m pow_spec
    $f2py -c --opt='-O3' --backend=distutils combine_spectra.f90 -m com_spe
else
    echo "Using meson backend..."
    $f2py -c --opt='-O3' fill_power_spectra.f90 -m pow_spec
    $f2py -c --opt='-O3' combine_spectra.f90 -m com_spe
fi
