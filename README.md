# SA-GPR: Symmetry-Adapted Gaussian Process Regression

A Python implementation for machine learning of tensorial properties of atomistic systems.

## Reference

Andrea Grisafi, David M. Wilkins, Gabor Csányi, Michele Ceriotti, "Symmetry-Adapted Machine Learning for Tensorial Properties of Atomistic Systems", *Phys. Rev. Lett.* **120**, 036002 (2018)

## Features

- Efficient kernel calculation and regression for tensorial properties
- Easy installation with standard Python scientific libraries
- Full compatibility with original functionality
- Cross-platform: Works on any system with Python
- Uses optimized NumPy operations for computational performance

## Requirements

The following Python packages are required:

- `numpy` - For numerical computations
- `scipy` - For scientific computing functions
- `sympy` - For symbolic mathematics
- `ase` - Atomic Simulation Environment for structure handling

## C++ Acceleration Module

For improved performance, SA-GPR provides a C++ extension module (`kernels_cpp`) for the most computationally intensive kernel calculations.  

### Requirements

- A C++17 compatible compiler (e.g., `g++`, `clang++`)
- Python development headers (e.g., `python3-dev`)
- NumPy

### Compilation

To build the C++ extension, run the following command in the `src/utils/` directory:

```bash
cd src/utils
python3 setup.py build_ext --inplace
```

This will generate a shared library file (e.g., `kernels_cpp.cpython-3x-*.so`) in the same directory. The Python code will automatically use the C++ version if it is available.

If you modify the C++ source (`kernels.cpp`), simply rerun the above command to recompile.

## Installation

To use SA-GPR, simply ensure the required Python packages are installed (see Requirements above). If you want to use the C++ acceleration, please follow the instructions in the "C++ Acceleration Module" section to compile the extension before running the main scripts. All code can be run directly from the `src/` directory.

## Workflow

SA-GPR follows a two-step process for machine learning tensorial properties:

1. **Kernel Calculation**: Compute similarities (kernels) between molecules/bulk systems
2. **Regression**: Minimize prediction error and generate weights for predictions

These steps are performed using two main scripts:

- `sa_gpr_kernels.py`: Computes the kernels (step 1)
- `sa_gpr_apply.py`: Performs the regression (step 2)

## Examples

The `example/` directory contains sample data for different systems:

- `water_monomer/`: Dielectric tensors of water monomers
- `water_dimer/`: Water dimer calculations
- `water_zundel/`: Zundel cation examples
- `water_bulk/`: Bulk water with 32 molecules

### Setup

Before running examples, source the environment settings:

```bash
source env.sh
```

### 1. Water Monomer Example

Learn the energy of water monomers (scalar L=0 component, equivalent to standard SOAP):

```bash
cd example/water_monomer
sa_gpr_kernels.py -lval 0 -f coords_1000.xyz -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen O
```

This creates an L=0 kernel file with:
- Gaussian width: 0.3 Å
- Angular cutoff: l=6
- Radial cutoff: 4.0 Å
- Central atom weight: 1.0
- Centered on oxygen atoms

Perform regression:

```bash
sa_gpr_apply.py -r 0 -k kernel0_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt -rdm 200 -ftr 1.0 -f coords_1000.xyz -p "potential" -lm 1e-8
```

**Note:** For gas-phase clusters, cell vectors should give zero cell volume.

### 2. Zundel Cation Example

#### Learning Full Hyperpolarizability Tensor

For expensive calculations, split the kernel matrix into blocks:

```bash
cd example/water_zundel
python ../src/scripts/make_blocks.py -f coords_1000.xyz -n 10
```

In each generated Block folder, run:

```bash
sa_gpr_kernels.py -lval 1 -f coords.xyz -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen O
sa_gpr_kernels.py -lval 3 -f coords.xyz -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen O
```

Reconstruct the full kernel:

```bash
rebuild_kernel.py -l 1 -ns 1000 -nb 10 -rc 4.0 -lc 6 -sg 0.3 -cw 1.0
rebuild_kernel.py -l 3 -ns 1000 -nb 10 -rc 4.0 -lc 6 -sg 0.3 -cw 1.0
```

Perform regression:

```bash
sa_gpr_apply.py -r 3 -k kernel1_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0.txt kernel3_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0.txt -rdm 200 -ftr 1.0 -f coords_1000.xyz -p "beta" -lm 1e-6 1e-3
```

#### Learning Dipole Moment

The dipole moment (L=1 tensor) can be learned using the existing L=1 kernel:

```bash
sa_gpr_apply.py -r 1 -k kernel1_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0.txt -rdm 200 -ftr 1.0 -f coords_1000.xyz -p "mu" -lm 1e-3
```

#### Learning Components Separately

Split tensors into spherical components:

```bash
cartesian_to_spherical.py -f coords_1000.xyz -p "beta" -r 3 -o "processed_coords_1000.xyz"
```

Perform separate regressions:

```bash
# L=1 component
regression.py -k kernel1_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0.txt -f processed_coords_1000.xyz -p "beta_L1" -rdm 200 -nc 5 -ftr 1.0 -lm 1e-6 -o outputL1.out

# L=3 component
regression.py -k kernel3_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0.txt -f processed_coords_1000.xyz -p "beta_L3" -l 3 -rdm 200 -nc 5 -ftr 1.0 -lm 1e-6 -o outputL3.out
```

### 3. Bulk Water Example

For condensed-phase systems like liquid water:

```bash
cd example/water_bulk
python ../../src/scripts/make_blocks.py coords_1000.xyz 10
```

In each Block folder:

```bash
sa_gpr_kernels.py -lval 0 -f coords.xyz -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen O
sa_gpr_kernels.py -lval 2 -f coords.xyz -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen O
```

Reconstruct and perform regression:

```bash
rebuild_kernel.py -l 0 -ns 1000 -nb 100 -rc 4.0 -lc 6 -sg 0.3 -cw 1.0
rebuild_kernel.py -l 2 -ns 1000 -nb 100 -rc 4.0 -lc 6 -sg 0.3 -cw 1.0
sa_gpr_apply.py -r 2 -k kernel0_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt kernel2_1000_sigma0.3_lcut6_cutoff4.0_cweight1.0_n0.txt -rdm 200 -ftr 1.0 -f coords_1000.xyz -p "epsilon" -lm 1e-4 1e-4
```

## Technical Implementation

### Core Functions

The package implements two core computational functions:

- **`combine_spectra`**: Combines power spectra for L=0 SOAP kernel computation
- **`fill_spectra`**: Fills power spectra for L>0 spherical tensor SOAP kernel computation

These functions use optimized NumPy operations and maintain mathematical precision equivalent to the original implementation.

### Performance Notes

- Uses NumPy's optimized linear algebra routines
- Performance is typically 2-5x slower than highly optimized Fortran for very large systems (when not using the C++ module)
- Significantly improved portability and ease of installation
- Recommended for most research applications

## Project Structure

```
SA-GPR/
├── src/
│   ├── sa_gpr_kernels.py         # Kernel calculation
│   ├── sa_gpr_apply.py           # Regression application
│   ├── regression.py             # Direct regression tool
│   ├── cartesian_to_spherical.py # Tensor conversion
│   ├── scripts/                  # Utility scripts
│   └── utils/                    # Core utilities
│       ├── power_spectra.py      # Python implementation of core functions
│       ├── kernels.py            # Kernel computation
│       ├── parsing.py            # Input parsing
│       ├── read_xyz.py           # Structure reading
│       └── kern_utils.py         # Kernel utilities
├── example/                      # Example datasets
│   ├── water_monomer/
│   ├── water_dimer/
│   ├── water_zundel/
│   └── water_bulk/
└── env.sh                        # Environment setup
```

## Contact

- david.wilkins@epfl.ch
- andrea.grisafi@epfl.ch

## License

Please refer to the original publication for citation requirements when using this software in research.

---

*This implementation of SA-GPR provides an accessible and efficient tool for the scientific community to explore tensorial properties of atomistic systems.*
