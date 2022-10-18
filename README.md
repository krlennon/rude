# Rheological Universal Differential Equations

This repository contains the code used to produce all results in the preprint: [Scientific Machine Learning for Modeling and Simulating Complex Fluids](https://arxiv.org/abs/2210.04431).

## Compatibility

The scripts used to train and test RUDEs on shear rheometric data were developed using [Julia](https://julialang.org/downloads/) v1.6.3, with the following packages: DiffEqFlux v1.43.0, Flux v0.12.7, Optim v1.4.1, DifferentialEquations v6.19.0, PyPlot v2.10.0, OrdinaryDiffEq v5.64.0, DataInterpolations v3.6.1, BSON v0.3.5, and FFTW v1.4.5.

Computational fluid dynamics simulations were performed using [OpenFOAM](https://openfoam.org/download/archive/) v9, with the [rheoTool](https://github.com/fppimenta/rheoTool) v5.0 toolbox.

The scripts used to process experimental linear response data and to plot OpenFOAM simulation results were developed using [Python](https://www.python.org/downloads/) 3.8.5, with the following packages: NumPy 1.19.2, Matplotlib 3.3.2, SciPy 1.5.2, CSV 1.0, and pandas 1.1.3.

## Contents

### `giesekus`

- `rude.jl`: Julia script to generate synthetic shear stress LAOS data for the Giesekus model, train a RUDE using this data, and test the RUDE on shear and normal stress data in LAOS as well as shear stress data in shear startup
- `tbnn.bson`: pre-trained model weights for a RUDE trained using the `rude.jl` script
- `OpenFOAM`: directory containing the files needed to simulate a 4:1 contraction flow
	- `Giesekus`: directory containing the setup files for an OpenFOAM simulation of the Giesekus model, including the mesh file (`system/blockMeshDict`), details for the integration and finite volume schemes (`system/fvSchemes` and `system/fvSolution`), definition of the constitutive model (`constant/constitutiveProperties`), and initial conditions (`0`)
	- `Oldroyd-BLog`: directory containing the setup files for an OpenFOAM simulation of the Oldroyd-B model (defined by the log-conformation tensor). Same file structure as the `Giesekus` directory
	- `RUDE`: directory containing the setup files for an OpenFOAM simulation of a trained RUDE. Same file structure as the `Giesekus` directory, with three additional files (`weights1.txt`, `weights2.txt`, and `weights3.txt`) containing the pre-trained weights for the TBNN layers obtained using the `rude.jl` script
	- `simdata`: directory containing selected results from the OpenFOAM simulations, including centerline velocities, velocity profiles, the velocity field, and streamlines
	- `make_plots.py`: Python script for plotting the simulation results contained in `simdata`

### `gel`

- `data`: directory containing experimental data for the metal-crosslinked polymer hydrogel. Includes the raw data output by TRIOS (`raw`) and pre-processed CSV files for LAOS and SAOS tests
- `fitlr.py`: Python script to fit the linear response of the gel to a single-mode Maxwell model
- `rude.jl`: Julia script to train a RUDE using the LAOS data, and predict the shear and normal stress response in another LAOS experiment (as well as an amplitude-sweep in LAOS)
- `weights_201.bson`: partially pre-trained model, terminated after an epoch of 200 training iterations on the lowest-amplitude LAOS experiment
- `weights_402.bson`: partially pre-trained model, terminated after an epoch of 200 training iterations on the two lowest-amplitude LAOS experiments, with training initialized using the `weights_201.bson` network
- `weights_603.bson`: fully pre-trained model, terminated after an epoch of 200 training iterations on all three training LAOS curves, with training initialized using the `weights_402.bson` network

## Contibuting

Inquiries and suggestions can be directed to krlennon[at]mit.edu.

## License

[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)

