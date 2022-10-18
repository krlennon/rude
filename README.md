# Rheological Universal Differential Equations

This repository contains the code used to produce all results in the preprint: [Scientific Machine Learning for Modeling and Simulating Complex Fluids](https://arxiv.org/abs/2210.04431).

## Compatibility

The scripts used to train and test RUDEs on shear rheometric data were developed using [Julia](https://julialang.org/downloads/) v1.6.3, with the following packages: DiffEqFlux v1.43.0, Flux v0.12.7, Optim v1.4.1, DifferentialEquations v6.19.0, PyPlot v2.10.0, OrdinaryDiffEq v5.64.0, DataInterpolations v3.6.1, BSON v0.3.5, and FFTW v1.4.5.

Computational fluid dynamics simulations were performed using [OpenFOAM](https://openfoam.org/download/archive/) v9, with the [rheoTool](https://github.com/fppimenta/rheoTool) v5.0 toolbox.

The scripts used to process experimental linear response data and to plot OpenFOAM simulation results were developed using [Python](https://www.python.org/downloads/) 3.8.5 with the following packages: NumPy 1.19.2, Matplotlib 3.3.2, SciPy 1.5.2, CSV 1.0, and pandas 1.1.3.

## Contents

### `giesekus`

## Contibuting

Inquiries and suggestions can be directed to krlennon[at]mit.edu.

## License

[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)

