# Accuracy and Computational Cost of Standard and Sequential Time-Window Physics-Informed Neural Networks for Transient Heat Conduction with a Moving Gaussian Heat Source

Source code accompanying the study on transient heat conduction with a moving Gaussian heat source using finite-difference and Physics-Informed Neural Network (PINN) approaches.

## Overview

This repository contains the source code used for the numerical simulations, PINN training, sequential time-window training, and computational analyses presented in the associated study.

The study investigates a two-dimensional transient heat conduction problem with a moving Gaussian heat source at scanning speeds of 1, 2, and 3 mm/s over a simulation period of 10 s.

Three computational approaches are implemented:

1. **FTCS** — Forward Time-Centered Space finite-difference method used as the numerical reference solution.
2. **Standard PINN** — Physics-Informed Neural Network trained over the complete temporal domain.
3. **Sequential Time-Window PINN** — sequential PINN trained over 10 temporal windows with parameter transfer and temporal information transfer between consecutive windows.

## Repository Structure

```text
Transient-Heat-Conduction-PINN/
│
├── ftcs.py
├── pinn_standard.py
├── sequential_time_window_pinn.py
├── README.md
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Source Code

### `ftcs.py`

Implementation of the GPU-accelerated FTCS solver for the two-dimensional transient heat conduction equation with a moving Gaussian heat source.

The implementation includes:

- scanning speeds of 1, 2, and 3 mm/s;
- grid resolutions of 100 × 100, 200 × 200, 300 × 300, 400 × 400, and 500 × 500;
- Fourier-number-based stability assessment;
- time-step determination based on the numerical stability criterion;
- checkpoint and restart functionality;
- temperature-field output;
- centerline temperature extraction;
- time-history analysis; and
- grid-convergence analysis.

### `pinn_standard.py`

Implementation of the Standard PINN using JAX.

The model uses:

- input variables `(x, y, t)`;
- 8 hidden layers;
- 128 neurons per hidden layer;
- `tanh` activation;
- Adam optimization;
- learning rate of 10⁻³;
- 80,000 training epochs;
- global training over the 0–10 s temporal domain; and
- automatic differentiation for evaluation of the governing-equation residual.

The Standard PINN does not use sequential temporal-window training or parameter transfer between temporal windows.

### `sequential_time_window_pinn.py`

Implementation of the Sequential Time-Window PINN.

The 10 s simulation period is divided into ten consecutive temporal windows of 1 s:

```text
0–1 s
1–2 s
2–3 s
3–4 s
4–5 s
5–6 s
6–7 s
7–8 s
8–9 s
9–10 s
```

Each temporal window is trained sequentially.

The implementation includes:

- 10 temporal windows;
- 8,000 training epochs per window;
- parameter transfer between consecutive windows;
- temporal initial information transferred from the preceding window;
- source-focused collocation sampling;
- automatic differentiation using JAX;
- checkpoint and restart functionality; and
- window-level loss and computational-time tracking.

## Physical Problem

The computational domain is a two-dimensional rectangular domain with:

- Length: 35 mm
- Width: 17 mm
- Simulation time: 10 s
- Initial temperature: 20 °C
- Scanning speeds: 1, 2, and 3 mm/s

The heat input is represented by a moving Gaussian heat source.

The same physical problem is used as the basis for the FTCS reference solution and the two PINN approaches.

## Numerical Configuration

### FTCS

Five grid resolutions are considered:

```text
100 × 100
200 × 200
300 × 300
400 × 400
500 × 500
```

Numerical stability is assessed using the Fourier-number criterion.

The 500 × 500 grid is used as the reference resolution for comparison with the PINN approaches.

## PINN Configuration

The PINN architecture consists of:

```text
Input:              (x, y, t)
Hidden layers:      8
Neurons per layer:  128
Activation:         tanh
Output:             Temperature
```

The governing equation, initial condition, and boundary conditions are incorporated into the physics-informed loss.

### Standard PINN

The Standard PINN is trained globally over:

```text
t = 0–10 s
```

with a total of 80,000 training epochs.

### Sequential Time-Window PINN

The Sequential Time-Window PINN divides the temporal domain into ten consecutive 1 s windows.

Each window is trained for 8,000 epochs:

```text
10 windows × 8,000 epochs = 80,000 epochs
```

Model parameters are transferred between consecutive windows, while temporal information from the preceding window is used to provide the initial information for the next window.

Source-focused sampling is employed to increase collocation-point concentration around the moving heat source.

## Computational Environment

The source codes are implemented in Python.

The computational implementations use:

- **CuPy** for GPU-accelerated FTCS calculations;
- **JAX** for PINN training and automatic differentiation;
- **Optax** for PINN optimization; and
- **Google Colab** as the computational environment.

The reported computational experiments were performed using an NVIDIA Tesla T4 GPU.

The required Python packages and versions are provided in `requirements.txt`.

## Reproducibility

The source codes in this repository correspond to the computational procedures used in the associated study.

The implementations include functionality for:

- numerical stability assessment;
- grid-convergence analysis;
- PINN training;
- checkpointing and restart;
- loss-history recording;
- computational-time recording;
- temperature-field generation; and
- centerline-temperature extraction.

For complete reproduction of the reported results, the source code should be used together with the corresponding dataset and configuration information archived in Zenodo.

## Data Availability

The numerical results and datasets generated using the source code are archived separately in Zenodo.

The associated dataset contains FTCS, Standard PINN, and Sequential Time-Window PINN results, including temperature fields, centerline profiles, training histories, computational-time data, comparison data, and publication figures.

**Zenodo DOI:** `[TO BE ADDED]`

## Software Availability

The source code used in this study is openly available through GitHub and archived in Zenodo.

**GitHub:** `[TO BE ADDED]`

**Software DOI:** `[TO BE ADDED]`

## Citation

If you use this source code or reproduce the computational results reported in the associated study, please cite the corresponding publication and software archive.

### Publication

> `[Publication citation to be added after publication.]`

### Software

> `[Software citation to be added after the Zenodo archive is created.]`

## License

This software is distributed under the MIT License.

See the `LICENSE` file for the full license text.

