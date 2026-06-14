# Open-Loop Simulation

## Objective

The objective of this component is to generate representative open-loop data for the Continuous Stirred Tank Reactor (CSTR) used throughout this research.

## Inputs

None.

## Outputs

- X_train.npy
- Y_train.npy
- X_val.npy
- Y_val.npy
- X_test.npy
- Y_test.npy
- normalization_stats.csv
- Representative open-loop trajectory

## Methodology

Amplitude-Modulated Pseudo-Random Binary Signal input trajectories are generated and applied to the CSTR model. The resulting  outputs are collected and normalized for subsequent model training.

## Representative Results

### Open-Loop Trajectory

![Open Loop](Results/open_loop_trajectory.png)

The figure above illustrates a representative trajectory generated during the open-loop simulation stage.

## Dependencies

- NumPy
- SciPy
- Matplotlib
- Sklearn
- Pandas
