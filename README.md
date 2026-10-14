# SSM-MPC: Spectral Submanifold Model Predictive Control

## Overview

This repository contains the modeling, control, and experimental implementation of Spectral Submanifold Model Predictive Control (SSM-MPC) for robotic systems.

The project includes SSM-based reduced-order models, nonlinear model predictive control (NMPC) algorithms, cable-length computation, hardware interfaces, and experimental validation programs for the **Diamond Robot** and **Trunk Robot**.

## Project Structure

The main directories are organized as follows:

| Directory | Description |
|---|---|
| `bin/` | Driver programs for the motion capture system. |
| `controller/` | Core NMPC algorithms and cable-length computation routines. |
| `Diamond_Experiments/` | Main experimental programs for validating SSM-MPC on the Diamond Robot. |
| `DriverNodeApp-v1.1.1/` | Motor driver application used for hardware control. |
| `dynamics/` | SSM model information, piecewise constant strain (PCS) models, and spiral and figure-eight reference trajectories for the Trunk Robot. |
| `measure/` | Measurement-related programs and utilities. |
| `obstacleInformation/` | Cached obstacle information obtained from motion capture measurements. |
| `Trunk_Experiments/` | Main experimental programs for validating SSM-MPC on the Trunk Robot. |
| `utils/` | Supporting utility functions and modules. |
| `zmq_message/` | Programs for testing the motor driver and transmitting control commands via ZeroMQ. |

## Experimental Validation

### Diamond Robot

The `Diamond_Experiments/` directory contains the main programs used to experimentally validate the SSM-MPC framework on the Diamond Robot.

### Trunk Robot

The `Trunk_Experiments/` directory contains the main programs used to experimentally validate SSM-MPC on the Trunk Robot. The associated reduced-order models, PCS models, and reference trajectories are provided in `dynamics/`.

## Control and Hardware Interfaces

The `controller/` directory contains the core NMPC implementation and cable-length computation routines.

The `bin/` directory provides the motion capture system drivers, while `DriverNodeApp-v1.1.1/` contains the motor driver application. The `zmq_message/` directory provides programs for motor driver testing and control command transmission.

Obstacle information acquired through the motion capture system is cached in `obstacleInformation/` for use in the experimental framework.
