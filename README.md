# THCS
Thermal-Hydraulic Coupling Solution
## Overview
Thermal-hydraulic coupling system, a physical process where flow and heat transfer interact mutually, is ubiquitous in nature and industrial applications. Although experimental measurements and solution of governing equations can be used to quantify physical fields such as velocity and temperature, acquiring dense and high-fidelity data is still challenging and costly. Here, we develop a physics-informed deep learning framework, the Thermal-Hydraulic Coupling Solution (THCS), that simultaneously exploits the information available from sparse data and governing equations (mass, momentum, and energy), combined by the integration of a physics-informed neural network, a thermal property mapping module, and a hydraulic parameterization modeling module. To demonstrate the capability of THCS, we choose typical turbulent convection, characterized by sharp changes in thermal properties, as fluids cross the pseudocritical temperature. The ablation experiments show favorable generalization ability and robustness, while the proposed multi-head structures achieve remarkable convergence stability. Finally, we apply THCS to practical experiment scenarios, demonstrating its potential to quantify physical fields from sparse data.
## System Requirements
### Hardware requirements
We train our THCS models on an NVIDIA GeForce RTX 3090 with 24 GB memory.
### Software requirements
#### OS requirements
- Window 10 Pro
- Linux: Ubuntu 20.04.6 LTS

#### Python requirements
- Python 3.10.13
- Pytorch 2.0.1+cu118
- Numpy 1.24.3
- Pandas 1.5.3
- scipy 1.10.1
- CoolProp 6.4.3.post1
- Matplotlib 3.7.1
- tensorboard 2.15.1
## How to run
### Dataset
Considering the raw DNS data size being over large, we provide the mat files where the data are processed by Farve averaging and Reynolds averaging.
### Implementation
Generally, we evaluate our THCS on five tasks:
- Generalization test
- Robustness test
- multihead structure
- multiple working conditions
- practical scenarios

Note that when running each file, it is necessary to modify the file paths of the model and data.
