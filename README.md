# IsaacSim-Franka

## Introduction
This repository provides a custom Reinforcement Learning (RL) environment using the Franka Panda arm. The environment is designed to accept visual information as input, allowing for the development and testing of vision-based RL algorithms. It integrates NVIDIA Isaac Sim for realistic simulation, providing a robust platform for robotics research and experimentation.

> **Note:**  
> This repository is currently in an **experimental stage**. Some features may not work as expected, and further refinement is ongoing. Users are encouraged to provide feedback and contribute.

### Current Limitations:
- Training with **single image observations** using SAC (Soft Actor-Critic) often fails to improve the reward. 
- However, when using **proprioceptive observations** (e.g., joint positions, velocities), noticeable reward improvement has been observed.


## Prerequisites
Before starting, make sure you have the following installed:

- NVIDIA Isaac Sim 4.1.0
- `cmake` and `build-essential` packages

You can install the required packages with the following command:

```bash
sudo apt install cmake build-essential
```

## Installation
First of all you should install **IsaacSim** at [Installation Guide of IsaacSim](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).
I recommend you to use conda virtual environment for this repository. 

```bash
mkdir ~/workspace
cd workspace
git clone git@github.com:jmSNU/IsaacSim-Franka.git
cd IsaacSim-Franka/IsaacLab
sudo apt install cmake build-essential
./isaaclab.sh -i
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.1.0"
```

## Recommended Protocol
```bash
cd $ISAACSIM_PATH
source setup_conda_env.sh && source setup_python_env.sh
conda activate "conda_virtual_env"
cd ~/workspace/IsaacSim-Franka
python IsaacLab/source/standalone/tutorials/00_sim/create_empty.py
```

## Environments
- [Isaac-Franka-Reach-v0](https://github.com/jmSNU/IsaacSim-Franka/blob/main/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/franka_manipulation/franka_reach) : Reach a target position using the Franka Panda arm.
- [Isaac-Franka-Push-v0](https://github.com/jmSNU/Isaacsim-Franka/tree/main/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/franka_manipulation/franka_push) : Push an object to a designated location.
- [Isaac-Franka-Lift-v0](https://github.com/jmSNU/Isaacsim-Franka/tree/main/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/franka_manipulation/franka_lift) : Lift an object to a specified height.
- [Isaac-Franka-Pick-v0](https://github.com/jmSNU/Isaacsim-Franka/tree/main/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/franka_manipulation/franka_pick) : Pick up an object and place it at a target location.

> **Note:**  
>The reward function design for all environments except Isaac-Franka-Reach-v0 and Isaac-Franka-Push-v0 follows the approach used in [Robosuite](https://robosuite.ai), a popular framework for robotic manipulation. Robosuite provides well-structured reward functions that focus on task-specific goals such as object lifting, and placing.
