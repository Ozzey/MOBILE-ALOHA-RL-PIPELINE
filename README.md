# MOBILE ALOHA REINFORCEMENT LEARNING

This repository is based on OmniIsaacGymEnvs & Isaac-Sim 2023 and is used for autonomous Mobile Aloha tasks. 
<p align="center">
  <img src="https://github.com/user-attachments/assets/f358545a-64be-44eb-8107-a6842809beca" alt="AlohaPickAppleKitchen" width="400"/>
  <img src="https://github.com/user-attachments/assets/26f03e0c-2af0-48c7-b91b-d30c338cbb3d" alt="AlohaPlacePlateKitchen" width="400"/>
  <img src="https://github.com/user-attachments/assets/a7426a74-511e-4f6f-9aea-060fd7d11219" alt="AlohaKitchenCabinet" width="400"/>
</p>



## Installation

1. Install [Isaac-Sim 2023.1.0](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) 
2. Install OmniisaacGymEnvs as a pip package. Follow the the installation procedure [here](https://github.com/isaac-sim/OmniIsaacGymEnvs)
3.
4. Clone the repository:
   ```
     git clone https://github.com/Ozzey/ALOHA-RL
   ```
5. Download the latest assets from [here](https://disk.yandex.com/d/mr830qgJpAeS_g) and place them in a new folder called ```assets``` in the root directory of the project.
6. Modify ```config.json``` to add the path to the assets as required.


## Training and Evaluation

1. **Training**: The following is example for training:
     ```PYTHON_PATH rlgames_train.py task=Cabinet headless=True```
   - Training with Livestream:
     ```PYTHON_PATH rlgames_train.py task=Cabinet headless=True enable_livestream=True ```
   - Default no. of environments for each task is 4096.
   - Avoid simulating during training process as it can freeze the system.
   - For testing purposes during simulation you need to modify mini-batch size and horizon lenght in cfg/train directory in the specific file.
   - Ideally for testing purposes during training use set the value of mini-batch size and horizon lenght to be **8192**

2. **Evaluation**: The agents can be evaluated after testing:  
    ```PYTHON_PATH scripts/rlgames_train.py task=Cabinet checkpoint=runs/Cabinet/nn/Cabinet.pth test=True num_envs=1```
   - Checkpoints can be used to continue training by setting ```test=False```.
   - It is possible to train the agents in simple environment and test in complicated (for example kitchen) but the succcess rate would be lower.
      
*For full set of functions checkout [OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs)*

## List of Tasks Available:

### Aloha (manipulator) Tasks

1. AlohaPick
2. AlohaPlace
3. AlohaPickPlace
4. AlohaPickCamera
5. AlohaPlaceCamera
6. AlohaPickPlaceCamera

### Mobile-Aloha (manipulator attached to robot) Tasks:

1. MobileAlohaPick
2. MobileAlohaPlace
3. MobileAlohaPickPlace
4. MobileAlohaPickCamera
5. MobileAlohaPlaceCamera
6. MobileAlohaPickPlaceCamera

## Text Based Input:

It is possible to inform the agent which object in needs to pick up. For now we consider only 2 objects, namely, apple and cube. To select apple run as following:
```
PYTHON_PATH rlgames_train.py task=AlohaPickLLM checkpoint=/path/to/checkpoin num_envs=1 test=True task.env.apple=True
```
For selecting cube, set ```task.env.apple=False```

## Upcoming Functionalities:

1. Mobile Tasks
2. Depth Camera and Sensors
3. GPT based encoding
4. Sim-2-Real


## **IMPORTANT**!!!:

* Make sure to install the correct versions of ISAAC-SIM (2023.1.0) , [OmniIsaacGymEnvs](5db8d510775d55a0b979fc18b361cd6aa5a97dde) and PyTorch-Nightly(2.1.0.dev20230616-devel) inorder to reproduce the results from the report

* If your GPU is not strong enough, or the simulation doesn't work, comment out ```get_kitchen()``` function (NEWER RTX MODELS USE DIFFERENT RAY-TRACING TECHNOLOGY WHICH IS MORE COMPUTATIONALLY HEAVY AND CAN FREEZE THE SIMULATION).

* Always use ```headless=True``` during training and ```num_envs=1``` during evaluation. 
