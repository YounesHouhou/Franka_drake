# Franka_drake

### Requirements

- Gymnasium
- Install Pydrake from https://drake.mit.edu/installation.html
- Install Manipulation :   
```bash
pip3 install manipulation --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly/ 
```
## 1) Instructions

First, run the following command one time to register the custom 'franka-v1' environment and make it usable by Gym.

```bash
python3 FRanka_gym_pos.py
```
You can now use the environment in the Test_env.ipynb notebook file.


## How is the custom environment built?

## How does the system compute an action?
