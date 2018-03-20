# Setting up CARLA simulator environment for Reinforcement Learning

 **Table of Contents**

   * [Setting up CARLA simulator environment for Reinforcement Learning](#setting-up-carla-simulator-environment-for-reinforcement-learning)
         * [Introduction](#introduction)
      * [Requirements:](#requirements)
      * [Setting up the CARLA Path](#setting-up-the-carla-path)
      * [Getting the required files for RL](#getting-the-required-files-for-rl)
      * [Playing with the Environment](#playing-with-the-environment)
         * [To create an CARLA environment](#to-create-an-carla-environment)
         * [Resetting the environment](#resetting-the-environment)
         * [Taking an action](#taking-an-action)
         * [Reading values after taking an action](#reading-values-after-taking-an-action)
         * [Rendering the game after each action](#rendering-the-game-after-each-action)
      * [Testing CARLA game as a human](#testing-carla-game-as-a-human)


### Introduction
If you didn't know, **[CARLA is an open-source simulator for autonomous driving research.](https://github.com/carla-simulator/carla "CARLA is an open-source simulator for autonomous driving research.")**

It can be used as an environment for training [ADAS](https://en.wikipedia.org/wiki/Advanced_driver-assistance_systems "ADAS"), and also for Reinforcement Learning.

This guide will help you set up the CARLA environment for RL. Most of my code here is inspired from [Intel Coach](https://github.com/NervanaSystems/coach "Intel Coach")'s setup of CARLA. I thought it'd be helpful to have a separte guide for this, to implement your own RL algorithms on top of it, instead of relying on Coach.

## Requirements:

- Download the[ latest CARLA release from here](https://github.com/carla-simulator/carla/releases " latest CARLA release from here"). (As of the time of writing, CARLA is in Experimental stage for Windows OS)
- Any Debian-based OS (Preferably Ubuntu 16.04)
- Python 3.x installed
- Python Dependencies required:
    - numpy
	- pygame
	- pynput (For testing the environment manually) [Optional]

## Setting up the CARLA Path

After downloading the release version, place in any accessible directory, preferably something like `/home/username/CARLA` or whatever.

Now open up your terminal, enter **`nano ~/.bashrc`** and include the PATH of the CARLA environment like:

```bash
export CARLA_ROOT=/home/username/CARLA
```

## Getting the required files for RL

Just clone (or fork) this repo by
```
git clone https://github.com/GokulNC/Setting-Up-CARLA-RL
```

All the required files for Environment's RL interface is present in the `Environment` directory (which you need not worry about)
*Note*: Most of the files are obtained from Intel Coach's interface for RL, with some modifications from my side.

## Playing with the Environment

### To create an CARLA environment
```python
from Environment.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv

env = CarlaEnv()  # To create an env
```

### Resetting the environment
```python
# returns the initial output values (as described in sections below)
initial_values = env.reset()
```

### Taking an action

```python
output = env.step(action_idx)
```

where **`action_idx`** is the discretized value of action corresponding to a specific action.

As of now, there are 9 discretized values, each corresponding to different actions as defined in  **`self.actions`** of `carla_environment_wrapper.py` like

```python
actions = {0: [0., 0.],
					1: [0., -self.steering_strength],
					2: [0., self.steering_strength],
					3: [self.gas_strength, 0.],
					4: [-self.brake_strength, 0],
					5: [self.gas_strength, -self.steering_strength],
					6: [self.gas_strength, self.steering_strength],
					7: [-self.brake_strength, -self.steering_strength],
					8: [-self.brake_strength, self.steering_strength]}
					
actions_description = ['NO-OP', 'TURN_LEFT', 'TURN_RIGHT', 'GAS', 'BRAKE',
									'GAS_AND_TURN_LEFT', 'GAS_AND_TURN_RIGHT',
									'BRAKE_AND_TURN_LEFT', 'BRAKE_AND_TURN_RIGHT']
```

(Feel free to modify it as you see fit)

### Reading values after taking an action

```python
# observation after taking the action
observation= output['observation']
# RGB image after taking the action
state = observation['observation']
# immediate reward after taking the action
reward = output['reward']
# boolean True/False indicating if episode is finished
# (collision has occured or time limit exceeded)
done = output['done'] 
# id of the last action taken
last_action_idx = output['action']
# information about the action taken & consequences
# (empty as of now, feel free to implement)
info = output['info']
```

### Rendering the game after each action
```python
env.render()
```

## Testing CARLA game as a human

I have included a file **`human_play.py`** which you can run by
```
python human_play.py
```

and play the game manually to get an understanding of it.
Use the arrow keys to play (`Up` to accelerate, `Down` to brake, `Left/Right` to steer)

