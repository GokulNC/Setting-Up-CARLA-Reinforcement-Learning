from Environment.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv
#import keyboard
# import pygame
# from pygame.locals import *
from pynput import keyboard
from threading import Thread
import time
steering_strength = 0.5
gas_strength = 1.0
brake_strength = -0.5

action = [0.0, 0.0]
reset = False
debug_logs = True

def start_listen():
	## Inspired from: https://pypi.python.org/pypi/pynput
	global action, reset, steering_strength, gas_strength, brake_strength

	def on_press(key):
		global action, reset, steering_strength, gas_strength, brake_strength
		if key == keyboard.Key.up: action[0] = gas_strength
		elif key == keyboard.Key.down:
			action[0] = brake_strength
			#if action[1]==0.0: action[0] = -brake_strength
		elif key == keyboard.Key.left:
			action[1] = -steering_strength
			#if action[0] == -brake_strength: action[0] = brake_strength
		elif key == keyboard.Key.right:
			action[1] = steering_strength
			#if action[0] == -brake_strength: action[0] = brake_strength
		elif key == keyboard.Key.space: reset = True

	def on_release(key):
		global action
		if key == keyboard.Key.up or key == keyboard.Key.down: action[0] = 0.0
		elif key == keyboard.Key.left or key == keyboard.Key.right:
			action[1] = 0.0


	# Collect events until released
	with keyboard.Listener(
			on_press=on_press,
			on_release=on_release) as listener:
		listener.join()



print("Creating Environment & resetting")
env = CarlaEnv(num_speedup_steps = 10)
print("Environment created")

actions = {0: (0., 0.),
				1: (0., -steering_strength),
				2: (0., steering_strength),
				3: (gas_strength, 0.),
				4: (brake_strength, 0.),
				5: (gas_strength, -steering_strength),
				6: (gas_strength, steering_strength),
				7: (brake_strength, -steering_strength),
				8: (brake_strength, steering_strength)}

action_map = {v: k for k, v in actions.items()} #https://stackoverflow.com/a/483833/5002496

frame_skip = 1

env.reset()
total_reward = 0

t = Thread(target=start_listen) # Start listening to key presses and update actions
t.start()

if debug_logs:
	frame_id = 0
	total_frames = 100 # No. of frames once to print the FPS rate
	start_time = time.time()

while True:
	frame_id += 1
	r = 0.0
	if debug_logs:
		print("Action: "+str(action)+" - ID: "+str(action_map[tuple(action)]))
		frame_id = (frame_id+1) % total_frames
		if frame_id==0:
			end_time = time.time()
			print("FPS: "+str(total_frames/(end_time-start_time)))
			start_time = end_time


	for _ in range(frame_skip):
		output = env.step(action_map[tuple(action)])

		r += output['reward']
		done = output['done']
		if done: break

	total_reward += r
	if reset:
		done = True

	if done:
		env.reset()
		reset = False
		print("Total reward in episode:"+str(total_reward))
		total_reward = 0