import sys
from os import path, environ
try:
	if 'CARLA_ROOT' in environ:
		sys.path.append(path.join(environ.get('CARLA_ROOT'), 'PythonClient'))
except ImportError:
	print("CARLA Environment variable CARLA_ROOT not set")
	sys.exit(1)
from carla.client import CarlaClient
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.sensor import Camera
from carla.client import VehicleControl

import numpy as np
import time
import subprocess
import signal
from Environment.environment_wrapper import EnvironmentWrapper
from Environment.utils import *


# enum of the available levels and their path
class CarlaLevel(Enum):
	TOWN1 = "/Game/Maps/Town01"
	TOWN2 = "/Game/Maps/Town02"

key_map = {
	'BRAKE': (274,),  # down arrow
	'GAS': (273,),  # up arrow
	'TURN_LEFT': (276,),  # left arrow
	'TURN_RIGHT': (275,),  # right arrow
	'GAS_AND_TURN_LEFT': (273, 276),
	'GAS_AND_TURN_RIGHT': (273, 275),
	'BRAKE_AND_TURN_LEFT': (274, 276),
	'BRAKE_AND_TURN_RIGHT': (274, 275),
}

class CarlaEnvironmentWrapper(EnvironmentWrapper):
	def __init__(self, num_speedup_steps = 30, require_explicit_reset=True, is_render_enabled=False):
		EnvironmentWrapper.__init__(self, is_render_enabled)

		self.episode_max_time = 100000
		self.allow_braking = True
		self.log_path = 'logs/CarlaLogs.txt'
		self.num_speedup_steps = num_speedup_steps
		# server configuration
		self.server_height = 256
		self.server_width = 360
		self.render_height = 256
		self.render_width = 360
		self.port = get_open_port()
		self.host = 'localhost'
		self.level = 'town2' #Why town2: https://github.com/carla-simulator/carla/issues/10#issuecomment-342483829
		self.map = CarlaLevel().get(self.level)

		# client configuration
		self.verbose = True
		self.depth = False
		self.stereo = False
		self.semantic_segmentation = False
		self.height = self.server_height * (1 + int(self.depth) + int(self.semantic_segmentation))
		self.width = self.server_width * (1 + int(self.stereo))
		self.size = (self.width, self.height)
		self.observation = None

		self.config = 'Environment/CarlaSettings.ini'
		if self.config:
			# load settings from file
			with open(self.config, 'r') as fp:
				self.settings = fp.read()
		else:
			# hard coded settings
			print("CarlaSettings.ini not found; using default settings")
			self.settings = CarlaSettings()
			self.settings.set(
				SynchronousMode=True,
				SendNonPlayerAgentsInfo=False,
				NumberOfVehicles=15,
				NumberOfPedestrians=30,
				WeatherId=1)
			self.settings.randomize_seeds()

			# add cameras
			camera = Camera('CameraRGB')
			camera.set_image_size(self.width, self.height)
			camera.set_position(200, 0, 140)
			camera.set_rotation(0, 0, 0)
			self.settings.add_sensor(camera)

		self.is_game_setup = False # Will be true only when setup_client_and_server() is called, either explicitly, or by reset()

		# action space
		self.discrete_controls = False
		self.action_space_size = 2
		self.action_space_high = [1, 1]
		self.action_space_low = [-1, -1]
		self.action_space_abs_range = np.maximum(np.abs(self.action_space_low), np.abs(self.action_space_high))
		self.steering_strength = 0.5
		self.gas_strength = 1.0
		self.brake_strength = 0.5
		self.actions = {0: [0., 0.],
						1: [0., -self.steering_strength],
						2: [0., self.steering_strength],
						3: [self.gas_strength, 0.],
						4: [-self.brake_strength, 0],
						5: [self.gas_strength, -self.steering_strength],
						6: [self.gas_strength, self.steering_strength],
						7: [-self.brake_strength, -self.steering_strength],
						8: [-self.brake_strength, self.steering_strength]}
		self.actions_description = ['NO-OP', 'TURN_LEFT', 'TURN_RIGHT', 'GAS', 'BRAKE',
									'GAS_AND_TURN_LEFT', 'GAS_AND_TURN_RIGHT',
									'BRAKE_AND_TURN_LEFT', 'BRAKE_AND_TURN_RIGHT']
		for idx, action in enumerate(self.actions_description):
			for key in key_map.keys():
				if action == key:
					self.key_to_action[key_map[key]] = idx

		# measurements
		self.measurements_size = (1,)
		self.autopilot = None

		# env initialization
		if not require_explicit_reset: self.reset(True)

		# render
		if self.automatic_render:
			image = self.get_rendered_image()
			self.renderer.create_screen(image.shape[1], image.shape[0])

	def setup_client_and_server(self):
		# open the server
		self.server = self._open_server()

		# open the client
		self.game = CarlaClient(self.host, self.port, timeout=99999999)
		self.game.connect(connection_attempts=100) #It's taking a very long time for the server process to spawn, so the client needs to wait or try sufficient no. of times lol
		scene = self.game.load_settings(self.settings)

		# get available start positions
		positions = scene.player_start_spots
		self.num_pos = len(positions)
		self.iterator_start_positions = 0
		self.is_game_setup = self.server and self.game
		return

	def close_client_and_server():
		self.game.disconnect()
		self.game = None
		self._close_server() #Assuming it will close properly lol; TODO: poll() if it's closed
		self.sever = None
		self.is_game_setup = False
		return


	def _open_server(self):
		# Note: There is no way to disable rendering in CARLA as of now
		# https://github.com/carla-simulator/carla/issues/286
		# decrease the window resolution if you want to see if performance increases
		# Command: $CARLA_ROOT/CarlaUE4.sh /Game/Maps/Town02 -benchmark -carla-server -fps=15 -world-port=9876 -windowed -ResX=480 -ResY=360 -carla-no-hud
		with open(self.log_path, "wb") as out:
			cmd = [path.join(environ.get('CARLA_ROOT'), 'CarlaUE4.sh'), self.map,
								  "-benchmark", "-carla-server", "-fps=10", "-world-port={}".format(self.port),
								  "-windowed -ResX={} -ResY={}".format(self.server_width, self.server_height),
								  "-carla-no-hud"]
			if self.config:
				cmd.append("-carla-settings={}".format(self.config))
			p = subprocess.Popen(cmd, stdout=out, stderr=out)
		return p

	def _close_server(self):
		os.killpg(os.getpgid(self.server.pid), signal.SIGKILL)

	def _update_state(self):
		# get measurements and observations
		measurements = []
		while type(measurements) == list:
			measurements, sensor_data = self.game.read_data()

		self.location = (measurements.player_measurements.transform.location.x,
						 measurements.player_measurements.transform.location.y,
						 measurements.player_measurements.transform.location.z)

		is_collision = measurements.player_measurements.collision_vehicles != 0 \
					   or measurements.player_measurements.collision_pedestrians != 0 \
					   or measurements.player_measurements.collision_other != 0
		if is_collision: print("Collision occured")

		speed_reward = measurements.player_measurements.forward_speed - 1
		if speed_reward > 30.:
			speed_reward = 30.
		self.reward = speed_reward \
					  - (measurements.player_measurements.intersection_otherlane * 5) \
					  - (measurements.player_measurements.intersection_offroad * 5) \
					  - is_collision * 100 \
					  - np.abs(self.control.steer) * 10

		# update measurements
		self.observation = {
			'observation': sensor_data['CameraRGB'].data,
			'measurements': [measurements.player_measurements.forward_speed],
		}
		self.autopilot = measurements.player_measurements.autopilot_control

		# action_p = ['%.2f' % member for member in [self.control.throttle, self.control.steer]]
		# screen.success('REWARD: %.2f, ACTIONS: %s' % (self.reward, action_p))

		if (measurements.game_timestamp >= self.episode_max_time) or is_collision:
			# screen.success('EPISODE IS DONE. GameTime: {}, Collision: {}'.format(str(measurements.game_timestamp),
			#                                                                      str(is_collision)))
			self.done = True
			print("Episode Ended")

	def _take_action(self, action_idx):
		if not self.is_game_setup:
			print("Reset the environment duh by reset() before calling step()")
			sys.exit(1)
		if type(action_idx) == int:
			action = self.actions[action_idx]
		else:
			action = action_idx

		self.control = VehicleControl()
		self.control.throttle = np.clip(action[0], 0, 1)
		self.control.steer = np.clip(action[1], -1, 1)
		self.control.brake = np.abs(np.clip(action[0], -1, 0))
		if not self.allow_braking:
			self.control.brake = 0
		self.control.hand_brake = False
		self.control.reverse = False

		self.game.send_control(self.control)

	def _restart_environment_episode(self, force_environment_reset=True):

		if not force_environment_reset and not self.done and self.is_game_setup:
			print("Can't reset dude, episode ain't over yet")
			return None #User should handle this

		if not self.is_game_setup:
			self.setup_client_and_server()
			if self.is_render_enabled: self.renderer.create_screen(self.render_width, self.render_height)
		else:
			self.iterator_start_positions += 1
			if self.iterator_start_positions >= self.num_pos:
				self.iterator_start_positions = 0

		try:
			self.game.start_episode(self.iterator_start_positions)
		except:
			self.game.connect()
			self.game.start_episode(self.iterator_start_positions)

		# start the game with some initial speed
		observation = None
		for i in range(self.num_speedup_steps):
			observation, reward, done, _ = self.step([1.0, 0])
		self.observation = observation

		return observation
