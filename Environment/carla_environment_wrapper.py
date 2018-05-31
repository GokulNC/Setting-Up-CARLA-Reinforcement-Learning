import sys, time, subprocess, signal
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
from carla.image_converter import depth_to_logarithmic_grayscale, depth_to_local_point_cloud, depth_to_array
from Environment.renderer import Renderer

import numpy as np
from Environment.environment_wrapper import EnvironmentWrapper
from Environment.utils import *
import Environment.carla_config as carla_config


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
	def __init__(self, num_speedup_steps = 30, require_explicit_reset=True, is_render_enabled=False, early_termination_enabled=False, run_offscreen=False, cameras=['SceneFinal'], save_screens=False):
		EnvironmentWrapper.__init__(self, is_render_enabled, save_screens)

		self.episode_max_time = 1000000
		self.allow_braking = True
		self.log_path = 'logs/CarlaLogs.txt'
		self.num_speedup_steps = num_speedup_steps
		self.is_game_ready_for_input = False
		self.run_offscreen = run_offscreen
		self.kill_when_connection_lost = True
		# server configuration

		self.port = get_open_port()
		self.host = 'localhost'
		self.level = 'town2' #Why town2: https://github.com/carla-simulator/carla/issues/10#issuecomment-342483829
		self.map = CarlaLevel().get(self.level)

		# client configuration
		self.verbose = True
		self.observation = None
		
		self.camera_settings = dict(
			ImageSizeX=carla_config.server_width,
			ImageSizeY=carla_config.server_height,
			FOV=90.0,
			PositionX=2.0, # 200 for Carla 0.7
			PositionY=0.0,
			PositionZ=1.40, # 140 for Carla 0.7
			RotationPitch = 0.0,
			RotationRoll = 0.0,
			RotationYaw = 0.0,
		)
		
		self.rgb_camera_name = 'CameraRGB'
		self.segment_camera_name = 'CameraSegment'
		self.depth_camera_name = 'CameraDepth'
		self.rgb_camera = 'SceneFinal' in cameras
		self.segment_camera = 'SemanticSegmentation' in cameras
		self.depth_camera = 'Depth' in cameras
		self.class_grouping = carla_config.class_grouping or [(i, ) for i in range(carla_config.no_of_classes)]
		self.autocolor_the_segments = False
		self.color_the_depth_map = False
		self.enable_coalesced_output = False
		
		self.max_depth_value = 1.0 #255.0 for CARLA 7.0
		self.min_depth_value = 0.0
		
		self.config = None #'Environment/CarlaSettings.ini'
		if self.config:
			# load settings from file
			with open(self.config, 'r') as fp:
				self.settings = CarlaSettings(fp.read())
		else:
			# hard coded settings
			#print("CarlaSettings.ini not found; using default settings")
			self.settings = CarlaSettings()
			self.settings.set(
				SynchronousMode=True,
				SendNonPlayerAgentsInfo=False,
				NumberOfVehicles=15,
				NumberOfPedestrians=30,
				WeatherId=1,
				SeedVehicles = 123456789,
				SeedPedestrians = 123456789)
			#self.settings.randomize_seeds()

		# add cameras
		if self.rgb_camera: self.settings.add_sensor(self.create_camera(self.rgb_camera_name, 'SceneFinal'))
		if self.segment_camera: self.settings.add_sensor(self.create_camera(self.segment_camera_name, 'SemanticSegmentation'))
		if self.depth_camera: self.settings.add_sensor(self.create_camera(self.depth_camera_name, 'Depth'))
		
		
		self.car_speed = 0
		self.is_game_setup = False # Will be true only when setup_client_and_server() is called, either explicitly, or by reset()

		# action space
		self.discrete_controls = True
		self.action_space_size = 2
		self.action_space_high = [1, 1]
		self.action_space_low = [-1, -1]
		self.action_space_abs_range = np.maximum(np.abs(self.action_space_low), np.abs(self.action_space_high))
		self.steering_strength = 0.35
		self.gas_strength = 1.0
		self.brake_strength = 0.6
		self.actions = {0: [0., 0.],
						1: [0., -self.steering_strength],
						2: [0., self.steering_strength],
						3: [self.gas_strength-0.15, 0.],
						4: [-self.brake_strength, 0],
						5: [self.gas_strength-0.3, -self.steering_strength],
						6: [self.gas_strength-0.3, self.steering_strength],
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
		self.kill_if_unmoved_for_n_steps = 15
		self.unmoved_steps = 0.0
		
		self.early_termination_enabled = early_termination_enabled
		if self.early_termination_enabled:
			self.max_neg_steps = 70
			self.cur_neg_steps = 0
			self.early_termination_punishment = 20.0

		# env initialization
		if not require_explicit_reset: self.reset(True)

		# render
		if self.automatic_render:
			self.init_renderer()
		if self.save_screens:
			create_dir(self.images_path)
			self.rgb_img_path = self.images_path+"/rgb/"
			create_dir(self.rgb_img_path)
			self.segmented_img_path = self.images_path+"/segmented/"
			create_dir(self.segmented_img_path)
			self.depth_img_path = self.images_path+"/depth/"
			create_dir(self.depth_img_path)
			

	def create_camera(self, camera_name, PostProcessing):
		#camera = Camera('CameraRGB')
		#camera.set_image_size(carla_config.server_width, carla_config.server_height)
		#camera.set_position(200, 0, 140)
		#camera.set_rotation(0, 0, 0)
		#self.settings.add_sensor(camera)
		camera = Camera(camera_name, **dict(self.camera_settings, PostProcessing=PostProcessing))
		return camera
		

	def setup_client_and_server(self, reconnect_client_only = False):
		# open the server
		if not reconnect_client_only:
			self.server = self._open_server()
			self.server_pid = self.server.pid # To kill incase child process gets lost

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

	def close_client_and_server(self):
		self._close_server()
		print("Disconnecting the client")
		self.game.disconnect()
		self.game = None
		self.server = None
		self.is_game_setup = False
		return

	def _open_server(self):
		# Note: There is no way to disable rendering in CARLA as of now
		# https://github.com/carla-simulator/carla/issues/286
		# decrease the window resolution if you want to see if performance increases
		# Command: $CARLA_ROOT/CarlaUE4.sh /Game/Maps/Town02 -benchmark -carla-server -fps=15 -world-port=9876 -windowed -ResX=480 -ResY=360 -carla-no-hud
		# To run off_screen: SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 <command> #https://github.com/carla-simulator/carla/issues/225
		my_env = None
		if self.run_offscreen:
			my_env = {**os.environ, 'SDL_VIDEODRIVER': 'offscreen', 'SDL_HINT_CUDA_DEVICE':'0'}
		with open(self.log_path, "wb") as out:
			cmd = [path.join(environ.get('CARLA_ROOT'), 'CarlaUE4.sh'), self.map,
								  "-benchmark", "-carla-server", "-fps=10", "-world-port={}".format(self.port),
								  "-windowed -ResX={} -ResY={}".format(carla_config.server_width, carla_config.server_height),
								  "-carla-no-hud"]
			if self.config:
				cmd.append("-carla-settings={}".format(self.config))
			p = subprocess.Popen(cmd, stdout=out, stderr=out, env=my_env)
		return p

	def _close_server(self):
		if self.kill_when_connection_lost:
			os.killpg(os.getpgid(self.server.pid), signal.SIGKILL)
			return
		no_of_attempts = 0
		while is_process_alive(self.server_pid):
			print("Trying to close Carla server with pid %d" % self.server_pid)
			if no_of_attempts<5:
				self.server.terminate()
			elif no_of_attempts<10:
				self.server.kill()
			elif no_of_attempts<15:
				os.kill(self.server_pid, signal.SIGTERM)
			else:
				os.kill(self.server_pid, signal.SIGKILL) 
			time.sleep(10)
			no_of_attempts += 1

	def check_early_stop(self, player_measurements, immediate_reward):
		
		if player_measurements.intersection_offroad>0.95 or immediate_reward < -1 or (self.control.throttle == 0.0 and player_measurements.forward_speed < 0.1 and self.control.brake != 0.0):
			self.cur_neg_steps += 1
			early_done = (self.cur_neg_steps > self.max_neg_steps)
			if early_done:
				print("Early kill the mad car")
				return early_done, self.early_termination_punishment
		else:
			self.cur_neg_steps /= 2 #Exponentially decay
		return False, 0.0
	
	def _update_state(self):
		# get measurements and observations
		measurements = []
		while type(measurements) == list:
			try:
				measurements, sensor_data = self.game.read_data()
			except:
				# Connection between cli and server lost; reconnect
				if self.kill_when_connection_lost: raise
				print("Connection to server lost while reading state. Reconnecting...........")
				self.close_client_and_server()
				self.setup_client_and_server(reconnect_client_only=False)
				self.done = True

		self.location = (measurements.player_measurements.transform.location.x,
						 measurements.player_measurements.transform.location.y,
						 measurements.player_measurements.transform.location.z)

		is_collision = measurements.player_measurements.collision_vehicles != 0 \
					   or measurements.player_measurements.collision_pedestrians != 0 \
					   or measurements.player_measurements.collision_other != 0

		# CARLA doesn't recognize if collision occured and colliding speed is less than 5 km/h (Around 0.7 m/s)
		# Ref: https://github.com/carla-simulator/carla/issues/13
		# Recognize that as a collision
		self.car_speed = measurements.player_measurements.forward_speed
		
		if self.control.throttle > 0 and self.car_speed < 0.75 and self.control.brake==0.0 and self.is_game_ready_for_input:
			self.unmoved_steps += 1.0
			if self.unmoved_steps > self.kill_if_unmoved_for_n_steps:
				is_collision = True
				print("Car stuck somewhere lol")
		elif self.unmoved_steps>0: self.unmoved_steps -= 0.50 #decay slowly, since it may be stuck and not accelerate few times
		
		if is_collision: print("Collision occured")
		
		speed_reward = self.car_speed - 1
		if speed_reward > 30.:
			speed_reward = 30.
		self.reward = speed_reward*1.2 \
					  - (measurements.player_measurements.intersection_otherlane * (self.car_speed+1.5)*1.2) \
					  - (measurements.player_measurements.intersection_offroad * (self.car_speed+2.5)*1.5) \
					  - is_collision * 250 \
					  - np.abs(self.control.steer) * 2
		# Scale down the reward by a factor
		self.reward /= 10
		
		if self.early_termination_enabled:
			early_done, punishment = self.check_early_stop(measurements.player_measurements, self.reward)
			if early_done:
				self.done = True
			self.reward -= punishment
		
		# update measurements
		self.observation = {
			#'observation': sensor_data['CameraRGB'].data,
			'acceleration': measurements.player_measurements.acceleration,
			'forward_speed': measurements.player_measurements.forward_speed,
			'intersection_otherlane': measurements.player_measurements.intersection_otherlane,
			'intersection_offroad': measurements.player_measurements.intersection_offroad
		}
		
		if self.rgb_camera:
			self.observation['rgb_image'] = sensor_data[self.rgb_camera_name].data
		if self.segment_camera:
			self.observation['segmented_image'] = sensor_data[self.segment_camera_name].data
		if self.depth_camera:
			self.observation['depth_map'] = sensor_data[self.depth_camera_name].data
		
		if self.segment_camera and self.depth_camera and self.enable_coalesced_output:
			self.observation['coalesced_data'] = coalesce_depth_and_segmentation(
						self.observation['segmented_image'], self.class_grouping, self.observation['depth_map'], self.max_depth_value)
		
		if self.segment_camera and (self.autocolor_the_segments or self.is_render_enabled):
			self.observation['colored_segmented_image'] = convert_segmented_to_rgb(carla_config.colors_segment, self.observation['segmented_image'])
		self.autopilot = measurements.player_measurements.autopilot_control

		# action_p = ['%.2f' % member for member in [self.control.throttle, self.control.steer]]
		# screen.success('REWARD: %.2f, ACTIONS: %s' % (self.reward, action_p))

		if (measurements.game_timestamp >= self.episode_max_time) or is_collision:
			# screen.success('EPISODE IS DONE. GameTime: {}, Collision: {}'.format(str(measurements.game_timestamp),
			#																	  str(is_collision)))
			self.done = True

	def _take_action(self, action_idx):
		if not self.is_game_setup:
			print("Reset the environment duh by reset() before calling step()")
			sys.exit(1)
		if type(action_idx) == int:
			action = self.actions[action_idx]
		else:
			action = action_idx

		self.control = VehicleControl()
		
		if self.car_speed>35.0 and action[0]>0:
			action[0] -= 0.20*(self.car_speed/35.0)
		self.control.throttle = np.clip(action[0], 0, 1)
		self.control.steer = np.clip(action[1], -1, 1)
		self.control.brake = np.abs(np.clip(action[0], -1, 0))
		if not self.allow_braking:
			self.control.brake = 0
		self.control.hand_brake = False
		self.control.reverse = False
		controls_sent = False
		while not controls_sent:
			try:
				self.game.send_control(self.control)
				controls_sent = True
			except:
				if self.kill_when_connection_lost: raise
				print("Connection to server lost while sending controls. Reconnecting...........")
				self.close_client_and_server()
				self.setup_client_and_server(reconnect_client_only=False)
				self.done = True
		return

		
	def init_renderer(self):
		self.num_cameras = 0
		if self.rgb_camera: self.num_cameras += 1
		if self.segment_camera: self.num_cameras += 1
		if self.depth_camera: self.num_cameras += 1
		self.renderer.create_screen(carla_config.render_width, carla_config.render_height*self.num_cameras)
		
	def _restart_environment_episode(self, force_environment_reset=True):

		if not force_environment_reset and not self.done and self.is_game_setup:
			print("Can't reset dude, episode ain't over yet")
			return None #User should handle this
		self.is_game_ready_for_input = False
		if not self.is_game_setup:
			self.setup_client_and_server()
			if self.is_render_enabled:
				self.init_renderer()
		else:
			self.iterator_start_positions += 1
			if self.iterator_start_positions >= self.num_pos:
				self.iterator_start_positions = 0

		try:
			self.game.start_episode(self.iterator_start_positions)
		except:
			self.game.connect()
			self.game.start_episode(self.iterator_start_positions)

		self.unmoved_steps = 0.0
		
		if self.early_termination_enabled:
			self.cur_neg_steps = 0
		# start the game with some initial speed
		self.car_speed = 0
		observation = None
		for i in range(self.num_speedup_steps):
			observation, reward, done, _ = self.step([1.0, 0])
		self.observation = observation
		self.is_game_ready_for_input = True

		return observation		
	
	def get_rendered_image(self):
		
		temp = []
		if self.rgb_camera: temp.append(self.observation['rgb_image'])
		if self.segment_camera:
			temp.append(self.observation['colored_segmented_image'])
		if self.depth_camera:
			if self.color_the_depth_map: temp.append(depthmap_to_rgb(self.observation['depth_map']))
			else: temp.append(depthmap_to_grey(self.observation['depth_map']))
			return np.vstack((img for img in temp))
	
	def save_screenshots(self):
		if not self.save_screens:
			print("save_screens is set False")
			return
		filename = str(int(time.time()*100))
		if self.rgb_camera:
			save_image(self.rgb_img_path+filename+".png", self.observation['rgb_image'])
		if self.segment_camera:
			np.save(self.segmented_img_path+filename, self.observation['segmented_image'])
		if self.depth_camera:
			save_depthmap_as_16bit_png(self.depth_img_path+filename+".png",self.observation['depth_map'],self.max_depth_value,0.95) #Truncating sky as 0
			#save_depthmap_as_16bit_png(self.images_path+"/depth_kitti/"+filename+".png", self.observation['depth_map'], self.max_depth_value, 0.09) #Truncate above 90m
