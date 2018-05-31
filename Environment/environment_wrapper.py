import numpy as np
np.set_printoptions(threshold=np.inf)
from Environment.utils import *
from Environment.renderer import Renderer
import Environment.carla_config as carla_config
import operator, time

class EnvironmentWrapper(object):
	def __init__(self, is_render_enabled=False, save_screens = False):
		"""
		:param tuning_parameters:
		:type tuning_parameters: Preset
		"""
		# env initialization
		self.game = []
		self.actions = {}
		#self.state = []
		self.reward = 0
		self.done = False
		self.default_action = 0
		self.episode_idx = 0
		self.last_episode_time = time.time()
		self.info = {}
		self.action_space_low = 0
		self.action_space_high = 0
		self.action_space_abs_range = 0
		self.actions_description = {}
		self.discrete_controls = True
		self.action_space_size = 0
		self.key_to_action = {}
		self.width = 1
		self.height = 1
		self.is_state_type_image = True
		self.measurements_size = 0
		self.phase = RunPhase.TRAIN
		self.record_video_every = 1000
		#self.env_id = self.tp.env.level
		self.video_path = 'temp/experiment-videos'
		self.images_path = 'screens'
		self.seed = None
		self.frame_skip = 1
		self.automatic_render = False # Otherwise it'll automatically render()
		self.wait_for_explicit_human_action = False
		self.game_is_open = True
		self.is_render_enabled = is_render_enabled or self.automatic_render
		self.save_screens = save_screens
		if self.is_render_enabled: self.renderer = Renderer()
		else: self.renderer = None
		self.observation = None

	@property
	def measurements(self):
		assert False

	@measurements.setter
	def measurements(self, value):
		assert False

	# @property
	# def observation(self):
	# 	assert False
	#
	# @observation.setter
	# def observation(self, value):
	# 	assert False

	def _idx_to_action(self, action_idx):
		"""
		Convert an action index to one of the environment available actions.
		For example, if the available actions are 4,5,6 then this function will map 0->4, 1->5, 2->6
		:param action_idx: an action index between 0 and self.action_space_size - 1
		:return: the action corresponding to the requested index
		"""
		return self.actions[action_idx]

	def _action_to_idx(self, action):
		"""
		Convert an environment action to one of the available actions of the wrapper.
		For example, if the available actions are 4,5,6 then this function will map 4->0, 5->1, 6->2
		:param action: the environment action
		:return: an action index between 0 and self.action_space_size - 1, or -1 if the action does not exist
		"""
		for key, val in self.actions.items():
			if val == action:
				return key
		return -1

	def get_action_from_user(self):
		"""
		Get an action from the user keyboard
		:return: action index
		"""
		if self.wait_for_explicit_human_action:
			while len(self.renderer.pressed_keys) == 0:
				self.renderer.get_events()

		if self.key_to_action == {}:
			# the keys are the numbers on the keyboard corresponding to the action index
			if len(self.renderer.pressed_keys) > 0:
				action_idx = self.renderer.pressed_keys[0] - ord("1")
				if 0 <= action_idx < self.action_space_size:
					return action_idx
		else:
			# the keys are mapped through the environment to more intuitive keyboard keys
			# key = tuple(self.renderer.pressed_keys)
			# for key in self.renderer.pressed_keys:
			for env_keys in self.key_to_action.keys():
				if set(env_keys) == set(self.renderer.pressed_keys):
					return self.key_to_action[env_keys]

		# return the default action 0 so that the environment will continue running
		return self.default_action

	def step(self, action_idx):
		"""
		Perform a single step on the environment using the given action
		:param action_idx: the action to perform on the environment
		:return: A dictionary containing the state, reward, done flag and action
		"""
		self.info['action'] = action_idx

		self._take_action(action_idx)

		self._update_state()

		if self.automatic_render:
			self.render()

		#self.state = self._preprocess_state(self.state)
		return self.observation, self.reward, self.done, self.info

	def render(self):
		"""
		Call the environment function for rendering to the screen
		"""
		if self.renderer is None or not (self.is_render_enabled):
			print("Unable to render: is_render_enabled is set False.")
			return

		img = self.get_rendered_image()
		self.renderer.render_image(img)
		#print(str(img))
	
	def save_screenshots(self):
		pass
	
	def reset(self, force_environment_reset=True):
		"""
		Reset the environment and all the variable of the wrapper
		:param force_environment_reset: forces environment reset even when the game did not end
		:return: A dictionary containing the state, reward, done flag and action
		"""
		self._restart_environment_episode(force_environment_reset)
		self.last_episode_time = time.time()
		self.done = False
		self.episode_idx += 1
		self.reward = 0.0
		self._update_state()

		# render before the preprocessing of the state, so that the image will be in its original quality
		if self.is_render_enabled:
			self.render()

		# TODO BUG: if the environment has not been reset, _preprocessed_state will be running on an already preprocessed state
		# TODO: see also _update_state above
		#self.state = self._preprocess_state(self.state)

		return self.observation

	def get_random_action(self):
		"""
		Returns an action picked uniformly from the available actions
		:return: a numpy array with a random action
		"""
		if self.discrete_controls:
			return np.random.choice(self.action_space_size)
		else:
			return np.random.uniform(self.action_space_low, self.action_space_high)

	def change_phase(self, phase):
		"""
		Change the current phase of the run.
		This is useful when different behavior is expected when testing and training
		:param phase: The running phase of the algorithm
		:type phase: RunPhase
		"""
		self.phase = phase

	def get_available_keys(self):
		"""
		Return a list of tuples mapping between action names and the keyboard key that triggers them
		:return: a list of tuples mapping between action names and the keyboard key that triggers them
		"""
		available_keys = []
		if self.key_to_action != {}:
			for key, idx in sorted(self.key_to_action.items(), key=operator.itemgetter(1)):
				if key != ():
					key_names = [self.renderer.get_key_names([k])[0] for k in key]
					available_keys.append((self.actions_description[idx], ' + '.join(key_names)))
		elif self.discrete_controls:
			for action in range(self.action_space_size):
				available_keys.append(("Action {}".format(action + 1), action + 1))
		return available_keys

	# The following functions define the interaction with the environment.
	# Any new environment that inherits the EnvironmentWrapper class should use these signatures.
	# Some of these functions are optional - please read their description for more details.

	def _take_action(self, action_idx):
		"""
		An environment dependent function that sends an action to the simulator.
		:param action_idx: the action to perform on the environment
		:return: None
		"""
		pass

	def _preprocess_state(self, state):
		"""
		Do initial state preprocessing such as cropping, rgb2gray, rescale etc.
		Implementing this function is optional.
		:param state: a raw state from the environment
		:return: the preprocessed state
		"""
		return state

	def _update_state(self):
		"""
		Updates the state from the environment.
		Should update self.state, self.reward, self.done and self.info
		:return: None
		"""
		pass

	def _restart_environment_episode(self, force_environment_reset=False):
		"""
		:param force_environment_reset: Force the environment to reset even if the episode is not done yet.
		:return:
		"""
		pass

	def get_rendered_image(self):
		"""
		Return a numpy array containing the image that will be rendered to the screen.
		This can be different from the state. For example, mujoco's state is a measurements vector.
		:return: numpy array containing the image that will be rendered to the screen
		"""
		pass
