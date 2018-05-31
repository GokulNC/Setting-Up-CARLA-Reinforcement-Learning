#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import inspect
import os
import numpy as np
import threading
from subprocess import call, Popen
import signal
import copy
import scipy.misc
import png

killed_processes = []

eps = np.finfo(np.float32).eps

class Enum(object):
	def __init__(self):
		pass

	def keys(self):
		return [attr.lower() for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

	def vals(self):
		vars = dict(inspect.getmembers(self, lambda a: not (inspect.isroutine(a))))
		return {key.lower(): vars[key] for key in vars}

	def get(self, string):
		if string.lower() in self.keys():
			return self.vals()[string.lower()]
		raise NameError('enum does not exist')

	def verify(self, string):
		if string.lower() in self.keys():
			return string.lower(), self.vals()[string.lower()]
		raise NameError('enum does not exist')

	def to_string(self, enum):
		for key, val in self.vals().items():
			if val == enum:
				return key
		raise NameError('enum does not exist')


class RunPhase(Enum):
	HEATUP = "Heatup"
	TRAIN = "Training"
	TEST = "Testing"


def list_all_classes_in_module(module):
	return [k for k, v in inspect.getmembers(module, inspect.isclass) if v.__module__ == module.__name__]


def parse_bool(value):
	return {'true': True, 'false': False}.get(value.strip().lower(), value)


def convert_to_ascii(data):
	import collections
	if isinstance(data, basestring):
		return parse_bool(str(data))
	elif isinstance(data, collections.Mapping):
		return dict(map(convert_to_ascii, data.iteritems()))
	elif isinstance(data, collections.Iterable):
		return type(data)(map(convert_to_ascii, data))
	else:
		return data

def save_image(outfilename, npimg):
	#img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
	#img.save( outfilename )
	scipy.misc.toimage(npimg).save(outfilename)

def convert_segmented_to_rgb(label_colours, segmented_img):
	no_of_classes = len(label_colours)

	# Below commented lines to dynamically color
	# import cv2
	# label_colours = np.zeros((no_of_classes, 3),dtype = np.uint8)
	# for j in range(0, no_of_classes):
	# 	hsv = [int(j/float(no_of_classes)*170.0),255,255]
	# 	# TODO: Remove OpenCV dependency below
	# 	label_colours[j] = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]

	r = segmented_img.copy()
	g = segmented_img.copy()
	b = segmented_img.copy()

	for l in range(0, no_of_classes):
		r[segmented_img==l] = label_colours[l][0]
		g[segmented_img==l] = label_colours[l][1]
		b[segmented_img==l] = label_colours[l][2]

	rgb = np.zeros(segmented_img.shape+(3,))
	rgb[:,:,0] = r
	rgb[:,:,1] = g
	rgb[:,:,2] = b

	return rgb
	
def coalesce_depth_and_segmentation(segmented_img, classes_combo, depth_map, depth_scaler=1.0):
	'''
	Input:	Takes segmented image and depth map
	Output:	Creates an array having no. of channels = no. of classes, each channel for each class (or some classes can be joined into a single channel)
			Each value in each channel output[channel][w][h] is of range (0-1], corresponding to the normalized depth of the specific class.
			For all other classes, output[channel][w][h] = 0
	Arguments:
		segmented_img: Semantically segmented image
		depth_map: Depth of the image
		classes_combo: List of tuples, each denoting which channel will have what classes' data (list length will be no. of channels)
		depth_scaler: The value by which will be divided to bring down to the scale [0,1]
	'''
	assert segmented_img.shape == depth_map.shape, "Dimensions of segmented image and depth map doesn't match"
	
	num_channels = len(classes_combo)
	output = np.full((num_channels,) + depth_map.shape, False)
	
	classes_combo = [(i,) if type(i) is int else i for i in classes_combo]
	
	for i in range(num_channels):
		output[i] = np.any([segmented_img==j for j in classes_combo[i]], axis=0)
		output[i] = (np.multiply(output[i],depth_map))/depth_scaler
	return output

def depthmap_to_grey(depth_map, scale_factor=1.0):
	## Distributing same value across three channels in RGB produces grayscale image
	
	normalized_depth = np.multiply(depth_map, scale_factor)
	logdepth = numpy.ones(normalized_depth.shape) + \
        (numpy.log(normalized_depth) / 5.70378)
	logdepth_scaled = np.clip(logdepth, 0.0, 1.0)*255.0
	return numpy.repeat(logdepth_scaled[:, :, numpy.newaxis], 3, axis=2)
	#grey_img = [ [[logdepth_scaled[i][j], logdepth_scaled[i][j], logdepth_scaled[i][j]] for j in range(logdepth_scaled.shape[1])]
	#		for i in range(logdepth_scaled.shape[0])]
	#return np.array(grey_img)

def depthmap_to_rgb(depth_map, scale_factor=1.0):
	## Output ranges from blue (H=200) to red (H=360), corresponding to relative near to far
	## depth_map*scale_factor scales each distance to range [0, 1]
	
	normalized_depth = np.multiply(depth_map,scale_factor)
	
	logdepth = np.ones(normalized_depth.shape) + \
		(np.log(normalized_depth) / 5.70378)
	logdepth = np.clip(logdepth, 0.0, 1.0)
	normalized_log_depth = ((logdepth*160.0)+200.0)/360.0
	rgb_img = [ [list(h_to_rgb(normalized_log_depth[i][j])) for j in range(normalized_log_depth.shape[1])] for i in range(normalized_log_depth.shape[0])]
	return np.array(rgb_img)
	#rgb_img = np.zeros(depth_map.shape + (3,))
	#for i in range(depth_map.shape[0]):
	#	for j in range(depth_map.shape[1]):
	#		rgb_img[i][j] = list(h_to_rgb(depth_map[i][j]))
	#return rgb_img
	
def hsv_to_rgb(h, s, v):
	## All values should be in range [0,1]. (h=1.0 means 360degree)
	## Ref: https://stackoverflow.com/a/26856771/5002496
	if s == 0.0: v*=255; return (v, v, v)
	i = int(h*6.) # XXX assume int() truncates!
	f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
	if i == 0: return (v, t, p)
	if i == 1: return (q, v, p)
	if i == 2: return (p, v, t)
	if i == 3: return (p, q, v)
	if i == 4: return (t, p, v)
	if i == 5: return (v, p, q)

def h_to_rgb(h):
	## Optimized equivalent to hsv_to_rgb(h, 1.0, 1.0)
	i = int(h*6.)
	f = (h*6.)-i
	q,t = int(255*(1.-f)), int(255*f)
	i%=6
	if i == 0: return (255, t, 0)
	if i == 1: return (q, 255, 0)
	if i == 2: return (0, 255, t)
	if i == 3: return (0, q, 255)
	if i == 4: return (t, 0, 255)
	if i == 5: return (255, 0, q)
	return (0, 0, 0)

def save_depthmap_as_16bit_png(filename, depth_map, scale_factor=1.0, invalidate_above=1.0):
	## depth_map*scale_factor scales each distance to range [0, 1] and convert values>invalidate_above as 0
	## Output is stored in KITTI format, i.e., 16-bit grayscale PNG
	## Dividing each 16-bit value by 256 gives meters as specified in KITTI docs
	out = np.multiply(depth_map, scale_factor) # Assuming it produces depth map in kilometers of range [0-1]
	out[out>invalidate_above] = 0.0
	#out = (65535*out).astype(np.uint16) # simply scale up as 16-bit values
	out = (256000*out).astype(np.uint16) #convert it to meters by *1000 and *256 to produce KITTI compliant 16bit ground truth
	with open(filename, 'wb') as f:
		writer = png.Writer(width=out.shape[1], height=out.shape[0], bitdepth=16, greyscale=True)
		writer.write(f, out.tolist())

def is_process_alive(pid):
	## Source: https://stackoverflow.com/questions/568271/how-to-check-if-there-exists-a-process-with-a-given-pid-in-python
	try:
		os.kill(pid, 0)
	except OSError:
		return False
	return True

def break_file_path(path):
	base = os.path.splitext(os.path.basename(path))[0]
	extension = os.path.splitext(os.path.basename(path))[1]
	dir = os.path.dirname(path)
	return dir, base, extension


def is_empty(str):
	return str == 0 or len(str.replace("'", "").replace("\"", "")) == 0

def create_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def read_json(filename):
	# read json file
	with open(filename, 'r') as f:
		dict = json.loads(f.read())
		return dict


def write_json(filename, dict):
	# read json file
	with open(filename, 'w') as f:
		f.write(json.dumps(dict, indent=4))


def path_is_valid_dir(path):
	return os.path.isdir(path)


def remove_suffix(name, suffix_start):
	for s in suffix_start:
		split = name.find(s)
		if split != -1:
			name = name[:split]
			return name


def parse_int(value):
	import ast
	try:
		int_value = int(value)
		return int_value if int_value == value else value
	except:
		pass

	try:
		return ast.literal_eval(value)
	except:
		return value


def set_gpu(gpu_id):
	os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
	os.environ['NVIDIA_VISIBLE_DEVICES'] = str(gpu_id)


def set_cpu():
	set_gpu("")


# dictionary to class
class DictToClass(object):
	def __init__(self, data):
		for name, value in data.iteritems():
			setattr(self, name, self._wrap(value))

	def _wrap(self, value):
		if isinstance(value, (tuple, list, set, frozenset)):
			return type(value)([self._wrap(v) for v in value])
		else:
			return DictToClass(value) if isinstance(value, dict) else value


# class to dictionary
def ClassToDict(x):
	# return dict((key, getattr(x, key)) for key in dir(x) if key not in dir(x.__class__))
	dictionary = x.__dict__
	return {key: dictionary[key] for key in dictionary.keys() if not key.startswith('__')}


def cmd_line_run(result, run_cmd, id=-1):
	p = Popen(run_cmd, shell=True, executable="/bin/bash")
	while result[0] is None or result[0] == [None]:
		if id in killed_processes:
			p.kill()
		result[0] = p.poll()


def threaded_cmd_line_run(run_cmd, id=-1):
	runThread = []
	result = [[None]]
	try:
		params = (result, run_cmd, id)
		runThread = threading.Thread(name='runThread', target=cmd_line_run, args=params)
		runThread.daemon = True
		runThread.start()
	except:
		runThread.join()
	return result


class Signal(object):
	"""
	Stores a stream of values and provides methods like get_mean and get_max
	which returns the statistics about accumulated values.
	"""
	def __init__(self, name):
		self.name = name
		self.sample_count = 0
		self.values = []

	def reset(self):
		self.sample_count = 0
		self.values = []

	def add_sample(self, sample):
		"""
		:param sample: either a single value or an array of values
		"""
		self.values.append(sample)

	def _get_values(self):
		if type(self.values[0]) == np.ndarray:
			return np.concatenate(self.values)
		else:
			return self.values

	def get_mean(self):
		if len(self.values) == 0:
			return ''
		return np.mean(self._get_values())

	def get_max(self):
		if len(self.values) == 0:
			return ''
		return np.max(self._get_values())

	def get_min(self):
		if len(self.values) == 0:
			return ''
		return np.min(self._get_values())

	def get_stdev(self):
		if len(self.values) == 0:
			return ''
		return np.std(self._get_values())


def force_list(var):
	if isinstance(var, list):
		return var
	else:
		return [var]


def squeeze_list(var):
	if len(var) == 1:
		return var[0]
	else:
		return var


# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
	def __init__(self, shape):
		self._shape = shape
		self._num_samples = 0
		self._mean = np.zeros(shape)
		self._std = np.zeros(shape)

	def reset(self):
		self._num_samples = 0
		self._mean = np.zeros(self._shape)
		self._std = np.zeros(self._shape)

	def push(self, sample):
		sample = np.asarray(sample)
		assert sample.shape == self._mean.shape, 'RunningStat input shape mismatch'
		self._num_samples += 1
		if self._num_samples == 1:
			self._mean[...] = sample
		else:
			old_mean = self._mean.copy()
			self._mean[...] = old_mean + (sample - old_mean) / self._num_samples
			self._std[...] = self._std + (sample - old_mean) * (sample - self._mean)

	@property
	def n(self):
		return self._num_samples

	@property
	def mean(self):
		return self._mean

	@property
	def var(self):
		return self._std / (self._num_samples - 1) if self._num_samples > 1 else np.square(self._mean)

	@property
	def std(self):
		return np.sqrt(self.var)

	@property
	def shape(self):
		return self._mean.shape


def get_open_port():
	import socket
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(("", 0))
	s.listen(1)
	port = s.getsockname()[1]
	s.close()
	return port


class timeout:
	def __init__(self, seconds=1, error_message='Timeout'):
		self.seconds = seconds
		self.error_message = error_message

	def _handle_timeout(self, signum, frame):
		raise TimeoutError(self.error_message)

	def __enter__(self):
		signal.signal(signal.SIGALRM, self._handle_timeout)
		signal.alarm(self.seconds)

	def __exit__(self, type, value, traceback):
		signal.alarm(0)


def switch_axes_order(observation, from_type='channels_first', to_type='channels_last'):
	"""
	transpose an observation axes from channels_first to channels_last or vice versa
	:param observation: a numpy array
	:param from_type: can be 'channels_first' or 'channels_last'
	:param to_type: can be 'channels_first' or 'channels_last'
	:return: a new observation with the requested axes order
	"""
	if from_type == to_type or len(observation.shape) == 1:
		return observation
	assert 2 <= len(observation.shape) <= 3, 'num axes of an observation must be 2 for a vector or 3 for an image'
	assert type(observation) == np.ndarray, 'observation must be a numpy array'
	if len(observation.shape) == 3:
		if from_type == 'channels_first' and to_type == 'channels_last':
			return np.transpose(observation, (1, 2, 0))
		elif from_type == 'channels_last' and to_type == 'channels_first':
			return np.transpose(observation, (2, 0, 1))
	else:
		return np.transpose(observation, (1, 0))


class LazyStack(object):
	"""
	A lazy version of np.stack which avoids copying the memory until it is
	needed.
	"""

	def __init__(self, history, axis=None):
		self.history = copy.copy(history)
		self.axis = axis

	def __array__(self, dtype=None):
		array = np.stack(self.history, axis=self.axis)
		if dtype is not None:
			array = array.astype(dtype)
		return array


def stack_observation(curr_stack, observation, stack_size):
	"""
	Adds a new observation to an existing stack of observations from previous time-steps.
	:param curr_stack: The current observations stack.
	:param observation: The new observation
	:param stack_size: The required stack size
	:return: The updated observation stack
	"""

	if curr_stack == []:
		# starting an episode
		curr_stack = np.vstack(np.expand_dims([observation] * stack_size, 0))
		curr_stack = switch_axes_order(curr_stack, from_type='channels_first', to_type='channels_last')
	else:
		curr_stack = np.append(curr_stack, np.expand_dims(np.squeeze(observation), axis=-1), axis=-1)
		curr_stack = np.delete(curr_stack, 0, -1)

	return curr_stack


def last_sample(state):
	"""
	given a batch of states, return the last sample of the batch with length 1
	batch axis.
	"""
	return {
		k: np.expand_dims(v[-1], 0)
		for k, v in state.items()
	}
