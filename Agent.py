import numpy as np
import random
from collections import deque
import plaidml.keras

plaidml.keras.install_backend()

import keras
from keras.layers import LeakyReLU

class Agent(object):

	def __init__(self):
		self.ACTION = 2
		self.batch = 32
		self.gamma = 0.99
		self.MaxMem = 50000
		self.learning_rate = 1e-6

		self.model = self._Create()
		self.Mem = deque()

	def _load(self,x):
		if x!=0:
			self.model.load_weights("Agent" + str(x) +".h5")
			print("Loaded model sucessfully from disk")

	def _save(self,x):
		self.model.save_weights("Agent" + str(x) +".h5")
		print("Saved model successfully to disk")

	def save_mem(self,obs,act,rwd,n_o,ter):
		if len(self.Mem) > self.MaxMem: self.Mem.popleft()
		self.Mem.append((obs,np.argmax(act),rwd,n_o,ter))

	def act(self,obs):
		act = np.argmax(self.model.predict(np.array([obs])))
		# print(self.model.predict(np.array([obs])))
		if act :
			return [0,1]
		else:
			return [1,0]

	def replay(self):
		mini_batch = random.sample(self.Mem,self.batch)
		S  = np.array([d[0] for d in mini_batch])
		S1 = np.array([d[3] for d in mini_batch])

		temp = self.model.predict(S)
		temp1 = self.model.predict(S1)

		for i in range(0,self.batch):
			if mini_batch[i][4]:
				temp[i][mini_batch[i][1]] = mini_batch[i][2]*10
			else:
				temp[i][mini_batch[i][1]] = mini_batch[i][2] + \
				self.gamma*np.max(temp1[i])

		self.model.train_on_batch(S,temp)

	def _Create(self):

		model = keras.models.Sequential()

		RAN = keras.initializers.RandomNormal(
			mean=0.0, stddev=.01, seed=None)

		ACT = 'linear'

		model.add(keras.layers.Reshape((1,80,80),
			input_shape=(1,80,80)))

		model.add(keras.layers.Conv2D(32,(20,20),
			data_format="channels_first",
			strides = 4,
			padding ="same",
			activation=ACT,
			use_bias=True,
			kernel_initializer=RAN,
			bias_initializer=RAN))

		model.add(LeakyReLU(alpha=.001))

		model.add(keras.layers.Conv2D(64,(5,5),
			data_format="channels_first",
			strides = 5,
			padding ="same",
			activation=ACT,
			use_bias=True,
			kernel_initializer=RAN,
			bias_initializer=RAN))

		model.add(LeakyReLU(alpha=.001))

		model.add(keras.layers.Conv2D(128,(3,3),
			data_format="channels_first",
			strides = 2,
			padding ="same",
			activation=ACT,
			use_bias=True,
			kernel_initializer=RAN,
			bias_initializer=RAN))

		model.add(LeakyReLU(alpha=.001))	

		model.add(keras.layers.Conv2D(256,(2,2),
			data_format="channels_first",
			strides = 2,
			padding ="same",
			activation=ACT,
			use_bias=True,
			kernel_initializer=RAN,
			bias_initializer=RAN))

		model.add(LeakyReLU(alpha=.001))	
		
		model.add(keras.layers.Flatten())

		model.add(keras.layers.Dense(256, 
			activation=ACT,
			use_bias=True,
			bias_initializer=RAN,
			kernel_initializer=RAN))

		model.add(LeakyReLU(alpha=.001))

		model.add(keras.layers.Dense(256, 
			activation=ACT,
			use_bias=True,
			bias_initializer=RAN,
			kernel_initializer=RAN))

		model.add(LeakyReLU(alpha=.001))

		model.add(keras.layers.Dense(self.ACTION,
			activation='linear',
			use_bias=True,
			bias_initializer=RAN,
			kernel_initializer=RAN))

		opt = keras.optimizers.Adam(lr = self.learning_rate)
		model.compile(optimizer=opt,loss='mse')

		model.summary()
		return model