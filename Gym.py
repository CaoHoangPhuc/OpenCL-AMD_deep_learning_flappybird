import gym
import numpy as np
import random
import plaidml.keras

plaidml.keras.install_backend()

import keras
# print(keras.backend.backend())
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import LeakyReLU


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.998
        self.learning_rate = 1e-4
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(keras.layers.Reshape((1,4), input_shape=(1,4)))
        model.add(Dense(4*4, activation=LeakyReLU(alpha=.001)))
        model.add(Dense(4*4*4, activation=LeakyReLU(alpha=.001)))
        model.add(Dense(4*4*4*4, activation=LeakyReLU(alpha=.001)))
        model.add(Dense(4*4*4, activation=LeakyReLU(alpha=.001)))
        model.add(Dense(4*4, activation=LeakyReLU(alpha=.001)))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size), -10
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0]), np.max(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        targets = self.model.predict(states)
        q_values_next = self.model.predict(next_states)

        targets[np.arange(batch_size), 0, actions] = rewards + self.gamma * np.max(q_values_next.reshape(batch_size,2), axis = 1) * (1 - dones)

        self.model.train_on_batch(states, targets)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("cartpole-dqn.h5")
    # train the agent
    done = False
    batch_size = 64
    for e in range(500):
        state = env.reset()  # Reset the environment and get the initial state
        env.render() # uncomment to render the game
        state = np.reshape(state[0], [1, 4])  # Reshape the state to have 4 features
        for time in range(1000):
            # env.render() # uncomment to render the game
            action, x = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}, sc: {}"
                      .format(e, 500, time, agent.epsilon, x ))
                break
                
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 50 == 0:
            agent.save("cartpole-dqn.h5")
            print("saved")
