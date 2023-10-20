import numpy as np
import plaidml.keras
plaidml.keras.install_backend()
import keras
from keras.layers import LeakyReLU
from keras.layers import LeakyReLU
from collections import deque
import random
import keras
from copy import deepcopy

import requests, time, threading
url = "https://api.casinoscores.com/svc-evolution-game-events/api/lightningroulette/latest"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

previous_data = None
new_data = False
def fetch_data():
    global previous_data, new_data
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data != previous_data:
                # New data is available
                print("New data:", data)
                with open("data_log.txt", "a") as file:
                    file.write(str(data['data']['result']['outcome']['number']) +"\n")
                new_data = True
                previous_data = data
            else:
                print("No new data")
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
        time.sleep(5)  # Wait for 20 seconds before the next request

# Create a thread for data fetching
data_fetch_thread = threading.Thread(target=fetch_data, daemon=True)


# Environment Setup
pre_train = True
class RouletteEnvironment:
    def __init__(self):
        self.num_states = 36
        self.num_actions = 7
        self.state = np.zeros(self.num_states) #np.random.randint(num_outcomes, size=num_outcomes)  # Represents the state (previous numbers)
        self.num = np.random.randint(0, 36)

    def step(self, action, ine = None):
        global new_data
            
        if ine is not None:
            outcome = ine
        else:
            try:
                while not new_data:
                    # Your main thread's tasks here
                    time.sleep(1)  # Sleep for 1 second or perform other tasks
                    
                outcome = previous_data['data']['result']['outcome']['number']
                new_data = False
            except KeyboardInterrupt:
                outcome = int(input("Predict is {}, input is: ".format(action)))
            # Simulate the roulette wheel
            # outcome = np.random.randint(0, 36)
            # outcome = (self.num*3 + 1 ) %37

        self.num = outcome
        if action == 6:
            reward = -0.1
        else:
            if outcome == 0:
                reward = -1 * ((action // 2) +1)
            elif (outcome % 2) == (action %2):
                reward = 1 * ((action // 2) +1)
            else:
                reward = - 1 * ((action // 2) +1)
        # reward = 1 if outcome == action else 0  # Simple reward: 1 for correct prediction, 0 otherwise

        # Update the state (for demonstration, we replace the oldest number with the new outcome)
        self.state[:-1] = self.state[1:]
        self.state[-1] = outcome

        return deepcopy(self.state), reward
        

# Double Q-Learning Agent
class DoubleQLearningAgent:
    
    EXPLORE = 1000
    INIT_EP = 0.9999
    FINL_EP = 0.0001

    def __init__(self, num_states, num_actions, learning_rate, discount_factor, epsilon):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # self.q1_table = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]
        # self.q2_table = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]

        self.Mem = deque()
        self.MaxMem = 500
        self.batch = 32

        RAN = keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None)
        # Create a simple Q-network
        self.model = keras.Sequential([
            keras.layers.Dense(16, use_bias=True, kernel_initializer=RAN, bias_initializer=RAN, activation=LeakyReLU(alpha=0.1), input_shape=(self.num_states,), name='input_layer'),
            keras.layers.Dense(36, use_bias=True, kernel_initializer=RAN, bias_initializer=RAN, activation=LeakyReLU(alpha=0.1), name='hidden_layer'),
            keras.layers.Dense(72, use_bias=True, kernel_initializer=RAN, bias_initializer=RAN, activation=LeakyReLU(alpha=0.1), name='hidden_layer2'),
            keras.layers.Dense(7, activation='linear', name='output_layer')
        ])
        self.model_q = keras.Sequential([
            keras.layers.Dense(72, use_bias=True, kernel_initializer=RAN, bias_initializer=RAN, activation=LeakyReLU(alpha=0.1), input_shape=(self.num_states,), name='input_layer'),
            keras.layers.Dense(36, use_bias=True, kernel_initializer=RAN, bias_initializer=RAN, activation=LeakyReLU(alpha=0.1), name='hidden_layer'),
            keras.layers.Dense(16, use_bias=True, kernel_initializer=RAN, bias_initializer=RAN, activation=LeakyReLU(alpha=0.1), name='hidden_layer2'),
            keras.layers.Dense(7, activation='linear', name='output_layer')
        ])
        
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        # Compile the model
        self.model_q.compile(optimizer=optimizer, loss='mse')
        self.model.compile(optimizer=optimizer, loss='mse')

    def save_mem(self, obs,q1,q2,rwd,n_o):
        if len(self.Mem) > self.MaxMem: self.Mem.popleft()
        self.Mem.append((obs,q1,q2,rwd,deepcopy(n_o)))
        if len(self.Mem) > self.batch: self.replay()

    def _load(self, x):
        self.model.load_weights("Agent_duo" + str(x) +".h5")
        self.model_q.load_weights("Agent_duo_q" + str(x) +".h5")
        print("Loaded model sucessfully from disk")

    def _save(self, x):
        self.model.save_weights("Agent_duo" + str(x) +".h5")
        self.model_q.save_weights("Agent_duo_q" + str(x) +".h5")
        print("Saved model successfully to disk")

    def q_table(self, state):
        return self.model.predict(state.reshape(1, -1), verbose=0)

    def q_table_q(self, state):
        return self.model_q.predict(state.reshape(1, -1), verbose=0)

    def check_result(self, outcome):
        if outcome == 0:
            return [-1, -1, -2, -2, -3, -3, -0.1]
        elif outcome %2 == 0:
            return [1, -1, 2, -2, 3, -3, -0.1]
        elif outcome %2 == 1:
            return [-1, 1, -2, 2, -3, 3, -0.1]
        
    def choose_action(self, state):
        # if self.epsilon > self.FINL_EP:
        #     self.epsilon -= (self.INIT_EP - self.FINL_EP)/self.EXPLORE

        # if random.random() < self.epsilon:
        #     return random.randint(0, self.num_actions-1), self.q_table(state), self.q_table_q(state)
        
        q_values = self.q_table(state)
        q_values_q = self.q_table_q(state)
        if random.random() < 0.5: #np.max(q_values) > np.max(q_values_q): #
            return np.argmax(q_values +q_values_q) , q_values, q_values_q
        else:
            return np.argmax(q_values_q +q_values) , q_values, q_values_q


    # def update_q1(self, )

    def update(self, state, action, next_state, reward):

        (max_next_action, q_table, q_table_q) = self.choose_action(next_state)
        target = reward + self.discount_factor * self.q_table(next_state)[0][max_next_action]
        target = reward + self.discount_factor * self.q_table_q(next_state)[0][max_next_action]
        q_table[0][action] = target
        q_table_q[0][action] = target
        self.model.train_on_batch(state.reshape(1, -1), q_table)
        self.model_q.train_on_batch(state.reshape(1, -1), q_table_q)

    def replay(self):
        mini_batch = random.sample(self.Mem, self.batch)
        
        S  = np.array([d[0] for d in mini_batch])
        S1 = np.array([d[4] for d in mini_batch])
        q = self.model.predict(S, verbose=0)
        q1 = self.model.predict(S1, verbose=0)
        q_q = self.model_q.predict(S, verbose=0)
        q_q1 = self.model_q.predict(S1, verbose=0)

        for i in range(self.batch):
            rewards = self.check_result(S1[i][-1])
            for z in range(7):
                q[i][z] = rewards[z] + self.discount_factor * q1[i][z]
                q_q[i][z] = rewards[z] + self.discount_factor* q_q1[i][z]

            # q[i][np.argmax(mini_batch[i][1])] = mini_batch[i][3] + self.discount_factor* np.max(q1[i])
            # q_q[i][np.argmax(mini_batch[i][2])] = mini_batch[i][3] + self.discount_factor* np.max(q_q1[i])

        self.model.train_on_batch(S,q)
        self.model_q.train_on_batch(S,q1)

# Training the Agent
data_fetch_thread.start()
def train_agent(agent, env, num_episodes):
    state = deepcopy(env.state)
    agent._load(1)
    global pre_train
    for episode in range(num_episodes):
        total_reward = 0
        done = False
        rounds = 0
        pre_train = False
        while not done:
            # total_reward = 0
            # done = False
            # out = []
            # with open("data_log.txt", "r") as file:
            #     for line in file:
            #         out.append(int(line))
            # if pre_train:
            #     for i in out:
            #         rounds +=1
            #         (action, q1, q2) = agent.choose_action(state)
            #         print("Sample Round: {}, predict: {}, amount: {}".format(
            #             rounds, 
            #             "NOBET" if action == 6 else "EVEN" if action%2==0 else "ODD", 
            #             "0" if action == 6 else str(action // 2 + 1)))
            #         next_state, reward = env.step(action, i)
            #         agent.save_mem(state, q1, q2,reward,next_state)
            #         if action != 6: total_reward += reward
            #         print("outcome:{}, profit: {}".format(next_state[-1], total_reward))
            #         # print(np.round(q1[0],2))
            #         # print(np.round(q2[0],2))
            #         # agent.update(state, action, next_state, reward)
            #         state = next_state
            #         if rounds % 500 == 0:
            #             agent._save(1)
            # else:
            
            rounds +=1
            (action, q1, q2) = agent.choose_action(state)
            print("Sample Round: {}, predict: {}, amount: {}".format(
                rounds, 
                "NOBET" if action == 6 else "EVEN" if action%2==0 else "ODD", 
                "0" if action == 6 else str(action // 2 + 1)))
            next_state, reward = env.step(action)
            if rounds > roulette_env.num_states:
                agent.save_mem(state, q1, q2,reward,next_state)
            if action != 6: total_reward += reward
            print("outcome:{}, profit: {}".format(next_state[-1], total_reward))
            # print(np.round(q1[0],2))
            # print(np.round(q2[0],2))
            # agent.update(state, action, next_state, reward)
            state = next_state
            agent._save(1)

            pre_train = False

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    roulette_env = RouletteEnvironment()
    agent = DoubleQLearningAgent(roulette_env.num_states, roulette_env.num_actions, learning_rate=0.001, discount_factor=0.9, epsilon=1)

    train_agent(agent, roulette_env, num_episodes=1000)
