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


out = []
with open("data_log.txt", "r") as file:
    for line in file:
        out.append(int(line))
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
        time.sleep(10)  # Wait for 20 seconds before the next request

# Create a thread for data fetching
data_fetch_thread = threading.Thread(target=fetch_data, daemon=True)



RAN = keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None)
Mem = deque()
# Define the number of possible outcomes (0-36)
num_outcomes = 36

MaxMem = 5000
batch = 32
# Create a simple Q-network
model = keras.Sequential([
    keras.layers.Dense(36, use_bias=True, kernel_initializer=RAN, bias_initializer=RAN, activation=LeakyReLU(alpha=0.1), input_shape=(num_outcomes,), name='input_layer'),
    keras.layers.Dense(36, use_bias=True, kernel_initializer=RAN, bias_initializer=RAN, activation=LeakyReLU(alpha=0.1), name='hidden_layer'),
    keras.layers.Dense(36, use_bias=True, kernel_initializer=RAN, bias_initializer=RAN, activation=LeakyReLU(alpha=0.1), name='hidden_layer2'),
    keras.layers.Dense(7, activation='linear', name='output_layer')
])

# Define the Q-network optimizer
optimizer = keras.optimizers.Adam(lr=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='mse')

def _load(x):
    if x!=0:
        model.load_weights("Agent" + str(x) +".h5")
        print("Loaded model sucessfully from disk")

def _save(x):
    model.save_weights("Agent" + str(x) +".h5")
    print("Saved model successfully to disk")

def save_mem(obs,act,rwd,n_o):
    if len(Mem) > MaxMem: Mem.popleft()
    Mem.append((obs,np.argmax(act),rwd,deepcopy(n_o)))

def replay():
    mini_batch = random.sample(Mem, batch)
    S  = np.array([d[0] for d in mini_batch])
    S1 = np.array([d[3] for d in mini_batch])

    temp = model.predict(S, verbose=0)
    temp1 = model.predict(S1, verbose=0)

    for i in range(batch):
        temp[i][mini_batch[i][1]] = mini_batch[i][2] + 0.9*np.max(temp1[i])

    model.train_on_batch(S,temp)

# Simulated environment (for demonstration purposes)
randomness = True
class RouletteEnvironment:
    def __init__(self):
        self.state = np.zeros(num_outcomes) #np.random.randint(num_outcomes, size=num_outcomes)  # Represents the state (previous numbers)
        self.num = np.random.randint(0, 36)

    def step(self, action, ine = None):
        global new_data
        # Simulate the roulette wheel and calculate reward based on the action
        # outcome = np.random.randint(0, num_outcomes)

        if ine is None:
            if randomness:
                outcome = np.random.randint(0, 36)
            else:
                # outcome = int(input("Predict is {}, input is: ".format(action)))
                # outcome = (self.num + 1 ) %37
                try:
                    print("AI predicted: "+ str(action))
                    while not new_data:
                        # Your main thread's tasks here
                        time.sleep(1)  # Sleep for 1 second or perform other tasks
                        
                    outcome = previous_data['data']['result']['outcome']['number']
                    new_data = False
                except KeyboardInterrupt:
                    outcome = int(input("Predict is {}, input is: ".format(action)))

            if outcome == 38:
                return None, None, True
            self.num = outcome
            if outcome == 37:
                _save(0)
                try:
                    print("AI predicted: "+ str(action))
                    while not new_data:
                        # Your main thread's tasks here
                        time.sleep(1)  # Sleep for 1 second or perform other tasks
                        
                    outcome = previous_data['data']['result']['outcome']['number']
                    new_data = False
                except KeyboardInterrupt:
                    outcome = int(input("Predict is {}, input is: ".format(action)))
                
            
            if not randomness:
                out.append(outcome)

        else:
            outcome = ine
        
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

        return deepcopy(self.state), reward, False
    
# Training loop (for demonstration)
num_episodes = 100

EXPLORE = 500
INIT_EP = 0.9999
FINL_EP = 0.0001

env = RouletteEnvironment()
state = deepcopy(env.state)

# Start the data fetching thread
data_fetch_thread.start()

_load(1)
for episode in range(num_episodes):
    rounds = 0
    total_reward = 0
    EPSILON = INIT_EP

    for step in range(500):  # In each episode, make predictions for 10 rounds
        rounds += 1
        # print(round, step, total_reward)
        q_values = model.predict(state.reshape(1, -1), verbose=0)

        action = np.argmax(q_values)

        if randomness:
            if random.random() < EPSILON:
                action = np.random.randint(0, 7)
                print("random act: {}".format(action), end="; ")        
            if EPSILON > FINL_EP:
                EPSILON -= (INIT_EP - FINL_EP)/EXPLORE

        next_state, reward, breakp = env.step(action)
        if breakp: 
            print("retrain")
            break
        if action != 6:
            total_reward += reward
        print("Sample Round: {}, predict: {}, outcome:{}, profit: {}".format(rounds, action, next_state[-1], total_reward))

        save_mem(state,q_values,reward,next_state)

        state = deepcopy(next_state)

        # Update the Q-network based on the Q-learning update rule
        if len(Mem) > batch:
            replay()
        target = reward + 0.9*np.max(model.predict(next_state.reshape(1, -1), verbose=0))
        q_values[0, action] = target
        model.fit(state.reshape(1, -1), q_values, epochs=1, verbose=0)

    randomness = False

    for i in deepcopy(out):
        rounds += 1
        # print(round, step, total_reward)
        q_values = model.predict(state.reshape(1, -1), verbose=0)
        action = np.argmax(q_values)
        next_state, reward, _ = env.step(action, i)
        if action != 6:
            total_reward += reward
        print("Sample Round: {}, predict: {}, outcome:{}, profit: {}".format(rounds, action, next_state[-1], total_reward))

        save_mem(state,q_values,reward,next_state)

        state = deepcopy(next_state)

        # Update the Q-network based on the Q-learning update rule
        if len(Mem) > batch:
            replay()
        target = reward + 0.9*np.max(model.predict(next_state.reshape(1, -1), verbose=0))
        q_values[0, action] = target
        model.fit(state.reshape(1, -1), q_values, epochs=1, verbose=0)

    # randomness = False
    # print("Sample Round: {}, predict: {}, outcome:{}, profit: {}".format(rounds, action, next_state[-1], total_reward))
    # print(f"Episode {episode + 1} - Total Reward: {total_reward}")
    # _save(1)
    # else: print("No save")

# To predict the next outcome given the current state
current_state = env.state
q_values = model.predict(current_state.reshape(1, -1))
predicted_action = np.argmax(q_values)
print(f"Predicted next outcome: {predicted_action}")
