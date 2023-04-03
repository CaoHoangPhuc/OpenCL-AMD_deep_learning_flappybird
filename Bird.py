import Agent
import cv2
import sys
sys.path.append("game/")
import random
import wrapped_flappy_bird as game
import numpy as np

Bird = Agent.Agent()

OBSERVE = 1000
EXPLORE = 100000.
INIT_EP = 0.5
FINL_EP = 0.0001
SAVE_SL = 3
LOAD_SL = 2

def preprocess(image):
    x_t = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return np.array([x_t])

def random_act(obs,x):
    if random.random() < x:
        if random.random() < 0.1:
            print("random_act ")
            return [0,1]
        else:
            return [1,0]
    else:
        return Bird.act(obs)

def play_flappy():

    TIME_S  = 1     
    EPSILON = INIT_EP

    Bird._load(LOAD_SL)

    F_obs, reward, terminal = flappy_game.frame_step([1,0])
    F_obs = preprocess(F_obs)
    Bird.act(F_obs)

    while 1:        

        obs = F_obs
        terminal = 0

        while terminal == 0:
            
            action = random_act(obs,EPSILON)
            ne_obs, reward, terminal = flappy_game.frame_step(action)
            ne_obs = preprocess(ne_obs)

            Bird.save_mem(obs,action,reward,ne_obs,terminal)          
            
            if (TIME_S > OBSERVE) and (EPSILON > FINL_EP):
                EPSILON -= (INIT_EP - FINL_EP)/EXPLORE
            
            if (TIME_S % 100 == 0):
                print(TIME_S,reward,np.argmax(action),round(EPSILON,5),round(
                np.amax(Bird.model.predict(np.array([obs]))[0]),5))
                
            if (TIME_S >= OBSERVE ):
                Bird.replay()
                
            if TIME_S %10000 == 0:
                Bird._save(SAVE_SL)

            TIME_S +=1
            obs = ne_obs

flappy_game = game.GameState()

play_flappy()
