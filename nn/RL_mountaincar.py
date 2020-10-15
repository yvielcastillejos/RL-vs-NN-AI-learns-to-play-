# Setting up the environment
import gym
import numpy as np
import random

env = gym.make('MountainCar-v0').env
print(env.action_space)
ascore = -200
episodes = 100
tsteps = 300
scores = []
observation_q = []
acceptedscores = []
traindata = []
avgscore = -110
# actions are 123, left, stop, right
# starting state is [-0.6, -0.4] as position,velocity
# reward of -1 if the position of the agent is less than 0.5 for each timestep

def main():
    observation = env.reset()
    for i_episode in range(episodes):
        i = 0
        observation = env.reset()
        print("-------------------------------------------------------------------")
        for t in range(tsteps):
            action = np.argmax(model(observation))
            observation_q.append([observation,action])
            observation, reward, done, _ = env.step(action)
            #print(observation)
            #print(score)
            score += reward
            if done:
                #print(bool(observation[0] >= env.goal_position))
                #print(env.goal_position)
                break
     return

env.close()
