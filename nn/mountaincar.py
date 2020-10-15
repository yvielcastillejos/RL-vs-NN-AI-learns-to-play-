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
        observation_q = []
        score = 0
        print("-------------------------------------------------------------------")
        for t in range(tsteps):
#            env.render()
            #print(t)
#            print(observation)
            # We want it in the form [observation, future action] where action is preferrably one hot coded
            if t<20:
              action = 2
            elif t<50:
              action = 0
            elif t < 80:
              action = 2
            elif t < 150: 
              action = 0
            elif t<220:
              action = 2#random.randrange(0,3)
            elif t< 300:
              action = 0
            elif t <350:
              action = 1
            elif t< 400:
              action = 1
            elif t < 250 and i==0:
              action = 2 #random.randrange(0,3)
              i = 1
            else:
              action = random.randrange(0,3)
            observation_q.append([observation,action])
            observation, reward, done, _ = env.step(action)
            #print(observation) 
            #print(score)
            score += reward   
            if done:
                #print(bool(observation[0] >= env.goal_position))
                #print(env.goal_position)
                ##print(observation)
                #print(done)
                #print(f"episode finished after {t+1} timesteps")
                break
        if score >= ascore:
            acceptedscores.append(score)
            for data, action in observation_q:
                if action == 0:
                    a = np.array([1,0,0])
                if action == 1:
                    a = np.array([0,1,0])
                if action == 2:
                    a = np.array([0,0,1])
                traindata.append([data, a])
        scores.append(score)
        print(np.array(traindata))
        np.save("data.npy", np.array(traindata))
    return

if __name__ == '__main__':
    main()

