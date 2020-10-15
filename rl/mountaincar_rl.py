import gym
import numpy as np


env = gym.make("MountainCar-v0").env
OS_size = [30,30]

# This is the discretized step we can make using a Q-table with size 30x30. By using such an approach, we can easily
# Find the state we are in in the Q table. (Done by subtracting the highest possihble values we can get to the lowest we can get and then deviding by the shape of the q table)
win_size = (env.observation_space.high - env.observation_space.low)/OS_size
# action is discrete(3)
Q_table = np.random.uniform(-2, 0, size =(OS_size + [env.action_space.n])) # size of 30,30,3
print(f'the shape of the Q table is {np.shape(Q_table)}')

def discrete_state(x):
    # this will get us the state as defined by the discretized step. For example, position can be 10 which is 10 steps from the lowest value we can get (for 1 discrete step)
    lowest = env.observation_space.low
    state = x
    highest = env.observation_space.high
    Qsize = [30]*len(highest)
    ds_step = (highest - lowest)/Qsize
    ds = (state - lowest)/(ds_step)
    return tuple(ds.astype(np.int))

ds = discrete_state(env.reset())
print(Q_table[ds])

def main():
   pass


if __name__ == "__main__":
    main()
