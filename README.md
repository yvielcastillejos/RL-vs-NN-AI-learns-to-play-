# RL-vs-NN-AI-learns-to-play-

In this repository, we will learn what the difference is when it comes to training a game with a Neural Network vs training a game with Reinforced Learning using Q-tables. We will ultimately decide which approach is the best for such a game by looking at different variables.

## Introduction:
Using a Neural Network approach and a Reinforcement Learning approach, we will train an AI to predict the best action our agent must take.

<img src = "https://www.novatec-gmbh.de/wp-content/uploads/1_mPGk9WTNNvp3i4-9JFgD3w.png" width = "500">

#### The Game
Mountain Car - TBA

### Neural Network Approach 
In the Neural Network Approach, we implement the following:
 - Data Acquisition through accessing the environment and thinking through the important information we can achieve/use from the environment (e.g. State observation values relative to the agent, reward per timestep, and hyperparameters such as the number of games)
 - Train the model using a Multi-Layer Perceptron with Pytorch. The design of the Neural Network is made through trial and error (i.e. finding which model gives the best accuracy). 
 - Using Cross Entropy as the loss function so that the error significantly affects how the loss function is optimized.
 - Plotting/Printing the training and validation accuracy and loss plots and seeing whether the model is successfully learning
 - Utilizing the model to predict an action that the environment can use and seeing whether the prediction gets the desired reward (i.e. do we get the results we want with the model)
 
### Q-Reinforcement Learning
In the RL Approach, we implement the Q-Learning algorithm by:
 - Accessing the environment and getting observation values
 - Implementing a Q-table.
 - Update the Q-table every timestep and figure out a Q-table that ultimately has values that make the agent "win" the game using the formula below.
 - predict am action and use it in the environment

We update the Q-table with this formula.
<img src ="https://github.com/yvielcastillejos/RL-vs-NN-AI-learns-to-play-/blob/main/index.png"  height = "100" width="500" >
## Results:
more TBA    
### Neural Network Results
For the Neural Network approach, we train a model with batched datasets gathered from the environment. We get the following:


<img src ="https://github.com/yvielcastillejos/RL-vs-NN-AI-learns-to-play-/blob/main/TrainvsValid.png" height = "250" width="250">

The graph is noisy due to the small batch size. This does not affect the essence of what the graph is trying to convey, however.
### Resources:
TBA
