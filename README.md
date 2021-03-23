# Deep-Cure-Learning
INF581 Reinforcement Learning Project. This project is focused on creating a simplified version of the Plague Inc. game as an environment and testing it against multiple reinforcement learning methods. Plague Inc. is, traditionally, a game where the player controls a disease aiming to take over the world. The player must input mutations and "skill points" to survive and infect the world. Recently, a cure mode also exists within the game. We propose a simple yet expressive environment that simulates the spreading of a disease in a country. Inverse to Plague Inc., the player chooses measures like obligatory face masks or curfew to stop the spread. We used three different agents to test this environment: Q Learning agent, Deep Q Learning (REINFORCE agent) and (1+1)-ES agent. 

## Repository Structure

The repository's important files are located as follows:

```bash
├── deep_cure_learning
│   ├── envs
│   │   └── deep_cure_env.py
│   ├── comparison.py
│   ├── reinforce_agent.py (theta.npy)
│   ├── no_action_agent.py 
│   ├── plotting.py
│   ├── q_table_agent.py (qtable-100.npy, qtable-1000.npy)
│   ├── stable.py (best_model.zip, dqn_stable.zip)
│   └── saes_agent.py (saes-theta.npy, saes-theta2.npy)
├── README.md
└── .gitignore
```
These files are the most important to our project.
The deep_cure_env.py is the file that dictates our environment and how it evolves in time. This is the main contribution of our work.
The files in the deep_cure_learning directory are all either agents or plotting and comparison scripts that we used to evaluate how the agents interacted with our custom environment. The _.npy_ filesin parenthesis following the agent scripts are the trained model values. 
To reproduce the work, please use `comparison.py` and `plotting.py`. A fixed seed was implemented to this end. Furthermore, these scripts were used to create the figures in the report.

## How to run code

Before running any files please install all requirements by running the command `pip install -r deep_cure_learning/requirements.txt`.
Each agent python file can be run to update their respective agent's model. 
However, to get the comparison between models of our environment, please run the `comparison.py` file. 
This will create a table with the metrics of performance as explained in our report.
