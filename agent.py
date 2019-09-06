import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1.0
        self.alpha = 0.01
        self.num_episodes = 1
        self.gamma = 1.0
    
    def get_epsilon_greedy_policy(self, state):
        policy = np.ones(self.nA) * self.epsilon / self.nA
        acts = self.Q[state]
        best_action = np.argmax(acts)
        policy[best_action] += 1 - self.epsilon
        return policy

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy = self.get_epsilon_greedy_policy(state)
        return np.random.choice(self.nA, p=policy)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #self.Q[state][action] += 1
        if not done:
            Qsa_next = np.max(self.Q[next_state])
        else:
            Qsa_next = 0
            self.num_episodes += 1
            self.epsilon  = 1. / self.num_episodes

        self.Q[state][action] += self.alpha * (reward + self.gamma * Qsa_next - self.Q[state][action])
                        