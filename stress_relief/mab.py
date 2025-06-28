import numpy as np
import random

class UCBBandit:
    """
    Multi-armed bandit using Upper Confidence Bound (UCB1).

    Attributes:
        n_arms (int): Number of arms.
        counts (np.ndarray): Number of times each arm was pulled.
        values (np.ndarray): Estimated mean reward for each arm.
        total_counts (int): Total number of pulls across all arms.
    """
    def __init__(self, mab_state):
        if mab_state is not None:
            self.n_arms = mab_state["n_arms"]
            self.counts = np.array(mab_state["counts"])
            self.values = np.array(mab_state["values"])
            self.total_counts = mab_state["total_counts"]
        else:
            self.n_arms = 5
            self.counts = np.zeros(self.n_arms)
            self.values = np.zeros(self.n_arms)
            self.total_counts = 0
        self.epsilon = 0.2
    
    def dump_states(self):
        return {
            "n_arms": int(self.n_arms),
            "counts": self.counts.tolist(),
            "values": self.values.tolist(),
            "total_counts": int(self.total_counts)
        }

    def select_arm(self):
        """
        Select an arm based on UCB1 criterion.
        """
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                # Ensure each arm is tried once
                return arm
        if random.random() < self.epsilon:
            return random.choice(range(self.n_arms))
        ucb_values = self.values + np.sqrt((2 * np.log(self.total_counts)) / self.counts)
        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        """
        Update counts and value estimates after receiving reward.
        """
        self.counts[arm] += 1
        self.total_counts += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = value + (reward - value) / n