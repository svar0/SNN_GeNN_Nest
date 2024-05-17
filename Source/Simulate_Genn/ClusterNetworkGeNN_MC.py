import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from Defaults import defaultSimulate as default
from Helper import ClusterModelGeNN
from Helper import GeNN_Models

class ClusterNetworkGeNN_MC(ClusterModelGeNN.ClusteredNetworkGeNN_Timing):
    def __init__(self, defaultValues, parameters, batch_size=1, NModel="LIF"):
        super().__init__(defaultValues, parameters, batch_size, NModel)
        self.labels = None
        self.transition_matrix = None
        self.state = None
    """
            Creates a Markov Chain and returns  the labels for each cluster, the transition matrix, and the state.

            Steps for the Markov Chain to the next state based on the current state.
            The new state is not the same as the current state and returns the new state
            
            Simulates the Markov Chain for a given number of steps and returns the list of states.
    """
    def create_MC(self, transition_matrix, initial_state):

        num_clusters = transition_matrix.shape[0]
        self.labels = {i: chr(65 + i) for i in range(num_clusters)}
        self.transition_matrix = transition_matrix
        self.state = initial_state

    def step_MC(self):

        if self.state is None or self.transition_matrix is None:
            raise ValueError("Markov Chain not initialized. Please run create_MC() first.")

        probabilities = self.transition_matrix[self.state].copy()
        probabilities[self.state] = 0  # Prevent transitioning to the same state
        probabilities = probabilities / probabilities.sum()  # Normalize the probabilities

        new_state = np.random.choice(len(probabilities), p=probabilities)
        self.state = new_state
        return self.state

    def simulate_MC(self, steps):

        if self.state is None or self.transition_matrix is None:
            raise ValueError("Markov Chain not initialized. Please run create_MC() first.")

        states = [self.state]
        for _ in range(steps):
            states.append(self.step_MC())
        return states


if __name__ == "__main__":
    sys.path.append("..")
    from Defaults import defaultSimulate as default

    EI_cluster_mc = ClusterNetworkGeNN_MC(default, {'n_jobs': 4, 'warmup': 500, 'simtime': 1200, 'stim_clusters': [3],
                                                    'stim_amp': 2.0, 'stim_starts': [600.], 'stim_ends': [1000.],
                                                    'matrixType': "PROCEDURAL_GLOBALG"})

    num_clusters = 3
    transition_matrix = np.random.dirichlet(np.ones(num_clusters), size=num_clusters)
    initial_state = 0

    EI_cluster_mc.create_MC(transition_matrix, initial_state)
    print(f"Initial State: {EI_cluster_mc.state}")
    print(f"Transition Matrix:\n{EI_cluster_mc.transition_matrix}")

    new_state = EI_cluster_mc.step_MC()
    print(f"New State after one step: {new_state}")

    steps = 2
    states = EI_cluster_mc.simulate_MC(steps)
    print(f"States after {steps} steps: {states}")

    spikes = EI_cluster_mc.create_and_simulate()
    print(EI_cluster_mc.get_parameter())
    plt.figure()
    plt.plot(spikes[0][0, :], spikes[0][1, :], '.', markersize=1)
    plt.show()
