import numpy as np
import sys
import matplotlib.pyplot as plt
import networkx as nx
from pydtmc import MarkovChain
import matplotlib.colors as mcolors

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
        self.visited_states = set()
        self.cluster_elements = self.assign_elements_to_clusters()

    def create_MC(self, transition_matrix, initial_state):
        num_clusters = transition_matrix.shape[0]
        self.labels = {i: chr(65 + i) for i in range(num_clusters)}
        self.transition_matrix = transition_matrix
        self.state = initial_state
        self.visited_states = {initial_state}

    def step_MC(self):
        if self.state is None or self.transition_matrix is None:
            raise ValueError("Markov Chain not initialized. Please run create_MC() first.")

        probabilities = self.transition_matrix[self.state].copy()
        for visited_state in self.visited_states:
            probabilities[visited_state] = 0

        if probabilities.sum() == 0:
            raise ValueError("All states have been visited. No more transitions possible.")

        probabilities = probabilities / probabilities.sum()
        new_state = np.random.choice(len(probabilities), p=probabilities)
        self.state = new_state
        self.visited_states.add(new_state)
        return self.state

    def simulate_MC(self, steps):
        if self.state is None or self.transition_matrix is None:
            raise ValueError("Markov Chain not initialized. Please run create_MC() first.")

        states = [self.state]
        try:
            for _ in range(steps):
                states.append(self.step_MC())
        except ValueError as e:
            print(e)
        return states

    def assign_elements_to_clusters(self):
        elements = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        num_clusters = self.params['Q']

        if num_clusters > len(elements):
            raise ValueError("Not enough elements to assign to clusters.")

        return {i: elements[i] for i in range(num_clusters)}

    def find_max(self):
        full_matrix = self.create_full_network_connectivity_matrix()
        exc_populations = self.Populations[0].get_Populations()

        row_start = 0
        max_values = []
        max_indices = []

        for i, source_pop in enumerate(exc_populations):
            pop_max_values = []
            pop_max_indices = []
            row_end = row_start + source_pop.size
            col_start = 0

            for j, target_pop in enumerate(exc_populations):
                col_end = col_start + target_pop.size

                for row in range(row_start, row_end):
                    row_slice = full_matrix[row, col_start:col_end]
                    max_index = np.argmax(row_slice)
                    row_max = row_slice[max_index]
                    pop_max_values.append(row_max)
                    pop_max_indices.append(j)

                col_start += target_pop.size

            max_values.append(pop_max_values)
            max_indices.append(pop_max_indices)
            row_start += source_pop.size

        return max_values, max_indices

    def create_markov_chain(self):
        max_values, max_indices = self.find_max()
        exc_populations = self.Populations[0].get_Populations()
        num_clusters = len(exc_populations)

        transition_matrix = np.zeros((num_clusters, num_clusters))

        for i in range(num_clusters):
            for max_val, target_idx in zip(max_values[i], max_indices[i]):
                transition_matrix[i, target_idx] = max_val

        return transition_matrix

    def plot_markov_chain(self, transition_matrix):
        fig, ax = plt.subplots(figsize=(10, 8))
        G = nx.DiGraph()
        exc_populations = self.Populations[0].get_Populations()

        epsilon = 1e-5
        log_prob_matrix = np.log(transition_matrix + epsilon)
        min_log_prob = np.min(log_prob_matrix[log_prob_matrix > -np.inf])
        max_log_prob = np.max(log_prob_matrix)

        for i, pop in enumerate(exc_populations):
            G.add_node(i, label=pop.name)

        edge_colors = []
        edge_widths = []
        for i in range(len(transition_matrix)):
            for j in range(len(transition_matrix[i])):
                if transition_matrix[i][j] > 0:
                    G.add_edge(i, j, weight=transition_matrix[i][j])
                    normalized_weight = (np.log(transition_matrix[i][j] + epsilon) - min_log_prob) / (max_log_prob - min_log_prob)
                    color = plt.cm.viridis(normalized_weight)
                    edge_colors.append(color)
                    edge_width = 2 if transition_matrix[i][j] == np.max(transition_matrix[i]) else 1
                    edge_widths.append(edge_width)

        pos = nx.circular_layout(G)
        nx.draw(G, pos, node_color='lightblue', with_labels=True, node_size=5000, font_size=15, ax=ax)
        edges = nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color=edge_colors, width=edge_widths, ax=ax)

        norm = mcolors.Normalize(vmin=min_log_prob, vmax=max_log_prob)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax)

        ax.set_title("Markov Chain Transition Diagram")
        plt.show()

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
