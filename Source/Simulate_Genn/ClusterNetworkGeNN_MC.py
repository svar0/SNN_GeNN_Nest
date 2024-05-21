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
        self.transition_matrix = None
        self.state = None
        self.visited_states = set()

    def create_MC(self, transition_matrix, initial_state):
        num_clusters = transition_matrix.shape[0]
        self.transition_matrix = transition_matrix
        self.state = initial_state
        self.visited_states = {initial_state}

    def step_MC(self):
        if self.state is None or self.transition_matrix is None:
            raise ValueError("Markov Chain not initialized")

        probabilities = self.transition_matrix[self.state].copy()
        for visited_state in self.visited_states:
            probabilities[visited_state] = 0

        if probabilities.sum() == 0:
            raise ValueError("No more transitions possible.")

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


    def plot_markov_chain(self, transition_matrix):
        fig, ax = plt.subplots(figsize=(10, 8))
        G = nx.DiGraph()

        num_clusters = transition_matrix.shape[0]
        self.labels = {i: str(i) for i in range(num_clusters)}

        for i in range(num_clusters):
            G.add_node(i, label=self.labels[i])

        edge_colors = []
        edge_widths = []
        edge_labels = {}
        for i in range(num_clusters):
            for j in range(num_clusters):
                if transition_matrix[i][j] > 0:
                    G.add_edge(i, j, weight=transition_matrix[i][j])
                    color = plt.cm.viridis(transition_matrix[i][j])
                    edge_colors.append(color)
                    edge_widths.append(2 * transition_matrix[i][j])
                    edge_labels[(i, j)] = f'{transition_matrix[i][j]:.2f}'

        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color='lightblue', with_labels=True, labels={i: str(i) for i in range(num_clusters)},
                node_size=4000, font_size=15, ax=ax)
        edges = nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color=edge_colors,
                                       width=edge_widths, connectionstyle="arc3,rad=0.1", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, ax=ax)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mcolors.Normalize(vmin=0, vmax=1))
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
