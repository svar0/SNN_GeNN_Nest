from pygenn import genn_model, genn_wrapper
import numpy as np
import time
import pickle
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from pydtmc import MarkovChain


sys.path.append("..")
import signal
import math
from Helper import ClusterHelper
from Helper import GeneralHelper
from Helper import GeNNHelper
from Helper import GeNN_Models
from Helper import ClusterModelBase


class ClusteredNetworkGeNN(ClusterModelBase.ClusteredNetworkBase):
    """
    Creates an object with functions to create neuron populations,
    stimulation devices and recording devices for an EI-clustered network.
    Provides also function to initialize PyGeNN, simulate the network and
    to grab the spike data.
    """

    def __init__(self, defaultValues, parameters, batch_size=1, NModel="LIF"):
        """
        Creates an object with functions to create neuron populations,
        stimulation devices and recording devices for an EI-clustered network.
        Initializes the object. Creates the attributes Populations, RecordingDevices and
        Currentsources to be filled during network construction.
        Attribute params contains all parameters used to construct network.

        Parameters:
            defaultValues (module): A Module which contains the default configuration
            parameters (dict):      Dictionary with parameters which should be modified from their default values
        """
        super().__init__(defaultValues, parameters)
        self.model = None
        self.Populations = []
        self.params['batch_size'] = batch_size
        self.duration_timesteps = 0
        self.runs = 0
        self.NeuronModel = NModel
        self.Timing = {'Sim': [], 'Download': []}
        self.cluster_elements = self.assign_elements_to_clusters()
        self.synapses = []
        self.synapse_ref = {}

    def clean_network(self):
        """
        Creates empty attributes of a network.
        """
        self.model = None
        self.Populations = []
        self.Timing = {'Sim': [], 'Download': []}

    def setup_GeNN(self, Name="EICluster"):
        """ Initializes a empty GeNN model.
        Reset the NEST kernel and pass parameters to it.
        Updates randseed of parameters to the actual used one if none is supplied.
        """
        self.model = genn_model.GeNNModel("float", Name, generateEmptyStatePushPull=False,
                                          generateExtraGlobalParamPull=False)
        self.model.dT = self.params.get('dt')
        self.params['randseed'] = self.params.get('randseed', np.random.randint(1000000))
        self.model._model.set_seed(self.params.get('randseed'))
        self.model._model.set_merge_postsynaptic_models(True)
        self.model._model.set_default_narrow_sparse_ind_enabled(True)

        assert self.params['batch_size'] >= 1, "batch_size has to be 1 or greater"
        if self.params['batch_size'] > 1:
            self.model.batch_size = self.params['batch_size']

    def create_populations(self):
        """
        Creates Q excitatory and inhibitory neuron populations with the parameters of the network.
        """
        # make sure number of clusters and units are compatible
        assert self.params['N_E'] % self.params['Q'] == 0, 'N_E needs to be evenly divisible by Q'
        assert self.params['N_I'] % self.params['Q'] == 0, 'N_I needs to be evenly divisible by Q'

        # network parameters

        if self.params['I_th_E'] is None:
            I_xE = self.params['I_xE']
        else:
            I_xE = self.params['I_th_E'] * (self.params['V_th_E'] - self.params['E_L']) / self.params['tau_E'] * \
                   self.params['C_m']

        if self.params['I_th_I'] is None:
            I_xI = self.params['I_xI']
        else:
            I_xI = self.params['I_th_I'] * (self.params['V_th_I'] - self.params['E_L']) / self.params['tau_I'] * \
                   self.params['C_m']

        if 'iaf_psc_exp' in self.params['neuron_type']:
            pass
        else:
            assert 'iaf_psc_exp' in self.params['neuron_type'], "iaf_psc_exp neuron model is the only implemented model"

        E_neuron_params = {'Vrest': self.params['E_L'], 'C': self.params['C_m'], 'TauM': self.params['tau_E'],
                           'TauRefrac': self.params['t_ref'], 'Vthresh': self.params['V_th_E'],
                           'Vreset': self.params['V_r'],
                           'Ioffset': I_xE}
        E_neuron_init = {
            "RefracTime": -0.1
        }
        I_neuron_params = {'Vrest': self.params['E_L'], 'C': self.params['C_m'], 'TauM': self.params['tau_I'],
                           'TauRefrac': self.params['t_ref'], 'Vthresh': self.params['V_th_I'],
                           'Vreset': self.params['V_r'],
                           'Ioffset': I_xI}
        I_neuron_init = {
            "RefracTime": -0.1
        }

        if self.params['V_m'] == 'rand':
            T_0_E = self.params['t_ref'] + ClusterHelper.FPT(self.params['tau_E'], self.params['E_L'], I_xE,
                                                             self.params['C_m'], self.params['V_th_E'],
                                                             self.params['V_r'])
            if np.isnan(T_0_E):
                T_0_E = 10.
            T_0_I = self.params['t_ref'] + ClusterHelper.FPT(self.params['tau_I'], self.params['E_L'], I_xI,
                                                             self.params['C_m'], self.params['V_th_I'],
                                                             self.params['V_r'])
            if np.isnan(T_0_I):
                T_0_I = 10.

        else:
            E_neuron_init["V"] = self.params['V_m']
            I_neuron_init["V"] = self.params['V_m']

        # create the neuron populations
        E_pops = []
        I_pops = []
        for q in range(self.params['Q']):
            if self.params['V_m'] == 'rand':
                E_neuron_init["V"] = [ClusterHelper.V_FPT(self.params['tau_E'], self.params['E_L'], I_xE,
                                                          self.params['C_m'], T_0_E * np.random.rand(),
                                                          self.params['V_th_E'], self.params['t_ref']) for i in
                                      range(int(self.params['N_E'] / self.params['Q']))]
            E_pops.append(self.model.add_neuron_population("Egif" + str(q), int(self.params['N_E'] / self.params['Q']),
                                                           self.NeuronModel, E_neuron_params, E_neuron_init))

        for q in range(self.params['Q']):
            if self.params['V_m'] == 'rand':
                I_neuron_init["V"] = [ClusterHelper.V_FPT(self.params['tau_I'], self.params['E_L'], I_xI,
                                                          self.params['C_m'], T_0_I * np.random.rand(),
                                                          self.params['V_th_E'], self.params['t_ref']) for i in
                                      range(int(self.params['N_I'] / self.params['Q']))]
            I_pops.append(self.model.add_neuron_population("Igif" + str(q), int(self.params['N_I'] / self.params['Q']),
                                                           self.NeuronModel, I_neuron_params, I_neuron_init))
        self.Populations = [GeNNHelper.SuperPopulation(E_pops, "Exc."), GeNNHelper.SuperPopulation(I_pops, "Inh.")]
        print(f"Creating {len(E_pops)} excitatory and {len(I_pops)} inhibitory populations.")
    def connect(self):
        """ Connects the excitatory and inhibitory populations with each other in the EI-clustered scheme
        """
        #  self.Populations[0] -> Excitatory super-population
        #  self.Populations[1] -> Inhibitory super-population
        # connectivity parameters
        js = self.params['js']  # connection weights
        N = self.params['N_E'] + self.params['N_I']  # total units

        delaySteps = int((self.params['delay'] + 0.5 * self.model.dT) // self.model.dT)
        psc_E = {"tau": self.params['tau_syn_ex']}  # synaptic time constant
        psc_I = {"tau": self.params['tau_syn_in']}  # synaptic time constant

        # if js are not given compute them so that sqrt(K) spikes equal v_thr-E_L and rows are balanced
        if np.isnan(js).any():
            js = ClusterHelper.calc_js(self.params)
        js *= self.params['s']

        # jminus is calculated so that row sums remain constant
        if self.params['Q'] > 1:
            jminus = (self.params['Q'] - self.params['jplus']) / float(self.params['Q'] - 1)
        else:
            self.params['jplus'] = np.ones((2, 2))
            jminus = np.ones((2, 2))

        # define the synapses and connect the populations
        # EE
        j_ee = js[0, 0] / np.sqrt(N)
        if self.params['fixed_indegree']:
            K_EE = int(self.params['ps'][0, 0] * self.params['N_E'] / self.params['Q'])
            print('K_EE: ', K_EE)
            conn_params_EE = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                          {"colLength": K_EE})

        else:
            conn_params_EE = genn_model.init_connectivity("FixedProbabilityNoAutapse",

                                                          {"prob": self.params['ps'][0, 0]})

        for i, pre in enumerate(self.Populations[0].get_Populations()):
            for j, post in enumerate(self.Populations[0].get_Populations()):
                if i == j:
                    # same cluster
                    self.model.add_synapse_population(str(i) + "EE" + str(j), self.params['matrixType'], delaySteps,
                                                      pre, post,
                                                      "StaticPulse", {}, {"g": self.params['jplus'][0, 0] * j_ee}, {},
                                                      {},
                                                      "ExpCurr", psc_E, {}, conn_params_EE
                                                      )
                else:
                    self.model.add_synapse_population(str(i) + "EE" + str(j), self.params['matrixType'], delaySteps,
                                                      pre, post,
                                                      "StaticPulse", {}, {"g": jminus[0, 0] * j_ee}, {},
                                                      {},
                                                      "ExpCurr", psc_E, {}, conn_params_EE
                                                      )

        # EI
        j_ei = js[0, 1] / np.sqrt(N)
        if self.params['fixed_indegree']:
            K_EI = int(self.params['ps'][0, 1] * self.params['N_I'] / self.params['Q'])
            print('K_EI: ', K_EI)
            conn_params_EI = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                          {"colLength": K_EI})
        else:
            conn_params_EI = genn_model.init_connectivity("FixedProbability",
                                                          {"prob": self.params['ps'][0, 1]})
        for i, pre in enumerate(self.Populations[1].get_Populations()):
            for j, post in enumerate(self.Populations[0].get_Populations()):
                if i == j:
                    # same cluster
                    self.model.add_synapse_population(str(i) + "EI" + str(j), self.params['matrixType'], delaySteps,
                                                      pre, post,
                                                      "StaticPulse", {}, {"g": j_ei * self.params['jplus'][0, 1]}, {},
                                                      {},
                                                      "ExpCurr", psc_I, {}, conn_params_EI
                                                      )
                else:
                    self.model.add_synapse_population(str(i) + "EI" + str(j), self.params['matrixType'], delaySteps,
                                                      pre, post,
                                                      "StaticPulse", {}, {"g": j_ei * jminus[0, 1]}, {},
                                                      {},
                                                      "ExpCurr", psc_I, {}, conn_params_EI
                                                      )
        # IE
        j_ie = js[1, 0] / np.sqrt(N)

        if self.params['fixed_indegree']:
            K_IE = int(self.params['ps'][1, 0] * self.params['N_E'] / self.params['Q'])
            print('K_IE: ', K_IE)
            conn_params_IE = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                          {"colLength": K_IE})
        else:
            conn_params_IE = genn_model.init_connectivity("FixedProbability",
                                                          {"prob": self.params['ps'][1, 0]})
        for i, pre in enumerate(self.Populations[0].get_Populations()):
            for j, post in enumerate(self.Populations[1].get_Populations()):
                if i == j:
                    # same cluster
                    self.model.add_synapse_population(str(i) + "IE" + str(j), self.params['matrixType'], delaySteps,
                                                      pre, post,
                                                      "StaticPulse", {}, {"g": j_ie * self.params['jplus'][1, 0]}, {},
                                                      {},
                                                      "ExpCurr", psc_E, {}, conn_params_IE
                                                      )
                else:
                    self.model.add_synapse_population(str(i) + "IE" + str(j), self.params['matrixType'], delaySteps,
                                                      pre, post,
                                                      "StaticPulse", {}, {"g": j_ie * jminus[1, 0]}, {}, {},
                                                      "ExpCurr", psc_E, {}, conn_params_IE
                                                      )

        # II
        j_ii = js[1, 1] / np.sqrt(N)
        if self.params['fixed_indegree']:
            K_II = int(self.params['ps'][1, 1] * self.params['N_I'] / self.params['Q'])
            print('K_II: ', K_II)
            conn_params_II = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                          {"colLength": K_II})
        else:
            conn_params_II = genn_model.init_connectivity("FixedProbability",
                                                          {"prob": self.params['ps'][1, 1]})
        for i, pre in enumerate(self.Populations[1].get_Populations()):
            for j, post in enumerate(self.Populations[1].get_Populations()):
                if i == j:
                    # same cluster
                    self.model.add_synapse_population(str(i) + "II" + str(j), self.params['matrixType'], delaySteps,
                                                      pre, post,
                                                      "StaticPulse", {}, {"g": j_ii * self.params['jplus'][1, 1]}, {},
                                                      {},
                                                      "ExpCurr", psc_I, {}, conn_params_II
                                                      )
                else:
                    self.model.add_synapse_population(str(i) + "II" + str(j), self.params['matrixType'], delaySteps,
                                                      pre, post,
                                                      "StaticPulse", {}, {"g": j_ii * jminus[1, 1]}, {}, {},
                                                      "ExpCurr", psc_I, {}, conn_params_II
                                                      )

        print('Js: ', js / np.sqrt(N))

    def create_learning_synapses(self):

        delaySteps = int((self.params['delay'] + 0.5 * self.model.dT) // self.model.dT)
        psc_E = {"tau": self.params['tau_syn_ex']}


        stdp_params = {"tau": 30.0,
                       "rho": 0.1,
                       "eta": 0.0002,
                       "wMin": -10.0,
                       "wMax": 10.0}

        # define the synapses and connect the populations
        # EE
        if self.params['fixed_indegree']:
            K_EE = int(self.params['ps'][0, 0] * self.params['N_E'] / self.params['Q'])
            print('K_EE: ', K_EE)
            conn_params_EE = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                          {"colLength": K_EE})

        else:
            conn_params_EE = genn_model.init_connectivity("FixedProbabilityNoAutapse",

                                                          {"prob": self.params['ps'][0, 0]})

        asymmetric_stdp = GeNN_Models.define_symmetric_stdp()
        for i, pre in enumerate(self.Populations[0].get_Populations()):
            for j, post in enumerate(self.Populations[0].get_Populations()):
                if  j==i:
                    continue
                # else:
                #     if j == (i+1):
                #         syn_dict = {"g": 0.2}
                #     else:
                #         syn_dict = {"g": 0.}
                synapse = self.model.add_synapse_population(str(i) + "STDP" + str(j), "SPARSE_INDIVIDUALG", delaySteps,
                                                          pre, post,
                                                          asymmetric_stdp, stdp_params, {"g": 0.009}, {},
                                                          {},
                                                          "ExpCurr", psc_E, {}, conn_params_EE
                                                          )
                #print(f"Creating STDP synapse between {pre.name} and {post.name}")
                self.synapses.append(synapse)
                self.synapse_ref[synapse] = (pre, post)
                synapse.weight_recording_enabled = True

    def assign_elements_to_clusters(self):
        elements = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        num_clusters = self.params['Q']

        if num_clusters > len(elements):
            raise ValueError("Not enough elements to assign to clusters.")

        return {i: elements[i] for i in range(num_clusters)}

    def create_markov_chain_transition_matrix(self, num_clusters):
        transition_matrix = np.random.dirichlet(np.ones(num_clusters), size=num_clusters)
        mc = MarkovChain(transition_matrix)
        print("Is the Markov Chain ergodic?", mc.is_ergodic)

        absorbing_states = []
        for i in range(num_clusters):
            if transition_matrix[i, i] == 1.0:
                absorbing_states.append(i)

        if absorbing_states:
            print("Absorbing states detected:", absorbing_states)
        else:
            print("No absorbing states detected.")

        return transition_matrix

    def generate_markov_chain_sequences(self, num_sequences, num_clusters):
        transition_matrix = self.create_markov_chain_transition_matrix(num_clusters)
        mc = MarkovChain(transition_matrix)
        sequences = []
        cluster_elements = self.assign_elements_to_clusters()

        for _ in range(num_sequences):
            num_steps = max(num_clusters, 2)
            initial_state = np.random.randint(0, num_clusters)
            sequence = mc.simulate(num_steps, initial_state=initial_state)
            sequence_labels = []

            for state in sequence:
                state = int(state)
                if 0 <= state < num_clusters:
                    sequence_labels.append(cluster_elements[state])
                else:
                    raise KeyError(f"State {state} is out of range for cluster elements")

            sequences.append(''.join(sequence_labels))

        return sequences


    def create_stimulation(self, sequence):
        cluster_stimulus = GeNN_Models.define_ClusterStim()

        stim_starts = []
        stim_ends = []
        current_start = self.params['warmup']

        for idx, _ in enumerate(sequence):
            stim_starts.append(current_start)
            stim_ends.append(current_start + self.params['stim_duration'])
            current_start += self.params['stim_duration'] + self.params['inter_stim_delay']

        for ii, element in enumerate(sequence):
            cluster_index = ord(element) - ord('A')
            if cluster_index < len(self.Populations[0].get_Populations()):
                self.model.add_current_source(
                    f"Stim_{element}_{ii}",
                    cluster_stimulus,
                    self.Populations[0].get_Populations()[cluster_index],
                    {"t_onset": stim_starts[ii],
                     "t_offset": stim_ends[ii],
                     "strength": self.params['stim_amp']}, {}
                )
                print(f"Stimulating cluster {cluster_index} ({element}) from {stim_starts[ii]} to {stim_ends[ii]}")

    def make_synapse_matrices(self):
        self.synapse_matrices = {}
        self.connectivity_matrices = {}

        for synapse in self.synapses:
            pre, post = self.synapse_ref[synapse]
            pre_pop_size = pre.size
            post_pop_size = post.size

            synapse.pull_var_from_device('g')
            synapse.pull_connectivity_from_device()

            gs = synapse.get_var_values('g')
            pre_inds = synapse.get_sparse_pre_inds()
            post_inds = synapse.get_sparse_post_inds()
            weight_matrix = np.zeros((pre_pop_size, post_pop_size), dtype=np.float32)

            for idx in range(len(gs)):
                weight_matrix[pre_inds[idx], post_inds[idx]] = gs[idx]

            synapse_name = f"{pre.name}_to_{post.name}"
            self.synapse_matrices[synapse_name] = weight_matrix

    def create_full_network_connectivity_matrix(self):
        exc_populations = self.Populations[0].get_Populations()
        total_neurons = sum(pop.size for pop in exc_populations)
        full_matrix = np.zeros((total_neurons, total_neurons), dtype=np.float32)

        row_start = 0
        for i, source_pop in enumerate(exc_populations):
            col_start = 0
            for j, target_pop in enumerate(exc_populations):
                matrix_name = f"{source_pop.name}_to_{target_pop.name}"
                matrix = self.synapse_matrices.get(matrix_name, np.zeros((source_pop.size, target_pop.size), dtype=np.float32))
                if np.any(matrix < 0):
                    print(f"Negative values from {source_pop.name} to {target_pop.name}")
                full_matrix[row_start:row_start + source_pop.size, col_start:col_start + target_pop.size] = matrix
                col_start += target_pop.size
            row_start += source_pop.size
        return full_matrix

    def display_full_network_connectivity_matrix(self):
        full_matrix = self.create_full_network_connectivity_matrix()
        plt.figure(figsize=(10, 8))
        plt.imshow(full_matrix, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Full Network Synaptic Weight Matrix')
        plt.xlabel('Neuron ID (Post-synaptic)')
        plt.ylabel('Neuron ID (Pre-synaptic)')
        plt.show()

    def normalize_matrix(self, matrix):
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = matrix / row_sums
        return normalized_matrix

    def display_full_normalized_network_connectivity_matrix(self):
        full_matrix = self.create_full_network_connectivity_matrix()
        normalized_matrix = self.normalize_matrix(full_matrix)
        plt.figure(figsize=(10, 8))
        plt.imshow(normalized_matrix, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Full Network Synaptic Weight Matrix (Normalized)')
        plt.xlabel('Neuron ID (Post-synaptic)')
        plt.ylabel('Neuron ID (Pre-synaptic)')
        plt.show()

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

    def create_recording_devices(self):
        """
        Activate spike recording in all neuron populations created by create_populations
        """
        # Enable spike recording
        for Pop in self.Populations[0].get_Populations():
            Pop.spike_recording_enabled = True
        for Pop in self.Populations[1].get_Populations():
            Pop.spike_recording_enabled = True

    def build_model(self, Force_rebuild=False):
        self.model.build(force_rebuild=Force_rebuild)

    def load_model(self, GPUspecificConstraint=50000 * 625000):  # Neurons times simulation steps which fit at the GPU
        for ii in range(0, 12):
            duration_timesteps = int(
                math.ceil((self.params['warmup'] + self.params['simtime']) / ((2 ** ii) * self.model.dT)))
            if (duration_timesteps * (self.params['N_E'] + self.params['N_I'])) < GPUspecificConstraint:
                break
        runs = 2 ** ii

        for ii in range(0, 5):
            try:
                self.model.load(num_recording_timesteps=duration_timesteps)
            except Exception as e:
                pass
            if self.model._loaded:
                runs *= 2 ** ii
                break
            duration_timesteps = int(math.ceil(duration_timesteps / 2))
            self.setup_network()
            self.build_model(Force_rebuild=False)
        self.duration_timesteps = duration_timesteps
        self.runs = runs

    def simulate_one_section(self):
        """
        Simulates network for a period of warmup+simtime
        """
        if self.duration_timesteps == 0:
            pass
        else:
            for jj in range(self.duration_timesteps):
                self.model.step_time()

    def get_spiketimes_section(self, timeZero = 0):
        """
        Extracts spikes of all populations created in create_populations.
        Cuts the warmup period away (only if the section contained the warmup period) and sets time relative to end
        of warmup. Ids 1:N_E correspond to excitatory neurons, N_E+1:N_E+N_I correspond to inhibitory neurons.

        Returns:
            List of spiketimes (np.array): Row 0: spiketimes, Row 1: neuron ID. Each list entry corresponds to
            1 simulation of the batch
        """
        # Download recording data
        Spiketimes = GeNNHelper.extractSpiketimes(self.model, self.params, self.Populations, timeZero=timeZero)
        return Spiketimes

    def simulate_and_get_recordings(self, timeZero = 0):
        SectionResults = []
        for ii in range(self.runs):
            startSim = time.time()
            self.simulate_one_section()
            endsimulate = time.time()
            SectionResults.append(self.get_spiketimes_section(timeZero=timeZero))
            endPullSpikes = time.time()
            self.Timing['Sim'].append(endsimulate - startSim)
            self.Timing['Download'].append(endPullSpikes - endsimulate)
        StartCreateNESTRep = time.time()
        if self.duration_timesteps == 0:
            spiketimesBatch = []
        else:
            TuplesConcat = [tuple(res[ii] for res in SectionResults) for ii in range(self.model.batch_size)]
            spiketimesBatch = [np.hstack(TuplesConcat[ii]) for ii in range(self.model.batch_size)]
            spiketimesBatch = [spikes[:, spikes[0, :] <= (self.params['warmup'] + self.params['simtime'])]
                               for spikes in spiketimesBatch]
        EndCreateNESTRep = time.time()
        self.Timing['Download'][-1] += (EndCreateNESTRep - StartCreateNESTRep)
        return spiketimesBatch

    def get_parameter(self):
        """
        Return:
            parameters (dict): Dictionary with all parameters for the simulation / network creation.
        """
        return self.params

    def setup_network(self):
        """
        Initializes GeNN model and creates the network, ready to be simulated.
        """
        self.setup_GeNN()
        self.create_populations()
        self.connect()
        self.create_learning_synapses()
        self.create_recording_devices()
        self.create_stimulation()

    def create_and_simulate(self):
        """
        Creates the EI-clustered network and simulates it with the parameters supplied in the object creation.
        Returns:
            list of spiketimes (np.array):  Row 0: spiketimes, Row 1: neuron ID.
                                    Ids 1:N_E correspond to excitatory neurons,
                                    N_E+1:N_E+N_I correspond to inhibitory neurons.
        """
        self.setup_network()
        self.build_model()
        self.load_model(GPUspecificConstraint=50000 * 625000)
        return self.simulate_and_get_recordings()

    def get_recordings(self):
        return self.simulate_and_get_recordings()

    def get_firing_rates(self, spiketimes=None):
        """
        Calculates the firing rates of all excitatory neurons and the firing rates of all inhibitory neurons
        created by self.create_populations. If spiketimes are not supplied, they get extracted.
        Parameters:
            spiketimes: (optional, np.array 2xT)   spiketimes of simulation
        Returns:
            list of (e_rate, i_rate) average firing rate of excitatory/inhibitory neurons (spikes/s) for the batches
        """
        if spiketimes is None:
            spiketimes = self.get_recordings()
        rates = [super(ClusteredNetworkGeNN, self).get_firing_rates(spiketimes_single_batch)
                 for spiketimes_single_batch in spiketimes]

        return rates

    def get_populations(self):
        return self.Populations[0], self.Populations[1]


class ClusteredNetworkGeNN_Timing(ClusteredNetworkGeNN):
    """
    Adds to EI clustered network:
        Measurement of runtime (attribute Timing)
        Changeable ModelBuildPipeline (list of functions)
        Firing rate estimation of exc. and inh. neurons
        Functions to save connectivity and create connectivity from file
    """

    def __init__(self, defaultValues, parameters, batch_size=1, NModel="LIF"):
        """
        Creates an object with functions to create neuron populations,
        stimulation devices and recording devices for an EI-clustered network.
        Initializes the object. Creates the attributes Populations, RecordingDevices and
        Currentsources to be filled during network construction.
        Attribute params contains all parameters used to construct network. ClusteredNetworkNEST_Timing objects
        measure the timing of the simulation and offer more functions than the base class.
        Parameters:
            defaultValues (module): A Module which contains the default configuration
            parameters (dict):      Dictionary with parameters which should be modified from their default values
        """
        super().__init__(defaultValues, parameters, batch_size=batch_size, NModel=NModel)
        self.ModelBuildPipeline = [self.setup_GeNN, self.create_populations,
                                   self.create_recording_devices, self.connect]

    def setup_network(self):
        """
        Initializes NEST and creates the network in NEST, ready to be simulated.
        Functions saved in ModelBuildPipeline are executed.
        nest.Prepare is executed in this function.
        """
        startbuild = time.time()
        for func in self.ModelBuildPipeline:
            func()
        endbuild = time.time()
        self.Timing['Build'] = endbuild - startbuild

    def build_model(self):
        startcompile = time.time()
        super().build_model()
        endcompile = time.time()
        self.Timing['Compile'] = endcompile - startcompile

    def load_model(self, GPUspecificConstraint=50000 * 625000):
        startLoad = time.time()
        super().load_model(GPUspecificConstraint=GPUspecificConstraint)
        endLoad = time.time()
        self.Timing['Load'] = endLoad - startLoad

    def set_model_build_pipeline(self, Pipeline):
        """
        Sets the ModelBuildPipeline.
        Parameters:
            Pipeline (list of functions):   ordered list of functions executed to build the network model
        """
        self.ModelBuildPipeline = Pipeline

    def get_timing(self):
        """
        Gets Timing information of simulation.
        Returns:
            Dictionary with the timing information of the different simulation phases in seconds.
        """
        return self.Timing

    def get_simulation(self, PathSpikes=None, timeout=None, GPUspecificConstraint=50000 * 625000):
        """
        Creates the network, simulates it and extracts the firing rates. If PathSpikes is supplied the spikes get saved
        to a pickle file. If a timeout is supplied, a timeout handler is created which stops the execution.
        Parameters:
            PathSpikes: (optional) Path of file for spiketimes
            timeout: (optional) Time of timeout in seconds
        Returns:
            Dictionary with firing rates, timing information (dict) and parameters (dict)
        """
        if timeout is not None:
            # Change the behavior of SIGALRM
            signal.signal(signal.SIGALRM, GeneralHelper.timeout_handler)
            signal.alarm(timeout)
            # This try/except loop ensures that
            #   you'll catch TimeoutException when it's sent.
        try:
            self.setup_network()
            self.build_model()
            self.load_model(GPUspecificConstraint=GPUspecificConstraint)
            spiketimes = self.simulate_and_get_recordings()
            rates = np.array(super().get_firing_rates(spiketimes))

            if PathSpikes is not None:
                with open(PathSpikes, 'wb') as outfile:
                    pickle.dump(spiketimes, outfile)
            return {'e_rate': rates[:, 0], 'i_rate': rates[:, 1], 'Timing': self.get_timing(),
                    'params': self.get_parameter(),
                    'spiketimes': spiketimes}

        except GeneralHelper.TimeoutException:
            print("Aborted - Timeout")
            return {'e_rate': np.array(-1), 'i_rate': np.array(-1), 'Timing': self.get_timing(),
                    'params': self.get_parameter(),
                    'spiketimes': [[], []]}

    def prepare_global_parameters(self):
        """
        Initializes the global parameters associated with neuron populations to 0. Global parameters have to be set
        initially before the model is loaded to the GPU.
        """
        GeNNHelper.prepareModelForLoad_Neurons(self.model)

    def reinitalizeModel(self):
        """
        Reinitalizes a loaded model -> e.g. Spike buffers are cleared, connectivity is initalized again from a new seed.
        membrane potentials are reset.
        """
        self.model.reinitialise()

    def getModelTime(self):
        """
        Gets the internal time of the GeNN model
        """
        return self.model.t

    def clearTiming(self):
        """
        Clears the Timing of the Model
        """
        self.Timing = {'Sim': [], 'Download': []}


if __name__ == "__main__":
    sys.path.append("..")
    from Defaults import defaultSimulate as default
    import matplotlib.pyplot as plt

    EI_cluster = ClusteredNetworkGeNN(default, {'n_jobs': 4, 'warmup': 500, 'simtime': 1200, 'stim_clusters': [3],
                                                'stim_amp': 2.0, 'stim_starts': [600.], 'stim_ends': [1000.],
                                                'matrixType': "PROCEDURAL_GLOBALG"
                                                })
    spikes = EI_cluster.create_and_simulate()
    print(EI_cluster.get_parameter())
    plt.figure()
    plt.plot(spikes[0][0, :], spikes[0][1, :], '.', markersize=1)
    plt.savefig('GeNN.png')

    del EI_cluster

    Cluster = ClusteredNetworkGeNN_Timing(default,
                                          {'n_jobs': 4, 'warmup': 1200, 'simtime': 1200,
                                           'stim_clusters': [3],
                                           'stim_amp': 0.5, 'stim_starts': [60.], 'stim_ends': [100.],
                                           'matrixType':  "SPARSE_GLOBALG",
                                           'I_th_E': 0.0, 'I_th_I': 0.0}, batch_size=2,
                                          NModel=GeNN_Models.define_iaf_psc_exp_Ie_multibatch())

    # Name has to be changed because PyGeNN will be confused if two objects with the same reference are present
    Cluster.set_model_build_pipeline([lambda: Cluster.setup_GeNN(Name="EICluster2"), Cluster.create_populations,
                                      Cluster.create_stimulation, Cluster.create_recording_devices, Cluster.connect,
                                      Cluster.prepare_global_parameters])

    Cluster.setup_network()
    Cluster.build_model()
    Cluster.load_model()  # set GPUspecificConstraint to a value like 5000000 to test if the splitting works
    print(Cluster.duration_timesteps)
    print(Cluster.runs)
    Cluster.Populations[0].set_global_Param('Ix', [1.23, 2.4])
    Cluster.Populations[1].set_global_Param('Ix', [0.8, 1.24])
    spiketimes = Cluster.simulate_and_get_recordings()
    rates = np.array(Cluster.get_firing_rates(spiketimes))
    print(rates)
    plt.figure()
    plt.plot(spiketimes[0][0, :], spiketimes[0][1, :], '.')
    plt.savefig('GeNN_1.png')
    plt.figure()
    plt.plot(spiketimes[1][0, :], spiketimes[1][1, :], '.')
    plt.savefig('GeNN_2.png')
    print(Cluster.Timing)
