import numpy as np
import pickle
import sys
import time

sys.path.append("..")
from Defaults import defaultSimulate as default
from Helper import ClusterModelGeNN
from ClusterNetworkGeNN_MC import ClusterNetworkGeNN_MC
from Helper import GeNN_Models
import psutil
import matplotlib.pyplot as plt

if __name__ == '__main__':

    MatrixType = 0
    FactorSize = 10
    FactorTime = 4
    Savepath = "Data.pkl"

    if len(sys.argv) == 2:
        FactorSize = int(sys.argv[1])
        FactorTime = int(sys.argv[1])

    elif len(sys.argv) == 3:
        FactorSize = int(sys.argv[1])
        FactorTime = int(sys.argv[2])
    elif len(sys.argv) == 4:
        FactorSize = int(sys.argv[1])
        FactorTime = int(sys.argv[2])
        MatrixType = int(sys.argv[3])
    elif len(sys.argv) == 5:
        FactorSize = int(sys.argv[1])
        FactorTime = int(sys.argv[2])
        MatrixType = int(sys.argv[3])
        Savepath = sys.argv[4]

    elif len(sys.argv) >= 6:
        FactorSize = int(sys.argv[1])
        FactorTime = int(sys.argv[2])
        MatrixType = int(sys.argv[3])
        Savepath = sys.argv[4]
        print("Too many arguments")

    print("FactorSize: " + str(FactorSize) + " FactorTime: " + str(FactorTime))

    CPUcount = psutil.cpu_count(logical=False)
    if CPUcount > 8:
        CPUcount -= 2

    startTime = time.time()
    baseline = {'N_E': 80, 'N_I': 20,  # number of E/I neurons -> typical 4:1
                'simtime': 600, 'warmup': 0}

    params = {'n_jobs': CPUcount, 'N_E': FactorSize * baseline['N_E'], 'N_I': FactorSize * baseline['N_I'], 'dt': 0.1,
              'neuron_type': 'iaf_psc_exp', 'simtime': 900, 'delta_I_xE': 0.,
              'delta_I_xI': 0., 'record_voltage': False, 'record_from': 1, 'warmup': 0.,
              'Q': 10, 'stim_amp':1.5, 'stim_duration': 160, 'inter_stim_delay': -50.0, 'no_stim' : 0
              }
    params['simtime'] = 360  # 2 * FactorTime * baseline['simtime']

    jip_ratio = 0.7 #0.95  # 0.7 default value  #works with 0.95 and gif wo adaptation
    jep = 3.8  # 2.8  #7 # clustering strength
    jip = 1. + (jep - 1) * jip_ratio
    params['jplus'] = np.array([[jep, jip], [jip, jip]])
    I_ths = [3.3, 2.6]  # 3,5,Hz        #background stimulation of E/I neurons -> sets firing rates and changes behavior
    # to some degree # I_ths = [5.34,2.61] 2.13,
    #              1.24# 10,15,Hzh

    params['I_th_E'] = I_ths[0]
    params['I_th_I'] = I_ths[1]
    timeout = 18000  # 5h
    if MatrixType >= 1:
        params['matrixType'] = "PROCEDURAL_GLOBALG"
    else:
        params['matrixType'] = "SPARSE_GLOBALG"

    for ii in range(1):
        EI_Network = ClusterNetworkGeNN_MC(default, params, batch_size=1, NModel="LIF")
        num_clusters = 3
        transition_matrix = np.random.dirichlet(np.ones(num_clusters), size=num_clusters)
        initial_state = 0
        EI_Network.create_MC(transition_matrix, initial_state)
        print(f"Initial State: {EI_Network.state}")
        print(f"Transition Matrix:\n{EI_Network.transition_matrix}")

        steps = 2
        states = EI_Network.simulate_MC(steps)
        print(f"States after {steps} steps: {states}")
        sequence = states

        stim_starts = [params['warmup'] + i * (params['stim_duration'] + params['inter_stim_delay']) for i in range(len(sequence))]
        stim_ends = [start + params['stim_duration'] for start in stim_starts]
        params['stim_starts'] = stim_starts
        params['stim_ends'] = stim_ends

        EI_Network.set_model_build_pipeline([
            lambda: EI_Network.setup_GeNN(Name="EICluster" + str(sequence)),
            EI_Network.create_populations,
            lambda: EI_Network.create_stimulation(sequence),
            EI_Network.create_recording_devices,
            EI_Network.connect,
            EI_Network.create_learning_synapses,
            EI_Network.prepare_global_parameters,
        ])

        EI_Network.setup_network()
        EI_Network.build_model()
        EI_Network.load_model()


        # Training
        num_epochs_train = 3
        for epoch in range(num_epochs_train):
            for ii, pop in enumerate(EI_Network.current_source):
                pop.extra_global_params['t_onset'].view[:] = stim_starts[ii] + EI_Network.model.t
                pop.extra_global_params['t_offset'].view[:] = stim_ends[ii] + EI_Network.model.t
                pop.extra_global_params['strength'].view[:] = params['stim_amp']
            print(f"Running simulation for epoch {epoch + 1} (Training)")
            spikes = EI_Network.simulate_and_get_recordings(timeZero=EI_Network.model.t)

            if epoch == 0:
                first_epoch_spikes_train = spikes
            if epoch == num_epochs_train - 2:
                last_epoch_spikes_train = spikes
            if epoch == num_epochs_train - 1:
                last_epoch_spikes_train_without = spikes

            g_trace = []
            z_trace = []

            for synapse in EI_Network.synapses:
                synapse.pull_var_from_device("g")
                synapse.pull_var_from_device("z")
                g_trace.append(synapse.vars["g"].view[0])
                z_trace.append(synapse.vars["z"].view[0])

            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(g_trace, 'o')
            ax1.set_title("Synaptic Weights (g) for One Synapse Over Time")
            ax1.set_xlabel("Time (ms)")
            ax1.set_ylabel("Weight (g)")
            plt.show()

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot( z_trace, 'o')
            ax2.set_title("Homeostatic Variables (z) for One Synapse Over Time")
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Homeostatic Variable (z)")
            plt.show()

        # Testing
        first_element_sequence = [sequence[0]]
        stim_starts_test = [params['warmup']]
        stim_ends_test = [params['warmup'] + params['stim_duration']]
        params['stim_starts'] = stim_starts_test
        params['stim_ends'] = stim_ends_test
        params['stim_amp'] = 0
        last_epoch_spikes_test = None

        num_epochs_test = 1

        for epoch in range(num_epochs_test):
            for ii, pop in enumerate(EI_Network.current_source):
                if ii == first_element_sequence[0]:
                    pop.extra_global_params['t_onset'].view[:] = stim_starts_test[0] + EI_Network.model.t
                    pop.extra_global_params['t_offset'].view[:] = stim_ends_test[0] + EI_Network.model.t
                    pop.extra_global_params['strength'].view[:] = params['stim_amp']
            print(f"Running simulation for epoch {epoch + 1} (Testing with stimulating only the first element)")
            spikes = EI_Network.simulate_and_get_recordings(timeZero=EI_Network.model.t)

            if epoch == num_epochs_test - 1:
                last_epoch_spikes_test = spikes

        EI_Network.make_synapse_matrices()
        EI_Network.create_full_network_connectivity_matrix()
        EI_Network.display_full_network_connectivity_matrix()
        EI_Network.display_full_normalized_network_connectivity_matrix()
        EI_Network.plot_markov_chain(transition_matrix)

        stim_starts_train = [params['warmup'] + i * (params['stim_duration'] + params['inter_stim_delay']) for i in
                             range(len(sequence))]
        stim_ends_train = [start + params['stim_duration'] for start in stim_starts_train]

        EI_Network.plot_spikes(first_epoch_spikes_train, "Spiketimes for Sequence: (First Epoch of Training)", stim_starts, stim_ends, sequence)
        EI_Network.plot_spikes(last_epoch_spikes_train, "Spiketimes for Sequence: (Last Epoch of Training)", stim_starts, stim_ends, sequence)
        EI_Network.plot_spikes(last_epoch_spikes_test, "Spiketimes for First Element of Sequence: (Last Epoch of Testing)", stim_starts, stim_ends, sequence)
