import numpy as np
import pickle
import sys
import time
sys.path.append("..")
from Defaults import defaultSimulate as default
from Helper import ClusterModelGeNN
from Helper import GeNN_Models
import psutil
import matplotlib.pyplot as plt

if __name__ == '__main__':

    MatrixType = 0
    FactorSize = 10
    FactorTime = 5
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

    # Added to adjust number of cores used to the running machine
    CPUcount=psutil.cpu_count(logical = False)
    if CPUcount > 8:
        CPUcount -= 2

    startTime = time.time()
    baseline = {'N_E': 80, 'N_I': 20,  # number of E/I neurons -> typical 4:1
                'simtime': 600, 'warmup': 10}

    params = {'n_jobs': CPUcount, 'N_E': FactorSize * baseline['N_E'], 'N_I': FactorSize * baseline['N_I'], 'dt': 0.1,
              'neuron_type': 'iaf_psc_exp', 'simtime': FactorTime * baseline['simtime'], 'delta_I_xE': 0.,
              'delta_I_xI': 0., 'record_voltage': False, 'record_from': 1, 'warmup': FactorTime * baseline['warmup'],
              'Q': 10, #'stim_clusters': [20], 'stim_starts': [500, 1000, 1500], 'stim_ends': [750, 1250, 1750],
              'stim_amp': 1.0, 'stim_duration': 200, 'inter_stim_delay': 100
              }
    # params['stim_starts'] = [params['warmup'] + i * 200 for i in range(params['Q'])]
    # params['stim_ends'] = [s + 100 for s in params['stim_starts']]

    jip_ratio = 0.75  # 0.75 default value  #works with 0.95 and gif wo adaptation
    jep = 4.0  # clustering strength
    jip = 1. + (jep - 1) * jip_ratio
    params['jplus'] = np.array([[jep, jip], [jip, jip]])
    I_ths = [2.13,
             1.24]  # 3,5,Hz        #background stimulation of E/I neurons -> sets firing rates and changes behavior
    # to some degree # I_ths = [5.34,2.61] # 10,15,Hzh

    params['I_th_E'] = I_ths[0]
    params['I_th_I'] = I_ths[1]
    timeout = 18000  # 5h
    if MatrixType >= 1:
        params['matrixType'] = "PROCEDURAL_GLOBALG"
    else:
        params['matrixType'] = "SPARSE_GLOBALG"
    # EI_Network = ClusterModelGeNN.ClusteredNetworkGeNN_Timing(default, params, batch_size=1, NModel="LIF")
    # sequences = EI_Network.generate_input_sequences(2)
    # for sequence in sequences:
    for ii in range(2):
        EI_Network = ClusterModelGeNN.ClusteredNetworkGeNN_Timing(default, params, batch_size=1, NModel="LIF")
        sequence = EI_Network.generate_input_sequences(1)[0]
        print(f"Running simulation for sequence: {sequence}")
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
        spiketimes = EI_Network.simulate_and_get_recordings()
        EI_Network.make_synapse_matrices()
        #EI_Network.display_matrices()
        EI_Network.create_full_network_connectivity_matrix()
        EI_Network.display_full_network_connectivity_matrix()
        plt.figure()
        plt.plot(spiketimes[0][0, :], spiketimes[0][1, :], '.', ms=0.5)
        plt.title(f"Spiketimes for Sequence: {sequence}")
        plt.xlabel("Time (ms)")
        #plt.ylabel("Neuron Index")
        neurons_per_cluster = params['N_E'] // params['Q']
        cluster_labels = {i * neurons_per_cluster: f'Cluster {i}' for i in range(params['Q'])}
        ax = plt.gca()
        for i in range(params['Q']):
            y = i * neurons_per_cluster
            plt.axhline(y=y, color='gray', linestyle='--')
            ax.text(-0.10, y + neurons_per_cluster / 2, f'{i}', transform=ax.get_yaxis_transform(),
                    horizontalalignment='right', verticalalignment='center')

        for start, end, cluster in zip(params['stim_starts'], params['stim_ends'], sequence):
                plt.axvline(x=start, color='red', linestyle='--', lw=0.8)
                plt.axvline(x=end, color='green', linestyle='--', lw=0.8)
                plt.text((start + end) / 2, plt.ylim()[1] * 0.95, f'{cluster}', horizontalalignment='center', color='black')
        plt.show()
