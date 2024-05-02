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
    FactorSize = 20
    FactorTime = 2
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
                'simtime': 900, 'warmup': 100}

    params = {'n_jobs': CPUcount, 'N_E': FactorSize * baseline['N_E'], 'N_I': FactorSize * baseline['N_I'], 'dt': 0.1,
              'neuron_type': 'iaf_psc_exp', 'simtime': FactorTime * baseline['simtime'], 'delta_I_xE': 0.,
              'delta_I_xI': 0., 'record_voltage': False, 'record_from': 1, 'warmup': FactorTime * baseline['warmup'],
              'Q': 20, 'stim_clusters': [3], 'stim_starts': [500, 1000, 1500], 'stim_ends': [750, 1250, 1750], 'stim_amp': 1.0
              }
    params['stim_starts'] = [params['warmup'] + i * 100 for i in range(params['Q'])]
    params['stim_ends'] = [s + 50 for s in params['stim_starts']]

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

    EI_Network = ClusterModelGeNN.ClusteredNetworkGeNN_Timing(default, params, batch_size=1, NModel="LIF")
    # Aribitary sequences(how many sequneces)
    sequences = EI_Network.generate_input_sequences(1)
    for seq in sequences:
        print(list(seq))
    EI_Network.set_model_build_pipeline([EI_Network.setup_GeNN, EI_Network.create_populations,
                                           EI_Network.create_stimulation,
                                           EI_Network.create_recording_devices,
                                           EI_Network.connect,
                                           EI_Network.create_learning_synapses,
                                           EI_Network.assign_elements_to_clusters
                                       ])
    # Creates object which creates the EI clustered network in NEST
    Result = EI_Network.get_simulation(timeout=timeout)
    stopTime = time.time()
    Result['Timing']['Total'] = stopTime - startTime
    print("Total time     : %.4f s" % Result['Timing']['Total'])
    print("Cluster elements:", EI_Network.cluster_elements)
    plt.figure()
    plt.plot(Result['spiketimes'][0][0, :], Result['spiketimes'][0][1, :], '.', ms=0.5)
    plt.show()
