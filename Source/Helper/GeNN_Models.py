import numpy as np
from pygenn import genn_model, genn_wrapper
# ********************************************************************************
#                      Model Definitions
# ********************************************************************************
# GIF neuron model - derived from NEST
def define_GIF():
    gif = genn_model.create_custom_neuron_class("gif",
                                            derived_params=[
                                                ("P33", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    np.exp(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("P30", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -1 / pars[0] * np.expm1(
                                                                                        -dt / (pars[0] / pars[4])) * (
                                                                                                pars[0] / pars[4]))()),
                                                ("P31", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -np.expm1(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("Psfa", genn_model.create_dpf_class(lambda pars, dt:
                                                                                     np.exp(-dt / pars[9]))()),
                                                ("Pstc", genn_model.create_dpf_class(lambda pars, dt:
                                                                                     np.exp(-dt / pars[7]))())
                                            ],
                                            param_names=["C_m", "t_ref", "V_reset", "E_L", "g_L", "I_e", "q_stc",
                                                         "tau_stc", "q_sfa", "tau_sfa", "V_T_star", "lambda_0",
                                                         "Delta_V"],
                                            var_name_types=[("V", "scalar"), ("RefracTime", "scalar"),
                                                            ("sfa", "scalar"), ("TH", "scalar"), ("stc", "scalar"),
                                                            ("lambda", "scalar"), ("u", "scalar")],
                                            sim_code="""
        $(TH) = $(V_T_star)+$(sfa);
        $(sfa) = $(sfa) * $(Psfa);

        if ($(RefracTime) <= 0.0)
        {
            $(V) = $(P30) * ($(I_e) - $(stc) + $(Isyn)) + $(P33) * $(V) + $(P31) * $(E_L);
            $(lambda) = $(lambda_0) * exp(($(V)-$(TH))/$(Delta_V));
            $(lambda) = - expm1( -1*$(lambda)*DT );
            $(u)=$(gennrand_uniform);
        }
        else
        {
            $(RefracTime) -= DT;
        }
        $(stc)=$(stc)*$(Pstc);
        """,
        reset_code="""
        $(V) = $(V_reset);
        $(RefracTime) = $(t_ref);
        $(sfa) += $(q_sfa);
        $(stc) += $(q_stc);
        $(lambda)=0;
        """,
        threshold_condition_code="$(RefracTime) <= 0.0 && $(u) <  $(lambda)"
        )
    return gif

def define_GIF_Ie_multibatch():
    gif = genn_model.create_custom_neuron_class("gif",
                                            derived_params=[
                                                ("P33", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    np.exp(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("P30", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -1 / pars[0] * np.expm1(
                                                                                        -dt / (pars[0] / pars[4])) * (
                                                                                                pars[0] / pars[4]))()),
                                                ("P31", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -np.expm1(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("Psfa", genn_model.create_dpf_class(lambda pars, dt:
                                                                                     np.exp(-dt / pars[9]))()),
                                                ("Pstc", genn_model.create_dpf_class(lambda pars, dt:
                                                                                     np.exp(-dt / pars[7]))())
                                            ],
                                            param_names=["C_m", "t_ref", "V_reset", "E_L", "g_L", "I_e", "q_stc",
                                                         "tau_stc", "q_sfa", "tau_sfa", "V_T_star", "lambda_0",
                                                         "Delta_V"],
                                            var_name_types=[("V", "scalar"), ("RefracTime", "scalar"),
                                                            ("sfa", "scalar"), ("TH", "scalar"), ("stc", "scalar"),
                                                            ("lambda", "scalar"), ("u", "scalar")],
                                            extra_global_params=[("Ix", "float*")], 

                                            sim_code="""
        $(TH) = $(V_T_star)+$(sfa);
        $(sfa) = $(sfa) * $(Psfa);

        if ($(RefracTime) <= 0.0)
        {
            $(V) = $(P30) * ($(I_e) - $(stc) + $(Isyn)+ $(Ix)[$(batch)]) + $(P33) * $(V) + $(P31) * $(E_L);
            $(lambda) = $(lambda_0) * exp(($(V)-$(TH))/$(Delta_V));
            $(lambda) = - expm1( -1*$(lambda)*DT );
            $(u)=$(gennrand_uniform);
        }
        else
        {
            $(RefracTime) -= DT;
        }
        $(stc)=$(stc)*$(Pstc);
        """,
        reset_code="""
        $(V) = $(V_reset);
        $(RefracTime) = $(t_ref);
        $(sfa) += $(q_sfa);
        $(stc) += $(q_stc);
        $(lambda)=0;
        """,

        threshold_condition_code="$(RefracTime) <= 0.0 && $(u) <  $(lambda)"
        )
    return gif  

def define_GIF_Ie_singleBatch():
    gif = genn_model.create_custom_neuron_class("gif",
                                            derived_params=[
                                                ("P33", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    np.exp(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("P30", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -1 / pars[0] * np.expm1(
                                                                                        -dt / (pars[0] / pars[4])) * (
                                                                                                pars[0] / pars[4]))()),
                                                ("P31", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -np.expm1(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("Psfa", genn_model.create_dpf_class(lambda pars, dt:
                                                                                     np.exp(-dt / pars[9]))()),
                                                ("Pstc", genn_model.create_dpf_class(lambda pars, dt:
                                                                                     np.exp(-dt / pars[7]))())
                                            ],
                                            param_names=["C_m", "t_ref", "V_reset", "E_L", "g_L", "I_e", "q_stc",
                                                         "tau_stc", "q_sfa", "tau_sfa", "V_T_star", "lambda_0",
                                                         "Delta_V"],
                                            var_name_types=[("V", "scalar"), ("RefracTime", "scalar"),
                                                            ("sfa", "scalar"), ("TH", "scalar"), ("stc", "scalar"),
                                                            ("lambda", "scalar"), ("u", "scalar")],
                                            extra_global_params=[("Ix", "float")], 

                                            sim_code="""
        $(TH) = $(V_T_star)+$(sfa);
        $(sfa) = $(sfa) * $(Psfa);

        if ($(RefracTime) <= 0.0)
        {
            $(V) = $(P30) * ($(I_e) - $(stc) + $(Isyn)+ $(Ix)) + $(P33) * $(V) + $(P31) * $(E_L);
            $(lambda) = $(lambda_0) * exp(($(V)-$(TH))/$(Delta_V));
            $(lambda) = - expm1( -1*$(lambda)*DT );
            $(u)=$(gennrand_uniform);
        }
        else
        {
            $(RefracTime) -= DT;
        }
        $(stc)=$(stc)*$(Pstc);
        """,
        reset_code="""
        $(V) = $(V_reset);
        $(RefracTime) = $(t_ref);
        $(sfa) += $(q_sfa);
        $(stc) += $(q_stc);
        $(lambda)=0;
        """,

        threshold_condition_code="$(RefracTime) <= 0.0 && $(u) <  $(lambda)"
        )
    return gif   


def define_GIF_Ie(batchsize=1):
    """
    Defines a full GIF model with a DC current into the membrane as global parameter. Depending on the batchsize either a
    single number is used (batch_size=1) or a pointer is used. If a single number is chosen, GeNN will transmit it to
    the GPU at every simulation step which slows the simulation down. In most cases it would be better to choose
    define_GIF_Ie_multibatch also for a batch_size=1 and use a list with only one element.
    """
    if batchsize>1:
        return define_GIF_Ie_multibatch()
    else:
        return define_GIF_Ie_singleBatch()

def define_ReducedGIF():
    gif = genn_model.create_custom_neuron_class("gif",
                                            derived_params=[
                                                ("P33", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    np.exp(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("P30", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -1 / pars[0] * np.expm1(
                                                                                        -dt / (pars[0] / pars[4])) * (
                                                                                                pars[0] / pars[4]))()),
                                                ("P31", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -np.expm1(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("Pstc", genn_model.create_dpf_class(lambda pars, dt:
                                                                                     np.exp(-dt / pars[7]))())
                                            ],
                                            param_names=["C_m", "t_ref", "V_reset", "E_L", "g_L", "I_e", "q_stc",
                                                         "tau_stc", "q_sfa", "tau_sfa", "V_T_star", "lambda_0",
                                                         "Delta_V"],
                                            var_name_types=[("V", "scalar"), ("RefracTime", "scalar"),
                                                            ("sfa", "scalar"), ("TH", "scalar"), ("stc", "scalar"),
                                                            ("lambda", "scalar"), ("u", "scalar")],

                                            sim_code="""

        if ($(RefracTime) <= 0.0)
        {         
            $(V) = $(P30) * ($(I_e) - $(stc) + $(Isyn)) + $(P33) * $(V) + $(P31) * $(E_L);
        }
        else
        {
            $(RefracTime) -= DT;
        }
        $(stc)=$(stc)*$(Pstc);
        
        """,
        reset_code="""
        $(V) = $(V_reset);
        $(RefracTime) = $(t_ref);
        $(stc) += $(q_stc);
        """,
        threshold_condition_code="$(RefracTime) <= 0.0 && $(V_T_star) <  $(V)"
        )
    return gif



def define_ReducedGIF_Ie_multibatch():
    gif = genn_model.create_custom_neuron_class("gif",
                                            derived_params=[
                                                ("P33", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    np.exp(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("P30", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -1 / pars[0] * np.expm1(
                                                                                        -dt / (pars[0] / pars[4])) * (
                                                                                                pars[0] / pars[4]))()),
                                                ("P31", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -np.expm1(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("Pstc", genn_model.create_dpf_class(lambda pars, dt:
                                                                                     np.exp(-dt / pars[7]))())
                                            ],
                                            param_names=["C_m", "t_ref", "V_reset", "E_L", "g_L", "I_e", "q_stc",
                                                         "tau_stc", "q_sfa", "tau_sfa", "V_T_star", "lambda_0",
                                                         "Delta_V"],
                                            var_name_types=[("V", "scalar"), ("RefracTime", "scalar"),
                                                            ("sfa", "scalar"), ("TH", "scalar"), ("stc", "scalar"),
                                                            ("lambda", "scalar"), ("u", "scalar")],
                                            extra_global_params=[("Ix", "float*")], 

                                            sim_code="""

        if ($(RefracTime) <= 0.0)
        {         
            $(V) = $(P30) * ($(I_e) - $(stc) + $(Isyn)+ $(Ix)[$(batch)]) + $(P33) * $(V) + $(P31) * $(E_L);
        }
        else
        {
            $(RefracTime) -= DT;
        }
        $(stc)=$(stc)*$(Pstc);
        
        """,
        reset_code="""
        $(V) = $(V_reset);
        $(RefracTime) = $(t_ref);
        $(stc) += $(q_stc);
        """,
        threshold_condition_code="$(RefracTime) <= 0.0 && $(V_T_star) <  $(V)"
        )
    return gif


def define_ReducedGIF_Ie_singlebatch():
    gif = genn_model.create_custom_neuron_class("gif",
                                            derived_params=[
                                                ("P33", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    np.exp(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("P30", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -1 / pars[0] * np.expm1(
                                                                                        -dt / (pars[0] / pars[4])) * (
                                                                                                pars[0] / pars[4]))()),
                                                ("P31", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -np.expm1(
                                                                                        -dt / (pars[0] / pars[4])))()),
                                                ("Pstc", genn_model.create_dpf_class(lambda pars, dt:
                                                                                     np.exp(-dt / pars[7]))())
                                            ],
                                            param_names=["C_m", "t_ref", "V_reset", "E_L", "g_L", "I_e", "q_stc",
                                                         "tau_stc", "q_sfa", "tau_sfa", "V_T_star", "lambda_0",
                                                         "Delta_V"],
                                            var_name_types=[("V", "scalar"), ("RefracTime", "scalar"),
                                                            ("sfa", "scalar"), ("TH", "scalar"), ("stc", "scalar"),
                                                            ("lambda", "scalar"), ("u", "scalar")],
                                            extra_global_params=[("Ix", "float")], 

                                            sim_code="""

        if ($(RefracTime) <= 0.0)
        {         
            $(V) = $(P30) * ($(I_e) - $(stc) + $(Isyn)+ $(Ix)) + $(P33) * $(V) + $(P31) * $(E_L);
        }
        else
        {
            $(RefracTime) -= DT;
        }
        $(stc)=$(stc)*$(Pstc);
        
        """,
        reset_code="""
        $(V) = $(V_reset);
        $(RefracTime) = $(t_ref);
        $(stc) += $(q_stc);
        """,
        threshold_condition_code="$(RefracTime) <= 0.0 && $(V_T_star) <  $(V)"
        )
    return gif


def define_ReducedGIF_Ie(batchsize=1):
    if batchsize>1:
        return define_ReducedGIF_Ie_multibatch()
    else:
        return define_ReducedGIF_Ie_singlebatch()



def define_iaf_psc_exp_Ie_singleBatch():
    iaf_psc_exp = genn_model.create_custom_neuron_class("iaf_psc_exp_Tune",
                                            derived_params=[
                                                ("P33", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    np.exp(
                                                                                        -dt / (pars[4])))()),
                                                ("P30", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -1 / pars[0] * np.expm1(
                                                                                        -dt / (pars[4])) * (
                                                                                                pars[4]))()),
                                                ("P31", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -np.expm1(
                                                                                        -dt / (pars[4])))())
                                            ],
                                            param_names=["C", "TauRefrac", "Vreset", "Vrest", "TauM", "Ioffset","Vthresh"],
                                            var_name_types=[("V", "scalar"), ("RefracTime", "scalar")],
                                            extra_global_params=[("Ix", "float")], 

                                            sim_code="""

        if ($(RefracTime) <= 0.0)
        { 
            $(V) = $(P30) * ($(Ioffset) + $(Isyn)+ $(Ix)) + $(P33) * $(V) + $(P31) * $(Vrest);
        }
        else
        {
            $(RefracTime) -= DT;
        }

        """,
        reset_code="""
        $(V) = $(Vreset);
        $(RefracTime) = $(TauRefrac);
        """,
        threshold_condition_code="$(RefracTime) <= 0.0 && $(Vthresh) <  $(V)"
        )
    return iaf_psc_exp



def define_iaf_psc_exp_Ie_multibatch():
    iaf_psc_exp = genn_model.create_custom_neuron_class("iaf_psc_exp_Tune",
                                            derived_params=[
                                                ("P33", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    np.exp(
                                                                                        -dt / (pars[4])))()),
                                                ("P30", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -1 / pars[0] * np.expm1(
                                                                                        -dt / (pars[4])) * (
                                                                                                pars[4]))()),
                                                ("P31", genn_model.create_dpf_class(lambda pars, dt:
                                                                                    -np.expm1(
                                                                                        -dt / (pars[4])))())
                                            ],
                                            param_names=["C", "TauRefrac", "Vreset", "Vrest", "TauM", "Ioffset","Vthresh"],
                                            var_name_types=[("V", "scalar"), ("RefracTime", "scalar")],
                                            extra_global_params=[("Ix", "float*")], 

                                            sim_code="""

        if ($(RefracTime) <= 0.0)
        { 
            $(V) = $(P30) * ($(Ioffset) + $(Isyn)+ $(Ix)[$(batch)]) + $(P33) * $(V) + $(P31) * $(Vrest);
        }
        else
        {
            $(RefracTime) -= DT;
        }

        """,
        reset_code="""
        $(V) = $(Vreset);
        $(RefracTime) = $(TauRefrac);
        """,
        threshold_condition_code="$(RefracTime) <= 0.0 && $(Vthresh) <  $(V)"
        )
    return iaf_psc_exp


def define_iaf_psc_exp_Ie(batchsize=1):
    """
    Defines a iaf model with exponential postsynaptic current with a DC current into the membrane as global parameter.
    Depending on the batchsize either a single number is used (batch_size=1) or a pointer is used.
    If a single number is chosen, GeNN will transmit it to the GPU at every simulation step which slows the simulation
    down. In most cases it would be better to choose  define_iaf_psc_exp_Ie_multibatch also for a batch_size=1 and use a
    list with only one element.
    """
    if batchsize>1:
        return define_iaf_psc_exp_Ie_multibatch()
    else:
        return define_iaf_psc_exp_Ie_singleBatch()



# def define_ClusterStim():
#     """
#     Defines a model of current source which will emit a current pulse between t_onset and t_offset at the given strength
#     and otherwise does not inject any current.
#     """
#     cluster_stimulus = genn_model.create_custom_current_source_class(
#         "cluster_stimulus",
#         param_names=['strength'],
#         extra_global_params=[('t_onset', 'float'), ('t_offset', 'float')],
#         injection_code=
#         """
#         if ((t>=$(t_onset))&&(t<$(t_offset))){
#             $(injectCurrent, $(strength));;
#         }
#         else {
#             $(injectCurrent, 0);
#         }
#         """)
#     return cluster_stimulus
def define_ClusterStim():
    """
    Defines a model of current source which will emit a current pulse between t_onset and t_offset at the given strength
    and otherwise does not inject any current.
    """
    cluster_stimulus = genn_model.create_custom_current_source_class(
        "cluster_stimulus",
        extra_global_params=[('t_onset', 'float'), ('t_offset', 'float'), ('strength', 'float')],
        injection_code=
        """
        if ((t>=$(t_onset))&&(t<$(t_offset))){
            $(injectCurrent, $(strength));;
        }
        else {
            $(injectCurrent, 0);
        }
        """)
    return cluster_stimulus



def define_Poisson_model():
    """
    Defines a model of Poisson generator with a given rate. It has a state variable timeStepToSpike which has to be set.
    """
    poisson_model = genn_model.create_custom_neuron_class(
        'poisson_model',
        var_name_types={('rate', 'scalar'), ('timeStepToSpike', 'scalar')},
        sim_code="""
        const scalar isi = 1000.0 / $(rate);
        if ($(timeStepToSpike) <= 0.0f) {
            $(timeStepToSpike) += isi * $(gennrand_exponential);
        }
        $(timeStepToSpike) -= 1.0;
        """,
        threshold_condition_code="$(timeStepToSpike) <= 0.0"
    )
    return poisson_model

# STDP rule
# def define_symmetric_stdp():
#     symmetric_stdp = genn_model.create_custom_weight_update_class(
#         "symmetric_stdp",
#         param_names=["tau", "rho", "eta", "wMin", "wMax"],
#         var_name_types=[("g", "scalar")],
#         sim_code=
#         """
#         $(addToInSyn, $(g));
#         const scalar dt = $(t) - $(sT_post);
#         const scalar timing = exp(-dt / $(tau)) - $(rho);
#         const scalar newWeight = 0.95*$(g) + ($(eta) * timing);
#         $(g) = fmin($(wMax), fmax($(wMin), newWeight));
#         """,
#         learn_post_code=
#         """
#         const scalar dt = $(t) - $(sT_pre);
#         const scalar timing = fmax(exp(-dt / $(tau)) - $(rho), -0.1*$(rho));
#         const scalar newWeight = 0.95*$(g) + ($(eta) * timing);
#         $(g) = fmin($(wMax), fmax($(wMin), newWeight));
#         """,
#         is_pre_spike_time_required=True,
#         is_post_spike_time_required=True
#     )
#     return symmetric_stdp

# def define_symmetric_stdp():
#     wu = genn_model.create_custom_weight_update_class(
#         "symmetric_stdp",
#         param_names=["tau", "rho", "eta", "tau_h", "z_star", "lambda_h", "lamda_n", "wMin", "wMax"],
#         var_name_types=[("g", "scalar")],
#         post_var_name_types=[("z", "scalar")],
#         derived_params=[
#             ("Pz", genn_model.create_dpf_class(lambda pars, dt:
#                                                np.exp(-dt / pars[0]))()),
#         ],
#         sim_code="""
#         $(addToInSyn, $(g));
#         const scalar dt = $(t) - $(sT_post);
#         const scalar timing = exp(-dt / $(tau)) - $(rho);
#         const scalar newWeight = $(g) + ($(eta) * timing);
#         $(g) = fmin($(wMax), fmax($(wMin), newWeight));
#         """,
#         learn_post_code="""
#         const scalar dt = $(t) - $(sT_pre);
#         const scalar timing = fmax(exp(-dt / $(tau)) - $(rho), -0.1*$(rho));
#         const scalar newWeight = $(g) - ($(eta) * timing);
#         $(g) = fmin($(wMax), fmax($(wMin), newWeight));
#         """,
#         post_spike_code="""
#         $(z) += 1000 / $(tau_h);
#         """,
#         post_dynamics_code="""
#         $(z) *= $(Pz);
#         """,
#         is_pre_spike_time_required=True,
#         is_post_spike_time_required=True)
#     return wu

# def define_symmetric_stdp():
#     wu = genn_model.create_custom_weight_update_class(
#         "symmetric_stdp",
#         param_names=["tau", "rho", "eta", "lambda_p", "lambda_n", "tau_h", "z_star", "lambda_h", "wMin", "wMax"],
#         var_name_types=[("g", "scalar")],
#         post_var_name_types=[("z", "scalar")],
#         derived_params=[
#             ("Pz", genn_model.create_dpf_class(lambda pars, dt:
#                                                np.exp(-dt / pars[0]))()),
#         ],
#         sim_code="""
#         $(addToInSyn, $(g));
#         const scalar dt = $(t) - $(sT_post);
#         const scalar timing = exp(-dt / $(tau)) - $(rho);
#         const scalar newWeight = $(g) + ($(lambda_p) * $(eta) * timing);
#         $(g) = fmin($(wMax), fmax($(wMin), newWeight));
#         """,
#         learn_post_code="""
#         const scalar dt = $(t) - $(sT_pre);
#         const scalar timing = exp(-dt / $(tau)) - $(rho);
#         const scalar newWeight = $(g) + ($(lambda_p) * $(eta) * timing);
#         $(g) = fmin($(wMax), fmax($(wMin), newWeight));
#         """,
#         post_spike_code="""
#         $(z) += 1000 / $(tau_h);
#         """,
#         post_dynamics_code="""
#         $(z) *= $(Pz);
#         """,
#         is_pre_spike_time_required=True,
#         is_post_spike_time_required=True)
#     return wu


def define_symmetric_stdp():
    wu = genn_model.create_custom_weight_update_class(
        "symmetric_stdp",
        param_names=["tau", "rho", "eta", "lambda_n", "tau_h", "z_star", "lambda_h", "wMin", "wMax"],
        var_name_types=[("g", "scalar")],
        post_var_name_types=[("z", "scalar")],
        derived_params=[
            ("Pz", genn_model.create_dpf_class(lambda pars, dt:
                                               np.exp(-dt / pars[0]))()),
        ],
        sim_code="""
        $(addToInSyn, $(g));
        const scalar dt = $(t) - $(sT_post);
        const scalar timing = exp(-dt / $(tau)) - $(rho);
        const scalar newWeight = $(g) + $(eta) * timing;
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
        """,
        learn_post_code="""
        const scalar dt = $(t) - $(sT_pre);
        const scalar timing = exp(-dt / $(tau)) - $(rho);
        const scalar newWeight = $(g) - ($(eta) * timing) + ($(lambda_h) * ($(z_star)-$(z))) - $(lambda_n);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
        """,
        post_spike_code="""
        $(z) += 1000 / $(tau_h);
        """,
        post_dynamics_code="""
        $(z) *= $(Pz);
        """,
        is_pre_spike_time_required=True,
        is_post_spike_time_required=True)
    return wu




