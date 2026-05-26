## Util functions 

import time

import jax
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import tabulate
from sbi import utils
from sbi.inference import SNPE
from sbi.utils import RestrictionEstimator , RestrictedPrior , get_density_thresholder
from jaxspec.data import ObsConfiguration
from jaxspec.data.util import fakeit_for_multiple_parameters
from jaxspec.model.abc import SpectralModel



class sbi_run():

    def __init__(self, yml_file):
        print('initialising')
        
        # open config yaml file
        self.yml_file = yml_file
        with open(yml_file , 'r') as config_file :
            self.config = yaml.safe_load(config_file)

        # Create a list of tuples containing variable name and value pairs
        table_data = [(key , value) for key , value in  self.config.items( )]

        # Create a frame around the table and print it
        print(tabulate(table_data , headers = ["Variable" , "Value"] , tablefmt = "fancy_grid"))

        self.path_pha =  self.config['path_pha']
        self.reference_pha =  self.config['reference_pha']
        self.energy_min =  self.config['energy_range'][0]
        self.energy_max =  self.config['energy_range'][1]
        self.jaxspec_model_expression =  self.config['jaxspec_model_expression']
        self.parameter_lower_bounds =  self.config['parameter_lower_bounds']
        self.parameter_upper_bounds =  self.config['parameter_upper_bounds']
        self.parameter_prior_types =  self.config['parameter_prior_types']
        self.parameter_states =  self.config['parameter_states']
        self.parameter_names_for_plots =  self.config['parameter_names_for_plots']
        self.restricted_prior_type = self.config['restricted_prior_type']
        self.number_of_simulations_for_restricted_prior =  self.config['number_of_simulations_for_restricted_prior']
        if self.restricted_prior_type=='cmin_cmax_restricted_prior' :
            self.number_of_rounds_for_restricted_prior = self.config['number_of_rounds_for_restricted_prior']
            self.c_min_for_restricted_prior =  self.config['c_min_for_restricted_prior']
            self.c_max_for_restricted_prior =  self.config['c_max_for_restricted_prior']
            self.fraction_of_valid_simulations_to_stop_restricted_prior =  self.config[
                'fraction_of_valid_simulations_to_stop_restricted_prior']
        elif self.restricted_prior_type=='cstat_restricted_prior' :
            self.number_of_rounds_for_restricted_prior = self.config['number_of_rounds_for_restricted_prior']
            self.good_fraction_for_cstat_restricted_prior=self.config['good_fraction_for_cstat_restricted_prior']
        self.restricted_prior = None

        self.type_of_inference=self.config["type_of_inference"]
        if self.type_of_inference=='single round inference' :
            self.number_of_simulations_for_train_set=self.config["number_of_simulations_for_train_set"]
            self.number_of_simulations_for_test_set =  self.config['number_of_simulations_for_test_set']
        elif self.type_of_inference=='multiple round inference' :
            self.number_of_simulations_for_train_set =  self.config['number_of_simulations_for_train_set']
            self.number_of_rounds_for_multiple_inference=self.config["number_of_rounds_for_multiple_inference"]
        else :
            print_message("Invalid type_of_inference: can either be single round inference or multiple round inference ")

        self.number_of_posterior_samples =  self.config['number_of_posterior_samples']

        self.path_outputs =  self.config["path_outputs"]
        if not os.path.exists(self.path_outputs) :
            os.makedirs(self.path_outputs)
            print(f"Directory '{self.path_outputs}' created.")
        else :
            print(f"Directory '{self.path_outputs}' already exists.")

        self.root_output_files = os.path.basename(self.yml_file).replace(".yml" , "_")
        self.x_obs_reference=None
        self.use_summary=self.config.get("use_summary_statistics", False)


    def read_data_and_init_global_prior( self ):
        print("Read the PHA and initialize the global prior")
        self.pha_filename = self.path_pha + self.reference_pha
        print(self.pha_filename , self.energy_min , self.energy_max)

        # Translate to lower case for all parameters

        parameter_prior_types = list(map(str.lower , self.parameter_prior_types))
        parameter_states = list(map(str.lower , self.parameter_states))

        # Apply log10 transformation conditionally
        parameter_lower_bounds_transformed = np.where(
            np.array(parameter_prior_types) == "loguniform" ,
            np.log10(self.parameter_lower_bounds) ,
            self.parameter_lower_bounds
        )

        parameter_upper_bounds_transformed = np.where(
            np.array(parameter_prior_types) == "loguniform" ,
            np.log10(self.parameter_upper_bounds) ,
            self.parameter_upper_bounds
        )

        # Filter free parameters
        free_indices = [i for i , state in enumerate(parameter_states) if state == "free"]
        self.free_parameter_lower_bounds_transformed = parameter_lower_bounds_transformed[free_indices]
        self.free_parameter_upper_bounds_transformed = parameter_upper_bounds_transformed[free_indices]
        self.free_parameter_prior_types = [parameter_prior_types[i] for i in free_indices]
        self.free_parameter_names_for_plots = [self.parameter_names_for_plots[i] for i in free_indices]
        # Modify all elements in the second array based on the condition
        self.free_parameter_names_for_plots_transformed = [f"Log({n})" if pt.lower( ) == "loguniform" else n
                                                        for pt , n in
                                                        zip(parameter_prior_types , self.free_parameter_names_for_plots)]

        # Convert arrays to strings
        self.free_parameter_lower_bounds_transformed_str = ', '.join(map(str , self.free_parameter_lower_bounds_transformed))
        self.free_parameter_upper_bounds_transformed_str = ', '.join(map(str , self.free_parameter_upper_bounds_transformed))
        self.free_parameter_prior_types_str = ', '.join(map(str , self.free_parameter_prior_types))
        self.free_parameter_names_for_plots_str = ', '.join(map(str , self.free_parameter_names_for_plots))
        self.free_parameter_names_for_plots_transformed_str = ', '.join(
            map(str , self.free_parameter_names_for_plots_transformed))

        # Create a new table for the additional variables
        table_data_free_parameters = [
            ("Free Parameter Lower Bounds (Transformed)" , self.free_parameter_lower_bounds_transformed_str) ,
            ("Free Parameter Upper Bounds (Transformed)" , self.free_parameter_upper_bounds_transformed_str) ,
            ("Free Parameter Prior Types" , self.free_parameter_prior_types_str) ,
            ("Free Parameter Names for Plots" , self.free_parameter_names_for_plots_str) ,
            ("Free Parameter Names for Plots (Transformed)" , self.free_parameter_names_for_plots_transformed_str) ,
        ]

        # Create a frame around the new table and print it
        print(tabulate(table_data_free_parameters , headers = ["Variable" , "Value"] , tablefmt = "fancy_grid"))

        # Read the observed spectrum
        obs = ObsConfiguration.from_pha_file(self.pha_filename , low_energy = self.energy_min , high_energy = self.energy_max)
        self.e_min_folded = obs.e_min_folded
        self.e_max_folded = obs.e_max_folded

        num_bins = len(obs.folded_counts)
        total_counts = np.sum(obs.folded_counts)
        print(f"Number of bins {num_bins} - Exposure time {obs.exposure:.1f}s - Number of counts {total_counts:.1f}")
        self.x_obs_reference = np.array(obs.folded_counts)

        if self.use_summary:

            counts = np.array(obs.folded_counts)
            energy_ref = np.hstack([obs.out_energies[0], obs.out_energies[1][-1]])
            energy_grid = np.linspace(energy_ref.min(), energy_ref.max(), 10)

            def summary_func(x):
                return summary_statistics_func(x, energy_grid=energy_grid, energy_ref=energy_ref)

            self.summary_func = summary_func
            self.x_obs_summary = self.summary_func(counts)[0].squeeze()


        self.x_obs_reference_exposure_time = obs.exposure

        low_v = torch.as_tensor(self.free_parameter_lower_bounds_transformed)
        high_v = torch.as_tensor(self.free_parameter_upper_bounds_transformed)

        self.prior = utils.BoxUniform(low = low_v , high = high_v)

    #===============================================================================================================

    def compute_x_sim( jaxspec_model_expression , parameter_states , thetas , pha_file , energy_min , energy_max ,
                    free_parameter_prior_types , parameter_lower_bounds , apply_stat = True , verbose = False ) :
        """
        # compute_x_sim: compute the simulated spectra with jaxspec fakeit like command.
        # It is therefore dependent on jaxspec, which currently has a limited number of models implemented.
        # It is possible to generate simulated spectra with other software, such as XSPEC, as long as the output format
        # remains similar. The output format is an array of spectra in counts. jaxspec is really powerful in terms of speed.
        # More models will be implemented as time goes (see the jaxspec documentation for the synthax of the models).

        Args:
            jaxspec_model_expression (_type_): _description_
            parameter_states (_type_): _description_
            thetas (_type_): _description_
            pha_file (_type_): _description_
            energy_min (_type_): _description_
            energy_max (_type_): _description_
            free_parameter_prior_types (_type_): _description_
            parameter_lower_bounds (_type_): _description_
            apply_stat (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        #
        # Apply the transformation if needed
        #
        thetas = torch.as_tensor(np.where(np.array(free_parameter_prior_types) == "loguniform" , 10. ** thetas , thetas))

        jaxspec_model = SpectralModel.from_string(jaxspec_model_expression)

        parameter_values = []
        index_theta = 0

        for i_param , param_state in enumerate(parameter_states) :
            if param_state == "free" :
                parameter_values.append([thetas[j][index_theta] for j in range(len(thetas))])
                index_theta += 1
                if verbose :
                    print(f"{param_state.lower( )} Parameter #{i_param + 1} of {jaxspec_model.n_parameters} ")

            elif param_state == "frozen" :
                parameter_values.append([parameter_lower_bounds[i_param] for j in range(len(thetas))])
                if verbose :
                    print(f"{param_state.lower( )} Parameter #{i_param + 1} of {jaxspec_model.n_parameters} ")

        params_to_set = jaxspec_model.params
        i_para = 0

        for l , param_set in params_to_set.items( ) :
            for param_name , _ in param_set.items( ) :
                upd_dict = {param_name : np.array(parameter_values[i_para])}
                param_set.update(upd_dict)
                i_para += 1

        folding_model = ObsConfiguration.from_pha_file(pha_file , energy_min , energy_max)

        if len(thetas) > 1 :
            print("Multiple thetas simulated -> parallelization with JAX required")
            start_time = time.perf_counter( )
            x = jax.jit(lambda s : fakeit_for_multiple_parameters(folding_model , jaxspec_model , s ,
                                                                apply_stat = apply_stat))(params_to_set)

            end_time = time.perf_counter( )
            duration_time = end_time - start_time
            print(f"It took just {duration_time:.1f} seconds for jax.jit to generate {len(thetas)} simulations")
        #    return torch.as_tensor(np.array(x).astype(np.float32))
        else :
            print("One single theta simulated -> parallelization with JAX not required")
            x = fakeit_for_multiple_parameters(folding_model , jaxspec_model , params_to_set , apply_stat = apply_stat)
        return torch.as_tensor(np.array(x).astype(np.float32))


# ====================================================================================================================

