# import modules
import time
import jax
import sys
import os
import yaml
import matplotlib
from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer , PlotConfig , Chain , Truth
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from sbi import utils
from sbi.inference import SNPE
# from sbi.utils import RestrictionEstimator , RestrictedPrior , get_density_thresholder
from jaxspec.data import ObsConfiguration
from jaxspec.data.util import fakeit_for_multiple_parameters
from jaxspec.model.abc import SpectralModel

from prc_utils import summary_statistics_func, print_message, compute_x_sim


# class for performing sbi
class sbi_run():
    
    # ===============================================================================================================
    # initialise the class

    def __init__(self, yml_file):
        
        # open config yaml file
        self.yml_file = yml_file
        with open(yml_file , 'r') as config_file :
            self.config = yaml.safe_load(config_file)

        # Create a list of tuples containing variable name and value pairs
        table_data = [(key , value) for key , value in  self.config.items( )]

        # Create a frame around the table and print it
        print(tabulate(table_data , headers = ["Variable" , "Value"] , tablefmt = "fancy_grid"))

        # read parameters
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


    # ===============================================================================================================    
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

        # initialise uniform prior from parameters
        self.prior = utils.BoxUniform(low = low_v , high = high_v)
        

    # ===============================================================================================================

    def plot_prior(self):
        theta_from_global_prior = self.prior.sample((10 * self.number_of_posterior_samples,))
        df_theta_from_global_prior = pd.DataFrame(theta_from_global_prior,
                                                columns=self.free_parameter_names_for_plots_transformed)

        c = ChainConsumer()
        c.set_plot_config(PlotConfig(usetex=True, serif=True, label_font_size=18, tick_font_size=14))
        c.add_chain(Chain(samples=df_theta_from_global_prior,
                        name="Global initial prior",
                        color="blue", bar_shade=True))

        fig = c.plotter.plot(figsize=(8, 10))
        fig.align_ylabels()
        fig.align_xlabels()

        png_filename = self.path_outputs + self.root_output_files + "prior.png"
        fig.savefig(png_filename, dpi=150, bbox_inches="tight")
        plt.close()



    # ===============================================================================================================
    def generate_train_and_test_data(self):
        
        # generate sample pairs from training
        self.theta_train = self.prior.sample((self.number_of_simulations_for_train_set,))
        self.x_train = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , self.theta_train ,
                                self.pha_filename ,
                                self.energy_min , self.energy_max ,
                                self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                verbose = False)
        
        # generate sample pairs from testing
        self.theta_test = self.prior.sample((self.number_of_simulations_for_test_set ,))
        self.x_test = compute_x_sim(self.jaxspec_model_expression , self.parameter_states, 
                                    self.theta_test , self.pha_filename , self.energy_min , self.energy_max ,
                                    self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                verbose = False)
        
    # ===============================================================================================================   
    def plot_prior_predictive_check(self):
        png_filename = self.path_outputs + self.root_output_files + "prior_predictive_check.png"
        
        fig, ax = plt.subplots(1, 1)
        plt.step(0.5 * (self.e_min_folded + self.e_max_folded), self.x_obs_reference, where="mid", color="red", linewidth=2., label=f"Observed spectrum ({np.int32(np.sum(self.x_obs_reference)):d} counts)")

        plt.fill_between(0.5 * (self.e_min_folded + self.e_max_folded),
            *np.percentile(self.x_train, [0., 100], axis=0),
            color="grey",
            alpha=0.2,
            step="mid",
            label=r"Prior coverage")

        self.logscale_values_low = np.logspace(np.log10(np.min(self.e_min_folded)), np.log10(1.), num=5, endpoint=False)
        self.logscale_values_high = np.logspace(np.log10(1.), np.log10(np.max(self.e_max_folded)), num=6, endpoint=True)
        self.logscale_values = np.concatenate((self.logscale_values_low, self.logscale_values_high))
        self.logscale_values_rounded = [round(val, 1) if val < 1 else int(val) for val in self.logscale_values]

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(self.logscale_values_rounded)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        plt.legend(frameon=False)
        plt.savefig(png_filename, dpi=150, bbox_inches="tight")
        plt.close()


# ====================================================================================================================

