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
import click
import dill as pickle
from scipy.stats import norm

from sbi import utils
from sbi.utils import RestrictionEstimator
from sbi.inference import SNPE
from sbi.analysis import pairplot, check_sbc, run_sbc, get_nltp, sbc_rank_plot



# from sbi.utils import RestrictionEstimator , RestrictedPrior , get_density_thresholder
from jaxspec.data import ObsConfiguration
from jaxspec.data.util import fakeit_for_multiple_parameters
from jaxspec.model.abc import SpectralModel

from prc_utils import summary_statistics_func, print_message, print_best_fit_parameters, \
                        compute_x_sim, compute_cstat, generate_function_for_cmin_cmax_restrictor


# class for performing sbi
class sbi_run():
    
    # ===============================================================================================================
    # initialise the class

    def __init__(self, yml_file):
        

        # open config yaml file
        self.yml_file = yml_file
        with open(yml_file , 'r') as config_file :
            self.config = yaml.safe_load(config_file)

        print('HERE', yml_file)

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

        # don't use the root here, output to main dir
        self.path_outputs =  self.config["path_outputs"]
        if not os.path.exists(self.path_outputs) :
            os.makedirs(self.path_outputs)
            print(f"Directory '{self.path_outputs}' created.")
        else :
            print(f"Directory '{self.path_outputs}' already exists.")

        self.root_output_files = os.path.basename(self.yml_file).replace(".yml" , "_")
        print('HERE', self.root_output_files)
        self.x_obs_reference=None
        self.use_summary=self.config.get("use_summary_statistics", False)


        # initialise plotting settings
        plt.rcParams.update({
        "text.usetex": True,        
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
         "axes.labelpad": 10,
        })


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
    def compute_restricted_prior(self):


        if self.restricted_prior_type=="cmin_cmax_restricted_prior" :
            
            generate_restrictor_function_kwargs = {"cmin" : self.c_min_for_restricted_prior ,
                                                   "cmax" : self.c_max_for_restricted_prior}
            
            select_good_x = generate_function_for_cmin_cmax_restrictor(**generate_restrictor_function_kwargs)


            restriction_estimator = RestrictionEstimator(decision_criterion=select_good_x, prior=self.prior)
            self.rp_proposals = [self.prior]

            for r in range(self.number_of_rounds_for_restricted_prior):
                # draw theta from restricted prior and simulate
                theta = self.rp_proposals[-1].sample((self.number_of_simulations_for_restricted_prior,))
                x = compute_x_sim(self.jaxspec_model_expression, self.parameter_states,
                                            theta,
                                            self.pha_filename, self.energy_min, self.energy_max,
                                            self.free_parameter_prior_types, self.parameter_lower_bounds,
                                            apply_stat=False, verbose=True)
                
                # add simulations
                restriction_estimator.append_simulations(theta, x)
                
                # training not needed in last round because classifier will not be used anymore.
                if (r < self.number_of_rounds_for_restricted_prior - 1):     
                    classifier = restriction_estimator.train()
                
                self.rp_proposals.append(restriction_estimator.restrict_prior())
                
            self.restricted_prior = self.rp_proposals[-1]

        else: 
            self.restricted_prior = None
        
            



    # ===============================================================================================================

    def plot_priors(self):
        
        # whether to plot multiple rounds of restriction
        plot_rp_rounds = [1,2, 3, -1]
        file_addon = ""

        # draw from the global prior and restricted
        theta_from_global_prior = self.prior.sample((10 * self.number_of_posterior_samples,))
        df_theta_from_global_prior = pd.DataFrame(theta_from_global_prior,
                                                columns=self.free_parameter_names_for_plots_transformed)

        c = ChainConsumer()
        c.set_plot_config(PlotConfig(usetex=True, serif=True, label_font_size=18, tick_font_size=14))
        c.add_chain(Chain(samples=df_theta_from_global_prior,
                        name="Global initial prior",
                        color="blue", bar_shade=True))
        
        # plot restricted priors if made
        if self.restricted_prior is not None:            
            if plot_rp_rounds is not None:
                file_addon = "_rp_rounds"
                cols = ['red', 'orange', 'green']
                for col, r in zip(cols, plot_rp_rounds):
                    theta_from_restricted_prior = self.rp_proposals[r].sample((10 * self.number_of_posterior_samples,))
                    df_theta_from_restricted_prior = pd.DataFrame(theta_from_restricted_prior,
                                                            columns=self.free_parameter_names_for_plots_transformed)
                    c.add_chain(Chain(samples=df_theta_from_restricted_prior,
                                name=f"Restricted prior R={r}",
                                color=col, bar_shade=True))



        fig = c.plotter.plot(figsize=(8, 10))
        fig.align_ylabels()
        fig.align_xlabels()

        png_filename = self.path_outputs + self.root_output_files + "prior" + file_addon + ".png"
        fig.savefig(png_filename, dpi=150, bbox_inches="tight")
        plt.close()



    # ===============================================================================================================
    def generate_train_and_test_data(self):
        
        # select prior type
        if self.restricted_prior is not None:
            prior = self.restricted_prior
        else:
            prior = self.prior


        # generate sample pairs from training
        self.theta_train = prior.sample((self.number_of_simulations_for_train_set,))
        print(f"Generating the simulations that will be used for the inference")
        start_time = time.perf_counter( )
        self.x_train = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , self.theta_train ,
                                self.pha_filename ,
                                self.energy_min , self.energy_max ,
                                self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                verbose = False)
        
        # generate sample pairs from testing
        self.theta_test = prior.sample((self.number_of_simulations_for_test_set ,))
        self.x_test = compute_x_sim(self.jaxspec_model_expression , self.parameter_states, 
                                    self.theta_test , self.pha_filename , self.energy_min , self.energy_max ,
                                    self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                verbose = False)
        end_time = time.perf_counter( )
        print(f'It took {end_time - start_time: 0.2f} second(s) to complete the simulations to be used for the inference ')
        self.duration_generation_theta_x = end_time - start_time
        
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



    #===================================================================================================================
    def run_sri(self):


        # initialise NPE trainer object
        inference = SNPE(prior = self.prior)
        
        # add simulations
        inference.append_simulations(self.theta_train, self.x_train)

        # train the density estimator NN
        density_estimator = inference.train()

        # construct DirectPosterior object
        self.posterior = inference.build_posterior(density_estimator)

        # run the observed spectra through the density estimator and sample from the cond'd distribution
        self.posterior_theta = self.posterior.sample((self.number_of_posterior_samples,),
                                                    x=self.x_obs_reference)


        # get median theta and percentiles
        self.best_fit_parameters = np.median(self.posterior_theta, axis=0)
        # self.mean_theta = np.mean(self.posterior_theta, axis=0)
        self.best_fit_parameters_lower_bounds, self.best_fit_parameters_upper_bounds = np.percentile(self.posterior_theta,
                                                                                        (16, 84), axis=0)

        # compute the median spectra
        self.x_from_median = compute_x_sim(self.jaxspec_model_expression, self.parameter_states,
                                        torch.tensor([self.best_fit_parameters]),
                                        self.pha_filename, self.energy_min, self.energy_max,
                                        self.free_parameter_prior_types, self.parameter_lower_bounds,
                                        apply_stat=False, verbose=True)

        
        # Now computing the cstat of the best fit and its deviation against the expected value
        # From Kaastra(2017) https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..51K/abstract
        #
        self.cstat_median_posterior_sample, self.cstat_dev_median_posterior_sample = compute_cstat(self.x_obs_reference, 
                                                                                                   np.array(self.x_from_median), 
                                                                                                   verbose = True)

        print_best_fit_parameters(self.x_obs_reference, self.free_parameter_names_for_plots , 
                                  self.free_parameter_prior_types ,self.best_fit_parameters , 
                                  self.best_fit_parameters_lower_bounds ,self.best_fit_parameters_upper_bounds ,
                                  self.cstat_median_posterior_sample , self.cstat_dev_median_posterior_sample)


        # compute spectra x for all posterior samples
        self.x_from_posterior_sample = compute_x_sim(self.jaxspec_model_expression,
                                                    self.parameter_states, self.posterior_theta,
                                                    self.pha_filename, self.energy_min, self.energy_max,
                                                    self.free_parameter_prior_types, self.parameter_lower_bounds,
                                                    apply_stat=True, verbose=True)
        

    # ==============================================================================================================
    def sbc_calibration(self):
        
        # select prior type
        if self.restricted_prior is not None:
            prior = self.restricted_prior
            filename_addon = "_restricted"
        else:
            prior = self.prior
            filename_addon = ""

        # sbc parameters
        num_sbc_runs = 1000  # should be ~100s or ideally 1000
        num_posterior_samples_sbc = 1000 # number of posterior samples per sbc run

        # generate ground truth parameters and corresponding simulated observations for sbc
        theta_sbc = prior.sample((num_sbc_runs,))
        x_sbc = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , theta_sbc ,
                                self.pha_filename ,
                                self.energy_min , self.energy_max ,
                                self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                verbose = False)

        # run sbc calibration
        ranks, dap_samples = run_sbc(
            theta_sbc, x_sbc, self.posterior, num_posterior_samples=num_posterior_samples_sbc
        )

        # collect metrics to judge posterior 
        check_stats = check_sbc(
            ranks, theta_sbc, dap_samples, num_posterior_samples=num_posterior_samples_sbc
        )

        print(f'summary statistics:\n'
              + f"kolmogorov-smirnov p-values \ncheck_stats['ks_pvals'] = {check_stats['ks_pvals'].numpy()}"
              + f"c2st accuracies \ncheck_stats['c2st_ranks'] = {check_stats['c2st_ranks'].numpy()}"
              + f"- c2st accuracies check_stats['c2st_dap'] = {check_stats['c2st_dap'].numpy()}")

        # histograms plot
        # ranks = ranks / num_posterior_samples_sbc # normalise ranks
        f, ax = sbc_rank_plot(
            ranks= ranks,
            num_posterior_samples=num_posterior_samples_sbc,
            plot_type="hist",
            num_bins=None,  # by passing None we use a heuristic for the number of bins.
        )

        for i, (axs, lab) in enumerate(zip(ax, self.free_parameter_names_for_plots_transformed)):
            axs.set_title(f"{lab} \nks_pval: {check_stats['ks_pvals'][i]:.2g}"
                          + f"\nc2st_ranks: {check_stats['c2st_ranks'][i]:.2g}"
                          + f"\nc2st_dap: {check_stats['c2st_dap'][i]:.2g}")
            axs.set_xlabel(f"Rank" )

        f.set_size_inches(10, 4)
        png_filename = self.path_outputs + self.root_output_files + "sbc_rank_plot" + filename_addon + ".png"
        f.savefig(png_filename, dpi=300, bbox_inches="tight")

        ### cumlative rank plot ###
        num_bins_used = num_sbc_runs // 20  # sbi's heuristic, since num_bins=None
        f, ax = sbc_rank_plot(
            ranks= ranks,
            num_posterior_samples=num_posterior_samples_sbc,
            plot_type="cdf",
            parameter_labels=self.free_parameter_names_for_plots_transformed,
            num_bins=num_bins_used,  # by passing None we use a heuristic for the number of bins.
        )

        ax.set_xlim(0, num_bins_used)
        ax.set_xticks(np.linspace(0, num_bins_used, 5))
        ax.set_xticklabels([f"{t:.2f}" for t in np.linspace(0, 1, 5)])
        ax.set_xlabel("Rank")
        ax.set_title('Rank Cumulative Density Function (PRC)')
        f.set_size_inches(8, 6)

        png_filename = self.path_outputs + self.root_output_files + "sbc_rank_plot_cumulative" + filename_addon + ".png"
        f.savefig(png_filename, dpi=300, bbox_inches="tight")
        


    # ==============================================================================================================
    def coverage_zz_plot(self):


        # select prior type
        if self.restricted_prior is not None:
            prior = self.restricted_prior
            filename_addon = "_restricted"
        else:
            prior = self.prior
            filename_addon = ""

        ### ------------------------------------
        ## define expected coverage parameters
        num_coverage_samples = 500 # i...L
        num_posterior_samples = 1000 # m...M

        # credible levels to test, e.g. central intervals of width alpha
        alphas = np.linspace(0.01, 0.99, 50)
        ### ------------------------------------

        # draw calibration sample pairs
        prior_samples = prior.sample((num_coverage_samples,))
        prior_predictives = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , prior_samples ,
                                self.pha_filename ,
                                self.energy_min , self.energy_max ,
                                self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                verbose = False)

        

        # for each simulation, draw posterior samples and compute the rank
        empirical_coverage = np.zeros_like(alphas)
        all_ranks = []

        for i in range(num_coverage_samples):
            theta_true = prior_samples[i]
            x_obs = prior_predictives[i]
            posterior_samples_i = self.posterior.sample((num_posterior_samples,), x=x_obs, 
                                                        show_progress_bars=False)

            # compute the joint log prob for each sample
            log_prob_true = self.posterior.log_prob(theta_true.unsqueeze(0), x=x_obs, norm_posterior=False)
            log_prob_samples = self.posterior.log_prob(posterior_samples_i, x=x_obs, norm_posterior=False)
            
            # compute ranks and normalise (0,1)
            rank = (log_prob_true <= log_prob_samples).sum().item()
            all_ranks.append(rank / num_posterior_samples)  

        all_ranks = np.array(all_ranks)

        # compute the indicator function and emperical coverage per alpha
        # the truth is 'covered' by the alpha HPD region if the log-prob is greater than 1-alpha
        for j, alpha in enumerate(alphas): 
            empirical_coverage[j] = np.mean(all_ranks <= (1 - alpha))
        

        # probit-transform both axes to get z-score framing
        z_nominal = norm.ppf(1 - alphas / 2)     
        z_empirical = norm.ppf((empirical_coverage / 2) + 0.5)


        # make the plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(z_nominal, z_empirical, label = "PRC coverage")
        

        max_z = max(z_nominal.max(), z_empirical.max())
        ax.plot([0, max_z], [0, max_z], "k--", lw=1, label="calibrated")

        ax.set_title('$z-z$ plot (PRC)')
        ax.set_xlabel(r"Nominal coverage ($z$-score)")
        ax.set_ylabel(r"Empirical coverage ($\hat{z}$-score)")
        
        ax.set_xlim(0,2.5)
        ax.set_ylim(0,2.5)

        ax.tick_params(axis="both", which="both", top=True, right=True, direction="in")
        ax.minorticks_on()
        ax.tick_params(axis="both", which="minor", top=True, right=True, direction="in")
        
        ax.legend()
        ax.grid()
        fig.tight_layout()

        png_filename = self.path_outputs + self.root_output_files + "coverage_zz" + filename_addon + ".png"
        fig.savefig(png_filename, dpi=300, bbox_inches="tight")

    
    # ==============================================================================================================
    def plot_sri_posteriors(self):

        # create the dataframe for chain consumer
        df4cc = pd.DataFrame(self.posterior_theta, columns=self.free_parameter_names_for_plots_transformed)

        if self.type_of_inference == "single round inference":
            plot_title = f"SRI ({self.number_of_simulations_for_train_set:d} simulations)"
        # elif self.type_of_inference == "multiple round inference":
            # plot_title = f"MRI ({self.number_of_simulations_for_train_set:d} x {self.number_of_rounds_for_multiple_inference:d} simulations)"

        c = ChainConsumer()
        c.set_plot_config(PlotConfig(usetex=True, serif=True, label_font_size=18, tick_font_size=14))
        c.add_chain(Chain(samples=df4cc, name=plot_title, color="blue", bar_shade=True))

        # add the samples median (not the truth, this method is used for convenience)
        truth_sri = dict(zip(df4cc.columns.values.tolist(), np.array(df4cc.median())))
        c.add_truth(Truth(location=truth_sri, color="blue"))

        fig = c.plotter.plot(figsize=(8, 10))
        fig.align_ylabels()
        fig.align_xlabels()

        png_filename = self.path_outputs + self.root_output_files + "posteriors_at_reference_spectrum.png"
        fig.savefig(png_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


        # try log prob plot
        log_probs = self.posterior.log_prob(self.posterior_theta, x=self.x_obs_reference)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        param_names = [r'$\log(N_h)$', r'$\Gamma$', r'$\log(N_{pl})$']

        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                if i == j:
                    ax.hist(self.posterior_theta[:, i].numpy(), bins=50)
                elif i > j:
                    sc = ax.scatter(
                        self.posterior_theta[:, j].numpy(),
                        self.posterior_theta[:, i].numpy(),
                        c=log_probs.numpy(),
                        cmap='viridis',
                        s=1,
                        alpha=0.5
                    )
                    plt.colorbar(sc, ax=ax, label=r'$\log q_\phi$')
                else:
                    ax.axis('off')
                
                if i == 2: ax.set_xlabel(param_names[j])
                if j == 0: ax.set_ylabel(param_names[i])

        png_filename = self.path_outputs + self.root_output_files + "log_probs.png"
        fig.savefig(png_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


    #===================================================================================================================
    def plot_sri_spectrum(self):

        if self.type_of_inference == "single round inference":
            plot_title = f"SRI ({self.number_of_simulations_for_train_set:d} simulations)"
        # elif self.type_of_inference == "multiple round inference":
            # plot_title = f"MRI ({self.number_of_simulations_for_train_set:d} x {self.number_of_rounds_for_multiple_inference:d} simulations)"

        # compute Gehrels approximation (low counts) for uncertainty
        gehrels_error_counts = (1. + (0.75 + np.array(self.x_obs_reference)) ** 0.5)

        # compute sigma-scaled residuals
        best_fit_residuals = (np.array(self.x_obs_reference) - np.array(self.x_from_median)) / np.array(gehrels_error_counts)

        # create plots
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True, height_ratios=[0.8, 0.2])
        plt.subplots_adjust(hspace=0.0)

        # plotting the data, best fit, and coverage
        ax[0].step(0.5 * (self.e_min_folded + self.e_max_folded), self.x_obs_reference, where="mid",
                label=f"Observed spectrum ({np.int32(np.sum(self.x_obs_reference)):d} counts)",
                color="black")
        ax[0].step(0.5 * (self.e_min_folded + self.e_max_folded), self.x_from_median.flatten(), where="mid",
                label=f"Best fit ({self.cstat_median_posterior_sample:0.1f}, {self.cstat_dev_median_posterior_sample:0.1f}$\sigma$)",
                color="blue")
        
        # coverage is defined over the spectral range, not in theta space
        ax[0].fill_between(
            0.5 * (self.e_min_folded + self.e_max_folded),
            *np.percentile(self.x_from_posterior_sample, [16, 84], axis=0),
            alpha=0.3,
            color="green",
            step="mid",
            label=r"$1-\sigma$ coverage",
        )
        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_ylabel("Counts")
        ax[0].legend(frameon=False)
        ax[0].set_title(plot_title)

        # Plotting residuals
        ax[1].step(0.5 * (self.e_min_folded + self.e_max_folded), best_fit_residuals.flatten(),
                label="Residuals", color="black")
        color = (0.15, 0.25, 0.45)
        ax[1].axhline(0, color=color, ls="--")
        ax[1].axhline(-3, color=color, ls=":")
        ax[1].axhline(3, color=color, ls=":")

        ax[1].set_yticks([-3, 0, 3], labels=[-3, 0, 3])
        ax[1].set_yticks(range(-3, 4), minor=True)
        ax[1].set_ylabel(r"Residuals ($\sigma$)")

        ax[1].set_xticks(self.logscale_values_rounded, labels=self.logscale_values_rounded)
        ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[1].set_xlabel("Energy (keV)")

        fig.align_ylabels()
        fig.tight_layout()

        png_filename = self.path_outputs + self.root_output_files + "reference_spectrum_and_folded_model.png"
        fig.savefig(png_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


    # ====================================================================================================================
    def load_run_from_pickle_file(self):
        self.pkl_filename = self.path_outputs + self.root_output_files + "run_results.pkl"

        with open(self.pkl_filename, "rb") as handle:
            loaded_self = pickle.load(handle)

        self.__dict__.update(loaded_self.__dict__)


    # ====================================================================================================================
    def save_run_in_pickle_file(self):
        self.pkl_filename=self.path_outputs + self.root_output_files + "run_results.pkl"

        if os.path.exists(self.pkl_filename) and click.confirm(
                f"{self.pkl_filename} exists. Do you still want to save the run results?" , default = False) :
            with open(self.pkl_filename , "wb") as handle :
                pickle.dump(self , handle , pickle.HIGHEST_PROTOCOL)
        else :
            with open(self.pkl_filename , "wb") as handle :
                pickle.dump(self , handle , pickle.HIGHEST_PROTOCOL)

        




# ====================================================================================================================

