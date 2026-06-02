## Util functions 
import time
from datetime import datetime

import jax
import ast
import sys
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import torch
from sklearn.linear_model import LinearRegression
import pandas as pd
from chainconsumer import ChainConsumer , PlotConfig , Chain , Truth

from jaxspec.data import ObsConfiguration
from jaxspec.data.util import fakeit_for_multiple_parameters
from jaxspec.model.abc import SpectralModel
from jaxspec.model.additive import Powerlaw, Blackbodyrad, Blackbody
from jaxspec.model.multiplicative import Tbabs, Phabs
from tabulate import tabulate


# Utility to print messages to the terminal in some format
def print_message(message):
    lines = message.split('\n')
    formatted_message="\n================================================================================\n"
    formatted_message+= '\n'.join("[LOG] " + line for line in lines)
    formatted_message+="\n================================================================================\n"
    print(formatted_message)


def summary_statistics_func(
    data: np.ndarray,
    energy_grid=None,
    energy_ref=None,
    with_basic_stats=True,
    with_sum=True,
    with_ratio=True,
    with_diff=True,
    with_energy_weighted=False,
):

    if data.ndim == 1:
        data = data[np.newaxis, :]  # (1, M)
    num_spectrum, num_bins = data.shape

    data_transformed_list = []
    labels = []

    if with_basic_stats:

        mean_x = np.mean(data, axis=1)
        std_x = np.std(data, axis=1, ddof=1)
        sum_x = np.sum(data, axis=1)

        data_transformed_list.append(mean_x)
        labels.append("Mean")
        data_transformed_list.append(std_x)
        labels.append("Std")
        data_transformed_list.append(sum_x)
        labels.append("Sum")

    if len(energy_grid) == 2:

        energies = energy_ref
        energy_bins_summary = np.append(energies[0], energies[1, -1])
        idx_low = np.searchsorted(energy_bins_summary, energy_grid.min())
        idx_high = np.searchsorted(energy_bins_summary, energy_grid.max())
        energy_bins_summary = energy_bins_summary[idx_low:idx_high + 1]

    else:
        energy_bins_summary = energy_grid

    counts = np.zeros((num_spectrum, len(energy_bins_summary),))
    energy_low_observation, energy_high_observation = energy_ref[:-1], energy_ref[1:]

    for i, (e_low_summary, e_high_summary) in enumerate(zip(energy_bins_summary[:-1], energy_bins_summary[1:])):
        counts_in_bin = np.sum(data[:, (energy_low_observation >= e_low_summary) & (energy_high_observation <= e_high_summary)], axis=1)
        counts[:, i] += counts_in_bin

        if with_sum:

            data_transformed_list.append(counts_in_bin)
            labels.append(f"Sums in band {e_low_summary:.4f}-{e_high_summary:.4f}")

    epsilon = 1
    # Hardness ratios
    if with_ratio:
        hardness_ratios = counts[:, 1:] / (counts[:, :-1] + epsilon)

        for i, (e_low_1, e_high_1, e_low_2, e_high_2) in enumerate(
                zip(
                    energy_bins_summary[:-2],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[2:]
                )):

            data_transformed_list.append(hardness_ratios[:, i])
            labels.append(f"Hardness ratio [{e_low_2:.2f}-{e_high_2:.2f}]/[{e_low_1:.2f}-{e_high_1:.2f}]")

    # Differential ratios
    if with_diff:
        differential_ratios = (counts[:, :-1] - counts[:, 1:]) / (counts[:, :-1] + counts[:, 1:] + epsilon)

        for i, (e_low_1, e_high_1, e_low_2, e_high_2) in enumerate(
                zip(
                    energy_bins_summary[:-2],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[2:]
                )):

            data_transformed_list.append(differential_ratios[:, i])
            labels.append(f"Differential ratio [{e_low_2:.2f}-{e_high_2:.2f}]/[{e_low_1:.2f}-{e_high_1:.2f}]")

    if with_energy_weighted:
        for i, (e_low_summary, e_high_summary) in enumerate(zip(energy_bins_summary[:-1], energy_bins_summary[1:])):
            idx = (energy_low_observation >= e_low_summary) & (energy_high_observation <= e_high_summary)
            average_counts = data[:, idx]

            if average_counts.sum() < len(average_counts):
                average_counts = np.ones_like(average_counts)

            average_energy = (energy_low_observation[idx] + energy_high_observation[idx])/2
            result = np.apply_along_axis(lambda x : np.average(average_energy, weights=x/x.sum()), 1, average_counts)
            data_transformed_list.append(result)
            labels.append(f"Weighted energy in {e_low_summary:.4f}-{e_high_summary:.4f}")

    data_transformed = np.column_stack(data_transformed_list)

    return data_transformed, labels




# ===============================================================================================================

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


# ===============================================================================================================
# This function computes the cstat, its expected value and variance.

def compute_cstat( data_in: object , model_in: object , with_cstat_dev=True, verbose: object = True ) -> object :
    from scipy.stats import norm
    import numpy

    #
    # From Kaastra(2017) https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..51K/abstract
    #

    def compute_ce_cv_from_kaastra_2017( mu ) :

        def f0( mu , k ) :
            import numpy as np
            import math
            #        print("before rounding,",mu,k)
            k = np.int32(k)
            pk_mu = (np.exp(-mu) * (mu ** k)) / math.factorial(k)
            if k > 0 :
                pk_mu = pk_mu * (mu - k + k * np.log(k / mu)) ** 2.
            if k == 0 :
                pk_mu = pk_mu * (mu) ** 2.

            return pk_mu

        import sys
        import numpy as np
        ce = 0.;
        cv = 0.

        if mu <= 0.5 : ce = -0.25 * mu ** 3. + 1.38 * mu ** 2. - 2. * mu * np.log(mu)
        if mu > 0.5 and mu <= 2. : ce = -0.00335 * mu ** 5 + 0.04259 * mu ** 4. - 0.27331 * mu ** 3. + 1.381 * mu ** 2. - 2. * mu * np.log(
            mu)
        if mu > 2 and mu <= 5. : ce = 1.019275 + 0.1345 * mu ** (0.461 - 0.9 * np.log(mu))
        if mu > 5 and mu <= 10. : ce = 1.00624 + 0.604 / mu ** 1.68
        if mu > 10 : ce = 1. + 0.1649 / mu + 0.226 / mu ** 2.

        if mu >= 0 and mu <= 0.1 : cv = 4. * (
                f0(mu , 0.) + f0(mu , 1.) + f0(mu , 2.) + f0(mu , 3.) + f0(mu , 4.)) - ce ** 2.
        if mu > 0.1 and mu <= 0.2 : cv = -262. * mu ** 4. + 195. * mu ** 3. - 51.24 * mu ** 2. + 4.34 * mu + 0.77005
        if mu > 0.2 and mu <= 0.3 : cv = 4.23 * mu ** 2. - 2.8254 * mu + 1.12522
        if mu > 0.3 and mu <= 0.5 : cv = -3.7 * mu ** 3. + 7.328 * mu ** 2 - 3.6926 * mu + 1.20641
        if mu > 0.5 and mu <= 1. : cv = 1.28 * mu ** 4. - 5.191 * mu ** 3 + 7.666 * mu ** 2. - 3.5446 * mu + 1.15431
        if mu > 1 and mu <= 2. : cv = 0.1125 * mu ** 4. - 0.641 * mu ** 3 + 0.859 * mu ** 2. + 1.0914 * mu - 0.05748
        if mu > 2 and mu <= 3. : cv = 0.089 * mu ** 3. - 0.872 * mu ** 2. + 2.8422 * mu - 0.67539
        if mu > 3 and mu <= 5. : cv = 2.12336 + 0.012202 * mu ** (5.717 - 2.6 * np.log(mu))
        if mu > 5 and mu <= 10. : cv = 2.05159 + 0.331 * mu ** (1.343 - np.log(mu))
        if mu > 10 : cv = 12. / mu ** 3. + 0.79 / mu ** 2. + 0.6747 / mu + 2.

        if ce == 0. or cv == 0. : sys.exit(
            "value of " + str(mu) + " not supported, please go back to Kaastra (2017)")
        #    print mu,ce,cv

        return ce , cv

    data = data_in.astype(numpy.float32)
    model = np.array(model_in).flatten( )
    #    print(np.shape(data))
    #    print(np.shape(model))

    if verbose : print("Total number of data bins=" , len(data))
    cstat = 0.
    ce_sum = 0.
    cv_sum = 0.
    chi2bfit = 0.
    for i in range(len(data)) :
        if model[i] <= 0 : model[i] = 1.0E-10
        if data[i] > 0. :  cstat += model[i] - data[i] - data[i] * np.log(model[i]) + data[i] * np.log(data[i])
        if data[i] <= 0. : cstat += model[i] - data[i] - data[i] * np.log(model[i]) + data[i]
        if data[i] > 0 : chi2bfit += ((data[i] - model[i]) ** 2) / data[i]
        if with_cstat_dev :
            ce , cv = compute_ce_cv_from_kaastra_2017(model[i])
            ce_sum += ce
            cv_sum += cv
    cstat = 2. * cstat
    if verbose : print(f"C-stat = {cstat:0.1f}")
    if verbose : print(f"Chi2  = {chi2bfit:0.1f}")
    if with_cstat_dev :
        if verbose : print(f"% Probability to get C-stat {cstat:0.1f} out of the expected C-stat {ce_sum:0.1f} "
                       f"with standard deviation {np.sqrt(cv_sum):0.1f} = {100. * norm.sf(np.abs((cstat - ce_sum) / np.sqrt(cv_sum))):0.1f}%"
                       f" - deviation ={(cstat - ce_sum) / np.sqrt(cv_sum):0.1f} sigma")

    if with_cstat_dev :
        return cstat , (cstat - ce_sum) / np.sqrt(cv_sum)
    else :
        return cstat
    

#=======================================================================================================================

# Utility to print the best bit parameters in a tabulated form.
def print_best_fit_parameters(x_obs,free_parameter_names,free_parameter_prior_types,median,lower,upper,cstat,cstat_dev):

    # Apply transformation for "loguniform" prior types without copy
    median = torch.as_tensor(np.where(np.array(free_parameter_prior_types) == "loguniform" , 10. ** median , median))
    lower = torch.as_tensor(np.where(np.array(free_parameter_prior_types) == "loguniform" , 10. ** lower , lower))
    upper = torch.as_tensor(np.where(np.array(free_parameter_prior_types) == "loguniform" , 10. ** upper , upper))


    # Create a table using a loop
    table_data = [("Parameter", "Best fit", "Negative error", "Positive error")]

    for name , m , l , u in zip(free_parameter_names , median , lower , upper) :
        table_data.append((name , f"{m:0.3f}" , f"-{m - l:0.3f}" , f"+{u - m:0.3f}"))

    # Print the table
    print(tabulate(table_data, tablefmt = "fancy_grid"))
    print_message(f"These are the best fit results\nBest fit c-stat={cstat:.3f} ({len(x_obs)-len(free_parameter_names):d} d.o.f) - c-stat deviation={cstat_dev:.3f}")




#=======================================================================================================================
# Functions for comparison plots
#=======================================================================================================================

def plot_sixsa_and_prc_posteriors(sixsa_run, sbi_run, output_path="comparison_posteriors.png"):
    """
    Plot posterior distributions from two SBI runs overlaid on each other.

    Parameters
    ----------
    sixsa_run : sixsa object
        The primary SIXSA run. Must have already called plot_posterior_results_at_x_obs()
        or otherwise populated best_fit_parameters and the posterior attribute.
    sbi_run : sixsa-compatible object
        A second run with the same interface as sixsa_run.
    output_path : str
        Path to save the output PNG.

    Returns
    -------
    str
        The output path of the saved PNG.
    """

    def get_posterior_samples(run):
        if run.use_summary:
            x = torch.as_tensor(np.array(run.x_obs_summary))
        else:
            x = torch.as_tensor(np.array(run.x_obs_reference))
        samples = run.posterior.sample((run.number_of_posterior_samples,), x=x)
        return samples

    def make_plot_title(run):
        if run.type_of_inference == "single round inference":
            return f"SRI ({run.number_of_simulations_for_train_set:d} simulations)"
        elif run.type_of_inference == "multiple round inference":
            return (
                f"MRI ({run.number_of_simulations_for_train_set:d}"
                f" x {run.number_of_rounds_for_multiple_inference:d} simulations)"
            )
        return "Unknown inference type"

    # --- Sample both posteriors ---
    samples_sixsa = get_posterior_samples(sixsa_run)
    samples_sbi   = get_posterior_samples(sbi_run)

    df_sixsa = pd.DataFrame(
        samples_sixsa.numpy() if hasattr(samples_sixsa, "numpy") else np.array(samples_sixsa),
        columns=sixsa_run.free_parameter_names_for_plots_transformed,
    )
    df_sbi = pd.DataFrame(
        samples_sbi.numpy() if hasattr(samples_sbi, "numpy") else np.array(samples_sbi),
        columns=sbi_run.free_parameter_names_for_plots_transformed,
    )

    title_sixsa = make_plot_title(sixsa_run)
    title_sbi   = make_plot_title(sbi_run)

    # --- Build ChainConsumer figure ---
    c = ChainConsumer()
    c.set_plot_config(
        PlotConfig(usetex=True, serif=True, label_font_size=18, tick_font_size=14)
    )

    c.add_chain(Chain(samples=df_sixsa, name=title_sixsa + " [SIXSA]", color="blue", bar_shade=True))
    c.add_chain(Chain(samples=df_sbi,   name=title_sbi   + " [PRC]",   color="red",  bar_shade=True))

    # Median truths for each run
    truth_sixsa = dict(zip(df_sixsa.columns.tolist(), np.array(df_sixsa.median())))
    truth_sbi   = dict(zip(df_sbi.columns.tolist(),   np.array(df_sbi.median())))

    c.add_truth(Truth(location=truth_sixsa, color="blue"))
    c.add_truth(Truth(location=truth_sbi,   color="red"))

    fig = c.plotter.plot(figsize=(8, 10))
    fig.align_ylabels()
    fig.align_xlabels()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison posterior plot to: {output_path}")

    return output_path

#=======================================================================================================================
