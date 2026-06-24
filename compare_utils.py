#=======================================================================================================================
# Functions for comparison plots
#=======================================================================================================================

## Util functions 
import time
from datetime import datetime
import os
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

# Ensure SIXSA's own directory is on the path so its internal imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SIXSA", "SIXSA_CODES"))
from SIXSA.SIXSA_CODES.sixsa_class import compute_x_sim, compute_cstat

#=======================================================================================================================

def get_posterior_samples(run):
    if run.use_summary:
        x = torch.as_tensor(np.array(run.x_obs_summary))
    else:
        x = torch.as_tensor(np.array(run.x_obs_reference))
    samples = run.posterior.sample((run.number_of_posterior_samples,), x=x)
    return samples


#=======================================================================================================================

def make_plot_title(run):
    if run.type_of_inference == "single round inference":
        return f"SRI ({run.number_of_simulations_for_train_set:d} simulations)"
    elif run.type_of_inference == "multiple round inference":
        return (
            f"MRI ({run.number_of_simulations_for_train_set:d}"
            f" x {run.number_of_rounds_for_multiple_inference:d} simulations)"
        )
    return "Unknown inference type"


#=======================================================================================================================
def plot_sixsa_and_prc_posteriors(sixsa_run, samples_sixsa, sbi_run, samples_sbi, output_path="comparison_posteriors.png"):
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

def get_spectrum_data(run, compute_x_sim_fn, posterior_samples=None, working_dir=None):
    """
    Compute best-fit spectrum, posterior predictive envelope, residuals, and
    cstat for a run.
 
    Parameters
    ----------
    run : sixsa object
    posterior_samples : tensor, optional
        Pre-drawn samples. If None, samples are drawn fresh via
        get_posterior_samples(run), avoiding redundant sampling when
        compare_posteriors and compare_spectra are both called.
 
    Returns
    -------
    dict with keys:
        posterior_samples, x_from_median, x_from_posterior_sample,
        residuals, cstat, cstat_dev
    """
    if posterior_samples is None:
        posterior_samples = get_posterior_samples(run)
 
    best_fit_parameters = np.median(posterior_samples, axis=0)
 
    original_dir = os.getcwd()
    if working_dir:
        os.chdir(working_dir)
    
    try:
        x_from_median = compute_x_sim_fn(
            run.jaxspec_model_expression,
            run.parameter_states,
            torch.tensor([best_fit_parameters]),
            run.pha_filename,
            run.energy_min,
            run.energy_max,
            run.free_parameter_prior_types,
            run.parameter_lower_bounds,
            apply_stat=False,
            verbose=False,
        )
    
        cstat, cstat_dev = compute_cstat(
            run.x_obs_reference, np.array(x_from_median), verbose=False
        )
    
        x_from_posterior_sample = compute_x_sim(
            run.jaxspec_model_expression,
            run.parameter_states,
            posterior_samples,
            run.pha_filename,
            run.energy_min,
            run.energy_max,
            run.free_parameter_prior_types,
            run.parameter_lower_bounds,
            apply_stat=True,
            verbose=False,
        )
    
        gehrels_errors = 1.0 + (0.75 + np.array(run.x_obs_reference)) ** 0.5
        residuals = (np.array(run.x_obs_reference) - np.array(x_from_median)) / gehrels_errors
    
    finally:
        os.chdir(original_dir)  # always restore, even if an exception is raised


    return {
        "posterior_samples":        posterior_samples,
        "x_from_median":            x_from_median,
        "x_from_posterior_sample":  x_from_posterior_sample,
        "residuals":                residuals,
        "cstat":                    cstat,
        "cstat_dev":                cstat_dev,
    }


#=======================================================================================================================

def compare_spectra(
    sixsa_run,
    sbi_run,
    data_a=None,
    data_b=None,
    output_path="comparison_spectra.png",
    label_a="SIXSA",
    label_b="PRC",
    color_a="blue",
    color_b="red",
):
    """
    Overlay folded spectrum plots from two SBI runs on a shared figure.

    Upper panel : observed spectrum + best-fit + 1-sigma posterior predictive
                  coverage for each run.
    Lower panel : residuals for each run.

    Parameters
    ----------
    sixsa_run : sixsa object
    sbi_run   : sixsa-compatible object
    output_path : str
    label_a, label_b : str   — legend labels for each run
    color_a, color_b : str   — colours for each run's best fit / residuals

    Returns
    -------
    str  — output_path
    """
    if data_a is None:
        print(f"Computing spectrum data for {label_a}...")
        data_a = get_spectrum_data(sixsa_run)

    if data_b is None:
        print(f"Computing spectrum data for {label_b}...")
        data_b = get_spectrum_data(sbi_run)

    e_mid_a = 0.5 * (sixsa_run.e_min_folded + sixsa_run.e_max_folded)
    e_mid_b = 0.5 * (sbi_run.e_min_folded   + sbi_run.e_max_folded)

    title_a = make_plot_title(sixsa_run)
    title_b = make_plot_title(sbi_run)

    fig, ax = plt.subplots(
        2, 1, figsize=(8, 10), sharex=True, height_ratios=[0.8, 0.2]
    )
    plt.subplots_adjust(hspace=0.0)

    # ------------------------------------------------------------------
    # Upper panel — spectra
    # ------------------------------------------------------------------

    # Observed spectrum (shared; plot once in black)
    ax[0].step(
        e_mid_a,
        sixsa_run.x_obs_reference,
        where="mid",
        color="black",
        label=f"Observed ({np.int32(np.sum(sixsa_run.x_obs_reference)):d} counts)",
    )

    for label, color, run, data, e_mid, title in [
        (label_a, color_a, sixsa_run, data_a, e_mid_a, title_a),
        (label_b, color_b, sbi_run,   data_b, e_mid_b, title_b),
    ]:
        ax[0].step(
            e_mid,
            np.array(data["x_from_median"]).flatten(),
            where="mid",
            color=color,
            label=(
                f"{label} best fit — {title} "
                f"(C={data['cstat']:.1f}, {data['cstat_dev']:.1f}$\\sigma$)"
            ),
        )
        ax[0].fill_between(
            e_mid,
            *np.percentile(data["x_from_posterior_sample"], [16, 84], axis=0),
            alpha=0.25,
            color=color,
            step="mid",
            label=f"{label} $1\\sigma$ coverage",
        )

    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].set_ylabel("Counts")
    ax[0].legend(frameon=False, fontsize=8)

    # ------------------------------------------------------------------
    # Lower panel — residuals
    # ------------------------------------------------------------------
    ref_color = (0.15, 0.25, 0.45)
    ax[1].axhline(0,  color=ref_color, ls="--")
    ax[1].axhline(-3, color=ref_color, ls=":")
    ax[1].axhline(3,  color=ref_color, ls=":")

    for label, color, data, e_mid in [
        (label_a, color_a, data_a, e_mid_a),
        (label_b, color_b, data_b, e_mid_b),
    ]:
        ax[1].step(
            e_mid,
            data["residuals"].flatten(),
            where="mid",
            color=color,
            label=f"{label} residuals",
        )

    ax[1].set_yticks([-3, 0, 3], labels=[-3, 0, 3])
    ax[1].set_yticks(range(-3, 4), minor=True)
    ax[1].set_ylabel(r"Residuals ($\sigma$)")
    ax[1].legend(frameon=False, fontsize=8)

    # Use sixsa_run's energy axis ticks (assumed shared)
    ax[1].set_xticks(
        sixsa_run.logscale_values_rounded,
        labels=sixsa_run.logscale_values_rounded,
    )
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].set_xlabel("Energy (keV)")

    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison spectra plot to: {output_path}")
    return output_path