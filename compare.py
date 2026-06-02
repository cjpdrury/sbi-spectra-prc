################################################################## 
# Code to compare the results from SIXSA and the reproduced code
##################################################################

import dill as pickle
import sys
import os


import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib.ticker
from chainconsumer import ChainConsumer, Chain, Truth, PlotConfig

# Add the SIXSA class directory to path so pickle can find sixsa_class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SIXSA", "SIXSA_CODES"))


### INPUT PARAMETERS ###
yml_filename = '1_sri_config_cmin_cmax_restrictor_spectrum_20000'
yml_path = 'SIXSA/SIXSA_YML_INPUT_FILES/' + yml_filename + ".yml"

sixsa_path = 'SIXSA/SIXSA_OUTPUTS/' + yml_filename + "_run_results.pkl"
output_dir = 'OUTPUTS/'

# import SIXSA results
with open(sixsa_path, "rb") as f:
    sixsa_run = pickle.load(f)

# for key, value in sixsa_run.__dict__.items():
    # print(key, type(value))



import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.backends.backend_pdf
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chainconsumer import ChainConsumer, Chain, Truth, PlotConfig


def compare_posteriors(sixsa_run, sbi_run, output_path="comparison_posteriors.png"):
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
    c.add_chain(Chain(samples=df_sbi,   name=title_sbi   + " [REPRO]",   color="red",  bar_shade=True))

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


# --- Example usage ---
# from compare_posteriors import compare_posteriors
import copy
sixsa_run_copy = copy.deepcopy(sixsa_run)
compare_posteriors(sixsa_run, sixsa_run_copy, output_path=output_dir+yml_filename+"_comparison.png")


