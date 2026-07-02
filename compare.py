################################################################## 
# Code to compare the results from SIXSA and the reproduced code
##################################################################

import dill as pickle
import sys
import os

from compare_utils import plot_sixsa_and_prc_posteriors, get_posterior_samples, get_spectrum_data, compare_spectra

# Import each compute_x_sim from its own directory
sys.path.insert(0, "./")
from prc_utils import compute_x_sim as compute_x_sim_prc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SIXSA", "SIXSA_CODES"))
from SIXSA.SIXSA_CODES.sixsa_class import compute_x_sim as compute_x_sim_sixsa




### INPUT PARAMETERS ########################################################
yml_filename = '1_sri_config_cmin_cmax_restrictor_spectrum_20000'
yml_path = 'SIXSA/SIXSA_YML_INPUT_FILES/' + yml_filename + ".yml"
output_dir = 'OUTPUTS/'

sixsa_dir = os.path.join(os.path.dirname(__file__), "SIXSA")
sixsa_path = 'SIXSA/SIXSA_OUTPUTS/' + yml_filename + "_run_results.pkl"
prc_path = output_dir + yml_filename + "_run_results.pkl"
#############################################################################

# import SIXSA and reproduced results
with open(sixsa_path, "rb") as f:
    sixsa_run = pickle.load(f)

with open(prc_path, "rb") as f:
    sbi_run = pickle.load(f)


# redraw samples (not pickled)
samples_sixsa = get_posterior_samples(sixsa_run)
samples_prc = get_posterior_samples(sbi_run)

# Pass into both functions to avoid resampling
spectra_data_sixsa = get_spectrum_data(sixsa_run, 
                                       compute_x_sim_fn=compute_x_sim_sixsa,
                                       posterior_samples=samples_sixsa,
                                       working_dir=sixsa_dir)

spectra_data_prc = get_spectrum_data(sbi_run, 
                                     compute_x_sim_fn=compute_x_sim_prc,
                                     posterior_samples=samples_prc
                                     )


# plot the comparison of both posteriors
plot_sixsa_and_prc_posteriors(sixsa_run, samples_sixsa, 
                              sbi_run, samples_prc, 
                              output_path=output_dir+yml_filename+"_comparison.png")


# compare posterior sampled spectra
compare_spectra(sixsa_run, sbi_run, 
                spectra_data_sixsa, spectra_data_prc,
                output_path=output_dir+yml_filename+"comparison_spectra.png")