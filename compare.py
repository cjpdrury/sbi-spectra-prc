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

from prc_utils import plot_sixsa_and_prc_posteriors


### INPUT PARAMETERS ########################################################
yml_filename = '1_sri_config_cmin_cmax_restrictor_spectrum_20000'
yml_path = 'SIXSA/SIXSA_YML_INPUT_FILES/' + yml_filename + ".yml"
output_dir = 'OUTPUTS/'

sixsa_path = 'SIXSA/SIXSA_OUTPUTS/' + yml_filename + "_run_results.pkl"
prc_path = output_dir + yml_filename + "_run_results.pkl"
#############################################################################

# import SIXSA and reproduced results
with open(sixsa_path, "rb") as f:
    sixsa_run = pickle.load(f)

with open(prc_path, "rb") as f:
    sbi_run = pickle.load(f)


# plot the comparison of both posteriors
plot_sixsa_and_prc_posteriors(sixsa_run, sbi_run, output_path=output_dir+yml_filename+"_comparison.png")


