from prc_class import sbi_run


def main():
    
    # initial params
    yml_dir = "YML_INPUT_FILES/"
    yml_file = yml_dir+"1_sri_config_cmin_cmax_restrictor_spectrum_20000.yml"

    # sections to run
    run_from_pickle = False
    make_plots = True
    run_calibration = True
    


    # setup inference
    sbi_demo = sbi_run(yml_file)
    sbi_demo.read_data_and_init_global_prior()

    # use existing results if wanted
    if run_from_pickle:
        sbi_demo.load_run_from_pickle_file()

    else:

        # restrict the priors
        sbi_demo.compute_restricted_prior()

        # sample from prior and simulate
        sbi_demo.generate_train_and_test_data()

        # run single round inference
        sbi_demo.run_sri()

        # save results 
        sbi_demo.save_run_in_pickle_file()
    

    # construct plots
    if make_plots:
        sbi_demo.plot_priors()
        sbi_demo.plot_prior_predictive_check()
        sbi_demo.plot_sri_spectrum()
        sbi_demo.plot_sri_posteriors()


    # calibration tests
    if run_calibration:
        sbi_demo.sbc_calibration()
        sbi_demo.coverage_zz_plot()



if __name__ == "__main__":
    main()
