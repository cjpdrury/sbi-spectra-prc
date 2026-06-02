from prc_class import sbi_run


def main():
    
    # initial params
    yml_file = "SISXA_YML_INPUT_FILES/" + "1_sri_config_cmin_cmax_restrictor_spectrum_20000.yml"
    make_plots = True
    

    # setup inference
    sbi_demo = sbi_run(yml_file)
    sbi_demo.read_data_and_init_global_prior()

    # sample from prior and simulate
    sbi_demo.generate_train_and_test_data()

    # run single round inference
    sbi_demo.run_sri()
    

    # construct plots
    if make_plots:
        sbi_demo.plot_prior()
        sbi_demo.plot_prior_predictive_check()
        sbi_demo.plot_sri_spectrum()
        sbi_demo.plot_sri_posteriors()


if __name__ == "__main__":
    main()
