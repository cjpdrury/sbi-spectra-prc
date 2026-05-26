from prc_class import sbi_run


def main():
    
    # initial params
    yml_file = "SISXA_YML_INPUT_FILES/" + "1_sri_config_cmin_cmax_restrictor_spectrum_20000.yml"
    

    # setup inference
    sbi_demo = sbi_run(yml_file)
    sbi_demo.read_data_and_init_global_prior()

    # plot prior
    # sbi_demo.plot_prior()
    
    # 



if __name__ == "__main__":
    main()
