from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.check_dir = '/models'
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/hk/tbsi/tracking/data/got10k_lmdb'
    settings.got10k_path = '/data/hk/tbsi/tracking/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/hk/tbsi/tracking/data/itb'
    settings.lasot_extension_subset_path_path = '/data/hk/tbsi/tracking/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/hk/tbsi/tracking/data/lasot_lmdb'
    settings.lasot_path = '/data/hk/TBSI-main/tracking/data/lasot'
    settings.lasher_path = '/home/gpu/A/datasets/LasHeR/test'
    settings.network_path = '/data/hk/tbsi/tracking/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/hk/TBSI-main/tracking/data/nfs'
    settings.otb_path = '/data/hk/TBSI-main/tracking/data/otb'
    settings.prj_dir = '/data/hk/TBSI-main/tracking'
    settings.result_plot_path = '/data/hk/tbsi/tracking/output/test/result_plots'
    settings.results_path = '/data/hk/tbsi-end/tracking/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = 'D:/.Downloadtbsi-endtracking\output/test/tracking_results\tbsi_track\vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_max_20'
    settings.segmentation_path = '/data/hk/tbsi/tracking/output/test/segmentation_results'
    settings.tc128_path = '/data/hk/TBSI-main/tracking/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/hk/TBSI-main/tracking/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/hk/TBSI-main/tracking/data/trackingnet'
    settings.uav_path = '/data/hk/TBSI-main/tracking/data/uav'
    settings.vot18_path = '/data/hk/TBSI-main/tracking/data/vot2018'
    settings.vot22_path = '/data/hk/TBSI-main/tracking/data/vot2022'
    settings.vot_path = '/data/hk/TBSI-main/tracking/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

