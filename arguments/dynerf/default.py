ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4,8],
    defor_depth = 0,
    net_width = 256,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=True,
    static_mlp=False

)
OptimizationParams = dict(
    dataloader=False,
    iterations = 50000, #14000
    batch_size=1 ,#4, changed due to batch size
    coarse_iterations = 3200, #250, #2500, #5000 #3000, #250 used for avoiding nan 3_agents
    densify_until_iter = 10_000,
    opacity_reset_interval = 60000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
)