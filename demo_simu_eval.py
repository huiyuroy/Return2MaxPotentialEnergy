import os
import random

import numpy as np
import lib.math.geometry as geo
import generator as generator
from core.env import RdwEnv

if __name__ == '__main__':

    # 1. create environment
    env = RdwEnv()
    # 2. agent must be added after the environment (env) is created, we load no rdw + r2mpe in the example
    env.add_agent(generator.obtain_agent(gainer='simple', rdwer='no', resetter='r2mpe', inputer='traj'),
                  agent_name='r2mpe')

    env.load_logger()  # load the logger, allow the custom design
    env.load_ui()  # load simulation ui, allow the custom design

    # 3. load physical space from files
    p_path = 'E:\\polyspaces\\phy'
    pname = 'test3'
    pscene = generator.load_scene(p_path + '\\' + pname + '.json')
    # pscene = generator.load_scene_contour(p_path + '\\' + pname + '.json')
    # pscene.update_grids_base_attr()
    # pscene.update_grids_visibility()
    # pscene.update_grids_weights()
    # pscene.update_grids_rot_occupancy(True)
    pscene.update_grids_runtime_attr()
    # activate the offline computation of the visible area, must be called if r2mpe is used
    pscene.calc_r2mpe_precomputation()
    # init physical start position
    init_p_loc = None
    while init_p_loc is None:
        x = random.randint(0, pscene.max_size[0])
        y = random.randint(0, pscene.max_size[1])
        if geo.chk_p_in_bound([x, y], pscene.bounds):
            init_p_loc = [x, y]

    v_path = 'E:\\polyspaces\\vir'
    v_files = [os.path.join(v_path, file) for file in os.listdir(v_path) if 'json' in file]

    # 4. batch loading virtual spaces
    for v_scene_path in v_files:
        _, v_scene_name = os.path.split(v_scene_path)
        v_name = v_scene_name.split(".")[0]
        v_path = 'E:\\polyspaces\\vir'
        vscene = generator.load_scene(v_path + '\\' + v_name + '.json')
        # vscene = generator.load_scene_contour(v_path + '\\' + v_name + '.json')
        # vscene.update_grids_base_attr()
        # vscene.update_grids_visibility()
        # vscene.update_grids_weights()
        # vscene.update_grids_rot_occupancy(True)
        vscene.update_grids_runtime_attr()

        # 5.set virtual and physical spaces of environment, must be called before env.prepare
        env.set_scenes(vscene, pscene)

        # 6. load simulated walking paths
        vtrajs = generator.load_trajectories('E:\\polyspaces\\vir', v_name)
        for vtraj in vtrajs:
            vtraj.range_targets(0, 500)  # set the walking targets, total number can be modified
        env.set_trajectories(vtrajs)

        # 7. preprocess the env component, must be called after changing the scene or simulation path.
        env.prepare()
        env.env_ui.enable_render()
        # 8. select trajectory
        for traj_idx, traj in enumerate(vtrajs):
            if traj_idx % 10 != 0:
                continue

            env.set_current_trajectory(traj)
            env.init_agents_state(p_loc=init_p_loc, p_fwd=[0, 1], v_loc=[0, 1], v_fwd=[0, 1])
            env.reset()

            while True:
                d = env.step()
                env.record()
                if d:
                    env.output_epi_info()
                    break
