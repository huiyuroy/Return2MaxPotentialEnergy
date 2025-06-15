import datetime
import os

import numpy as np
import pyrdw.lib.math.geometry as geo
import pyrdw.generator as generator
import pyrdw.default as default
from common import data_path

from pyrdw.core.env.base import RdwEnv

if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find('Return2MaxPotentialEnergy') + len('Return2MaxPotentialEnergy')]
    # 1.create env
    env = RdwEnv()
    # 2.add agents
    env.add_agent(default.obtain_agent(gainer='simple', rdwer='s2c', resetter='r2mpe', inputer='traj'), name='r2mpe')
    env.load_logger()  # 加载日志记录器，可以自己设计
    env.load_ui()  # 加载ui界面，可以自己设计

    # 3.load phy space
    p_path = data_path + '\\phy'
    pname = 'test0'
    pscene = generator.load_scene(p_path + '\\' + pname + '.json')
    pscene.update_grids_runtime_attr()
    pscene.calc_r2mpe_precomputation()  # if use r2mpe, must call this precomputation method

    # 4. load vir space, can load one or a batch
    v_path = data_path + '\\vir'
    v_files = [os.path.join(v_path, file) for file in os.listdir(v_path) if 'json' in file]

    for v_scene_path in v_files:
        _, v_scene_name = os.path.split(v_scene_path)
        v_name = v_scene_name.split(".")[0]
        vscene = generator.load_scene(v_path + '\\' + v_name + '.json')
        vscene.update_grids_runtime_attr()
        vscene.calc_r2mpe_precomputation()  # if use r2mpe, must call this precomputation method
        print(v_name)

        # 5.setup vir and phy spaces for env
        env.set_scenes(vscene, pscene)

        # 6.load vir trajectories
        vtrajs = generator.load_trajectories(v_path, v_name)
        for vtraj in vtrajs:
            vtraj.range_targets(0, 500)
        env.set_trajectories(vtrajs)  # setup trajectory for an env

        # 7.env prepare
        env.prepare()

        # 8. main loop
        for traj_idx, traj in enumerate(vtrajs):
            if traj_idx % 10 != 0:
                continue

            env.set_current_trajectory(traj)
            max_area_conv = None
            max_area = 0
            for conv in env.p_scene.conv_polys:
                conv_area = geo.calc_poly_area(np.array(conv.vertices))
                if conv_area > max_area:
                    max_area = conv_area
                    max_area_conv = conv
            init_p_loc = max_area_conv.center
            env.init_agents_state(p_loc=init_p_loc, p_fwd=[0, 1], v_loc=[0, 1], v_fwd=[0, 1])
            # 9. main thread
            env.reset()
            while True:
                d = env.step()
                env.render()
                env.record()
                if d:
                    all_data = env.output_epi_info()
                    d = datetime.datetime.now()
                    print("{}:{}:{}:{}  ".format(d.day, d.hour, d.minute, d.second), end="")
                    break
