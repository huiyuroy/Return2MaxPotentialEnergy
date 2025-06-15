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
    # 1.创建环境
    env = RdwEnv()
    # 2.添加agents，必须在env创建后调用，例子中加载了no+r21和no+r2mpe算法
    env.add_agent(default.obtain_agent(gainer='simple', rdwer='s2c', resetter='r2mpe', inputer='traj'),
                  name='r2mpe')

    env.load_logger()  # 加载日志记录器，可以自己设计
    env.load_ui()  # 加载ui界面，可以自己设计

    # 3.加载物理空间
    p_path = data_path + '\\phy'
    pname = 'test0'
    pscene = generator.load_scene(p_path + '\\' + pname + '.json')
    pscene.update_grids_runtime_attr()
    pscene.calc_r2mpe_precomputation()  # 如果采用R2mpe的任何组件，该函数必须调用以激活场景位置点可见区域离线计算

    v_path = data_path + '\\vir'
    v_files = [os.path.join(v_path, file) for file in os.listdir(v_path) if 'json' in file]

    for v_scene_path in v_files:
        # 4.加载虚拟空间,这里批量化加载，也可加载一个
        _, v_scene_name = os.path.split(v_scene_path)
        v_name = v_scene_name.split(".")[0]
        vscene = generator.load_scene(v_path + '\\' + v_name + '.json')
        vscene.update_grids_runtime_attr()
        vscene.calc_r2mpe_precomputation()  # 如果采用R2mpe的任何组件，该函数必须调用以激活场景位置点可见区域离线计算
        print(v_name)

        # 5.设置环境虚拟和真实空间
        env.set_scenes(vscene, pscene)  # 设置虚拟和物理空间，必须在prepare前调用

        # 6.加载虚拟空间模拟行走路径
        vtrajs = generator.load_trajectories(v_path, v_name)
        for vtraj in vtrajs:
            vtraj.range_targets(0, 500)  # 最长走编号0-500之间的目标，可自行调整
        env.set_trajectories(vtrajs)  # 设置模拟路径，仅在模拟重定向模式下调用

        # 7.env组件预处理，如果更换场景或者更换模拟路径后必须调用
        env.prepare()  # 加载预处理内容，例如场景可见性划分、离散占位网格处理等

        # 8.创建主循环
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
