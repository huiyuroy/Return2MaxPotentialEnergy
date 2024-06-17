import os
import random
import operator
import numpy as np
import pickle
from enum import Enum
import pygame.gfxdraw

from core import *

import core.agent as agent_base
import core.gain as gain_base
import core.rdwer as rdw_base
import core.resetter as reset_base
import core.input as walker_base

from core.space.scene import Scene, DiscreteScene
from core.space.boundary import Boundary
from core.space.grid import Tiling
from core.space.trajectory import Trajectory

from collections import deque
from distutils.util import strtobool
from util.file import load_json, save_json, save_grid_image

BOUND_ATTR_FORMAT = {
    "is_out_bound": bool,
    "points": list,
    "center": list,
    "barycenter": list,
    "cir_rect": list
}

NODE_ATTR_FORMAT = {
    "id": int,
    "pos": list,
    "loop_connect": list,
    "children_ids": list,
    "father_ids": list
}

TRI_ATTR_FORMAT = {
    "vertices": list,
    "barycenter": list,
    "in_circle": list,
    "out_edges": list,
    "in_edges": list
}
CONVEX_ATTR_FORMAT = {
    "vertices": list,
    "center": list,
    "barycenter": list,
    "cir_circle": list,
    "in_circle": list,
    "cir_rect": list,
    "out_edges": list,
    "in_edges": list
}

TILING_ATTR_FORMAT = {
    "id": int,
    "mat_loc": list,
    "center": list,
    "cross_bound": list,
    "rect": list,
    "h_width": float,
    "h_diag": float,
    "x_min": float,
    "x_max": float,
    "y_min": float,
    "y_max": float,
    "type": int
}

SCENE_ATTR_FORMAT = {
    "name": str,
    "bounds": list,
    "max_size": list,
    "out_bound_conv": dict,
    "out_conv_hull": dict,
    "scene_center": list,
    "tilings": list,
    "tilings_shape": list,
    "tiling_w": float,
    "tiling_x_offset": float,
    "tiling_y_offset": float,
    "tilings_data": list,
    "tilings_nei_ids": list,
    "tris": list,
    "tris_nei_ids": list,
    "conv_polys": list,
    "conv_nei_ids": list
}


def get_files(directory, extension):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.' + extension):
                filepath = os.path.join(root, filename)
                files.append(filepath)
    return files


def load_bound(bound_attr):
    bound = Boundary()
    bound.set_contour(bound_attr["is_out_bound"], bound_attr["points"])
    bound.center = np.array(bound_attr["center"])
    bound.barycenter = bound_attr["barycenter"]
    bound.cir_rect = bound_attr["cir_rect"]
    return bound


def load_tri(tri_attr):
    tri = geo.Triangle()
    tri.vertices = tuple(tri_attr["vertices"])
    tri.barycenter = tuple(tri_attr["barycenter"])
    tri.in_circle = tuple(tri_attr["in_circle"])
    tri.out_edges = tuple(tri_attr["out_edges"])
    tri.in_edges = tuple(tri_attr["in_edges"])
    return tri


def load_convex_poly(convex_attr):
    convex = geo.ConvexPoly()
    convex.vertices = tuple(convex_attr["vertices"])
    convex.center = np.array(convex_attr["center"])
    convex.barycenter = tuple(convex_attr["barycenter"])
    convex.cir_circle = tuple(convex_attr["cir_circle"])
    convex.in_circle = tuple(convex_attr["in_circle"])
    convex.out_edges = tuple(convex_attr["out_edges"])
    convex.in_edges = tuple(convex_attr["in_edges"])
    return convex


def load_tiling_base(scene, tiling_attr) -> Tiling:
    tiling = Tiling()
    tiling.corr_scene = scene
    tiling.id = tiling_attr["id"]
    tiling.mat_loc = tuple(tiling_attr["mat_loc"])
    tiling.cross_bound = np.array(tiling_attr["cross_bound"])
    tiling.corr_conv_cin = tiling_attr["corr_conv_cin"]
    tiling.corr_conv_inters = []
    for c in tuple(tiling_attr["corr_conv_inters"]):
        tiling.corr_conv_inters.append(np.array(c))
    tiling.corr_conv_inters = tuple(tiling.corr_conv_inters)
    tiling.corr_conv_ids = tuple(tiling_attr["corr_conv_ids"])
    tiling.type = tiling_attr["type"]
    tiling.rela_patch = tiling_attr["rela_patch"]
    return tiling


def load_tiling_visibility(tiling: Tiling, tiling_attr) -> Tiling:
    tiling.vis_tri = tuple(tiling_attr["visible_tri"])
    return tiling


def load_tiling_extend(tiling, tiling_attr) -> Tiling:
    tiling.flat_weight = tiling_attr['flat_weight']
    tiling.flat_grad = np.array(tiling_attr['flat_grad'])
    tiling.nearst_obs_grid_id = tiling_attr["nearst_obs_grid_id"]
    tiling.sur_grids_ids = tuple(tiling_attr['surround_grids_ids'])
    tiling.sur_obs_grids_ids = tuple(tiling_attr['surround_obs_grids_ids'])
    tiling.sur_obs_bound_grids_ids = tuple(tiling_attr['surround_obs_bound_grids_ids'])
    tiling.sur_360_partition = tuple(tiling_attr['surround_360_partition'])
    tiling.sur_occu_safe = bool(tiling_attr['surround_occupancy_safe'])
    return tiling


def load_contours(scene, data):
    scene.name = data['name']
    scene.bounds = []
    for bound_attr in data['bounds']:
        scene.bounds.append(load_bound(bound_attr))
    scene.max_size = data["max_size"]
    scene.out_bound_conv = load_convex_poly(data["out_bound_conv"])
    scene.out_conv_hull = load_convex_poly(data["out_conv_hull"])
    scene.scene_center = np.array(data["scene_center"])


def load_segmentation(scene, data):
    scene.tris = []
    for tri_attr in data["tris"]:
        scene.tris.append(load_tri(tri_attr))
    scene.tris_nei_ids = data["tris_nei_ids"]
    scene.conv_polys = []
    for conv_attr in data["conv_polys"]:
        scene.conv_polys.append(load_convex_poly(conv_attr))
    scene.conv_nei_ids = data["conv_nei_ids"]
    scene.conv_area_priority = data["conv_area_priority"]
    scene.conv_collision_priority = data["conv_collision_priority"]


def load_grids(scene, base_data, vis_data, extend_data):
    scene.tilings = [None] * len(base_data["tilings"])
    for tiling_attr in base_data["tilings"]:
        tiling = load_tiling_base(scene, tiling_attr)
        scene.tilings[tiling.id] = tiling

    for tiling_attr in vis_data["tilings"]:
        tiling = scene.tilings[tiling_attr["id"]]
        load_tiling_visibility(tiling, tiling_attr)

    for tiling_attr in extend_data["tilings"]:
        tiling = scene.tilings[tiling_attr["id"]]
        load_tiling_extend(tiling, tiling_attr)

    scene.tilings = tuple(scene.tilings)
    scene.tilings_shape = tuple(base_data["tilings_shape"])
    scene.tiling_w = base_data["tiling_w"]
    scene.tiling_w_inv = 1 / scene.tiling_w
    scene.tiling_offset = np.array(base_data["tiling_offset"])


def load_scene(tar_path, simple_load=False):
    scene = DiscreteScene()
    scene_dir, scene_name = os.path.split(tar_path)
    s_name = scene_name.split(".")[0]
    contour_data = load_json(tar_path)
    load_contours(scene, contour_data)
    if simple_load:
        return scene
    segment_data = load_json(scene_dir + '\\segment\\{}_seg.json'.format(s_name))
    load_segmentation(scene, segment_data)

    grid_base = load_json(scene_dir + '\\grid\\{}_base.json'.format(s_name))
    grid_vis = load_json(scene_dir + '\\grid\\{}_visibility.json'.format(s_name))
    grid_extend = load_json(scene_dir + '\\grid\\{}_extend.json'.format(s_name))
    load_grids(scene, grid_base, grid_vis, grid_extend)

    return scene


def load_scene_contour(tar_path):
    scene = DiscreteScene()
    scene_dir, scene_name = os.path.split(tar_path)
    s_name = scene_name.split(".")[0]
    contour_data = load_json(tar_path)
    load_contours(scene, contour_data)

    return scene


def load_trajectories(tar_path, scene_name):
    def load_trajectory(t_path):
        traj_data = load_json(t_path)
        traj_type = traj_data['type']
        traj_tars = traj_data['targets']
        return traj_type, traj_tars

    all_traj_files = get_files(tar_path + '\\simu_trajs\\{}'.format(scene_name), 'json')
    trajs = []

    for t_type, t_tars in list(map(load_trajectory, all_traj_files)):
        t = Trajectory()
        t.type = t_type
        t.tar_data = tuple(t_tars)
        t.tar_num = len(t_tars)
        t.end_idx = t.tar_num - 1
        trajs.append(t)
    return tuple(trajs)


def save_bound(bound):
    return {"is_out_bound": bound.is_out_bound,
            "points": np.array(np.around(bound.points, decimals=4), dtype='float').tolist(),
            "center": np.array(np.around(bound.center, decimals=4), dtype='float').tolist(),
            "barycenter": np.array(np.around(bound.barycenter, decimals=4), dtype='float').tolist(),
            "cir_rect": np.array(np.around(bound.cir_rect, decimals=4), dtype='float').tolist()}


def save_tri(tri):
    return {"vertices": np.array(np.around(tri.vertices, decimals=4), dtype='float').tolist(),
            "barycenter": np.array(np.around(tri.barycenter, decimals=4), dtype='float').tolist(),
            "in_circle": [np.array(np.around(tri.in_circle[0], decimals=4), dtype='float').tolist(),
                          float(np.around(tri.in_circle[1], decimals=4))],
            "out_edges": np.array(tri.out_edges).copy().tolist(),
            "in_edges": np.array(tri.in_edges).copy().tolist()}


def save_convex_poly(convex):
    return {"vertices": np.array(np.around(convex.vertices, decimals=4), dtype='float').tolist(),
            "center": np.array(np.around(convex.center, decimals=4), dtype='float').tolist(),
            "barycenter": np.array(np.around(convex.barycenter, decimals=4), dtype='float').tolist(),
            "cir_circle": [np.array(np.around(convex.cir_circle[0], decimals=4), dtype='float').tolist(),
                           float(np.around(convex.cir_circle[1], decimals=4))],
            "in_circle": [np.array(np.around(convex.in_circle[0], decimals=4), dtype='float').tolist(),
                          float(np.around(convex.in_circle[1], decimals=4))],
            "cir_rect": np.array(convex.cir_rect).copy().tolist(),
            "out_edges": np.array(convex.out_edges).copy().tolist(),
            "in_edges": np.array(convex.in_edges).copy().tolist()}


def save_tiling_base(tiling: Tiling):
    return {"id": tiling.id,
            "mat_loc": tiling.mat_loc,
            "type": tiling.type,
            "cross_bound": np.array(np.around(tiling.cross_bound, decimals=4), dtype='float').tolist(),
            "corr_conv_inters": [np.array(np.around(c, decimals=4), dtype='float').tolist() for c in
                                 tiling.corr_conv_inters],
            "corr_conv_ids": np.array(tiling.corr_conv_ids).tolist(),
            "corr_conv_cin": tiling.corr_conv_cin,
            "rela_patch": tiling.rela_patch}


def save_tiling_vis(tiling: Tiling):
    return {"id": tiling.id, "visible_tri": np.array(tiling.vis_tri).tolist()}


def save_tiling_extend(tiling):
    return {"id": tiling.id,
            'flat_weight': tiling.flat_weight,
            'flat_grad': np.array(np.around(tiling.flat_grad, decimals=6), dtype='float').tolist(),
            "nearst_obs_grid_id": tiling.nearst_obs_grid_id,
            'surround_grids_ids': tiling.sur_grids_ids,
            'surround_obs_grids_ids': tiling.sur_obs_grids_ids,
            'surround_obs_bound_grids_ids': tiling.sur_obs_bound_grids_ids,
            'surround_occupancy_safe': tiling.sur_occu_safe,
            'surround_360_partition': tiling.sur_360_partition
            }


def save_contours(scene):
    return {'name': scene.name,
            'bounds': list(map(save_bound, scene.bounds)),
            'max_size': np.array(np.around(scene.max_size, decimals=4), dtype='float').tolist(),
            "out_bound_conv": save_convex_poly(scene.out_bound_conv),
            "out_conv_hull": save_convex_poly(scene.out_conv_hull),
            "scene_center": np.array(np.around(scene.scene_center, decimals=4), dtype='float').tolist()}


def save_segmentation(scene):
    return {"tris": list(map(save_tri, scene.tris)),
            "tris_nei_ids": pickle.loads(pickle.dumps(scene.tris_nei_ids)),
            "conv_polys": list(map(save_convex_poly, scene.conv_polys)),
            "conv_nei_ids": pickle.loads(pickle.dumps(scene.conv_nei_ids)),
            "conv_area_priority": np.array(np.around(scene.conv_area_priority, decimals=6), dtype='float').tolist(),
            "conv_collision_priority": np.array(np.around(scene.conv_collision_priority, decimals=6),
                                                dtype='float').tolist()}


def save_grids(scene):
    base_info = {"tilings": tuple(map(save_tiling_base, scene.tilings)),
                 "tilings_shape": np.array(scene.tilings_shape).tolist(),
                 "tiling_w": scene.tiling_w,
                 "tiling_offset": scene.tiling_offset.tolist()}
    vis_info = {"tilings": tuple(map(save_tiling_vis, scene.tilings))}
    extend_info = {"tilings": tuple(map(save_tiling_extend, scene.tilings))}
    return base_info, vis_info, extend_info


def save_roadmap(scene):
    def save_node(n):
        return {'id': n.id,
                'pos': np.array(n.pos).tolist(),
                'loop_connect': np.array(n.rela_loop_id).tolist(),
                'children_ids': np.array(n.child_ids).tolist(),
                'father_ids': np.array(n.father_ids).tolist()}

    return {'name': scene.name, 'data': list(map(save_node, scene.nodes))}


def save_scene(scene, tar_path=None):
    print('saving {}'.format(scene.name))
    contour_data = save_contours(scene)
    segment_data = save_segmentation(scene)
    grid_base, grid_vis, grid_extend = save_grids(scene)
    road_attr = save_roadmap(scene)
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)
    save_json(contour_data, tar_path + '\\{}.json'.format(scene.name))
    save_json(segment_data, tar_path + '\\segment\\{}_seg.json'.format(scene.name))
    save_json(grid_base, tar_path + '\\grid\\{}_base.json'.format(scene.name))
    save_json(grid_vis, tar_path + '\\grid\\{}_visibility.json'.format(scene.name))
    save_json(grid_extend, tar_path + '\\grid\\{}_extend.json'.format(scene.name))
    save_json(road_attr, tar_path + '\\roadmap\\{}_rd.json'.format(scene.name))
    save_grid_image(scene.tilings_weights.copy() * 255, tar_path + '\\{}.bmp'.format(scene.name))


def load_trajectory_sample(f_path):
    if os.path.exists(f_path):
        fp = open(f_path, 'r+')
        line = fp.readline()
        traj_num = 0
        while line:
            data = line.split("#")
            code = data[1]
            if code == 'n':
                traj_num = int(data[2])
                break
        fp.close()
        return traj_num
    else:
        return None


def save_trajectory_file(f_path, tar_lists):
    fp = open(f_path, 'w')
    if len(tar_lists) == 0:
        print("no existed data")
        fp.close()
    else:
        fp.truncate(0)
        total_data_len = 0
        for tar_list in tar_lists:
            total_data_len += len(tar_list)
        fp.write("#n#{}\n".format(len(tar_lists)))
        item_id = 0
        traj_id = 0
        for cur_tar_list in tar_lists:
            fp.write("#s#{}#".format(traj_id) + "\n")
            for tar in cur_tar_list:
                x, y, z = tar
                fp.write("#t#{}#{}#{}#\n".format(round(x, 5), round(y, 5), round(z, 5)))
                item_id += 1
                print("\rsaving {}%".format(item_id / total_data_len * 100), end="")
            fp.write("#e#\n")
            traj_id += 1
        fp.close()
        print("\n", end="")


def obtain_agent(gainer='simple', rdwer='no', resetter='r21', inputer='traj', agent_manager='general'):
    """
    生成指定的重定向agent，可自行组合不同组件。注：部分特定类型agent必须设置特定组件。支持的重定向agent包括：

    1. No Rdw
    2. S2C
    3. S2O
    4. APF
    5. ARC
    6. APF-S2T
    7. ....
    Args:
        gainer:
        rdwer:
        resetter:
        inputer:
        agent_manager: 'general'-常规agent，无需特定设置，可与不需要特殊离线处理的增益控制器、重定向控制器和重置控制器组合；



    Returns:

    """

    tar_gain = None
    tar_reset = None
    tar_inputer = None
    tar_agent = None
    # -----------------设定增益管理器--------------------------------
    if 'simple' in gainer:
        tar_gain = gain_base.SimpleGainManager()
    elif 'arc' in gainer:
        tar_gain = gain_base.ARCGainManager()
    elif 'apfs2t' in gainer:
        tar_gain = gain_base.APFS2TGainManager()
    # -----------------设定重定向管理器--------------------------------
    if rdwer == 'no':
        tar_rdw = rdw_base.NoRdwManager()
    elif rdwer == 's2c':
        tar_rdw = rdw_base.S2CRdwManager()
        tar_rdw.steer_dampen[0] = 1.25
    elif rdwer == 's2o':
        tar_rdw = rdw_base.S2ORdwManager()
        tar_rdw.steer_dampen[0] = 1.25

    # -----------------设定重置管理器--------------------------------
    if 'r21' == resetter:
        tar_reset = reset_base.Turn21Resetter()
    elif 'r2g' == resetter:
        tar_reset = reset_base.TurnAPFGradientResetter()
    elif 'sfr2g' == resetter:
        tar_reset = reset_base.TurnAPFGradientStepForwardResetter()
    elif 'rarc' == resetter:
        tar_reset = reset_base.TurnArcResetter()
    elif 'tr2c' == resetter:  # traditional r2c
        tar_reset = reset_base.TurnCenterResetter()
    elif 'mr2c' == resetter:
        tar_reset = reset_base.TurnModifiedCenterResetter()
    elif 'r2t' == resetter:
        tar_reset = reset_base.TurnSteerForceResetter()

    elif 'r2mpe' == resetter:
        tar_reset = reset_base.TurnMaxProbEnergyResetter()

    # -----------------设定行走管理器--------------------------------
    if 'traj' in inputer:
        tar_inputer = walker_base.SimuTrajectoryInputer()
    elif 'live' in inputer:
        tar_inputer = walker_base.LiveInputer()

    if 'general' in agent_manager:
        tar_agent = agent_base.GeneralRdwAgent()

    tar_gain.load_params()
    tar_rdw.load_params()
    tar_reset.load_params()
    tar_inputer.load_params()

    tar_agent.set_manager(tar_gain, 'gain')
    tar_agent.set_manager(tar_rdw, 'rdw')
    tar_agent.set_manager(tar_reset, 'reset')
    tar_agent.set_manager(tar_inputer, 'inputer')

    return tar_agent
