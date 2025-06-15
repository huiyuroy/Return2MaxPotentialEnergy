from pyrdw.core.resetter import *


class TurnMaxProbEnergyResetter(BaseResetter):
    """
    基本思想：
      - 假设前提：人类的移动是基于短期目标的，是理性的，因此人类趋向于兴趣点移动；即使在无任何参考物的广阔虚拟空间，人每一步移动的预期在很大程度
                上趋向于保持原方向不变，转角越大，概率更低。因此这很符合高斯概率模型。
      - 使用前必须为物理空间调用calc_r2mpe_vis_range_simplify()以进行必要的场景预计算

    """

    def __init__(self):
        super().__init__()

        self.reset_type = 'T-2MPE'
        self.init_fwd = [0, 1]
        self.fov = 360
        self.h_fov = 0.5 * self.fov
        self.fov_rad = self.fov * DEG2RAD
        self.h_fov_rad = 0.5 * self.fov_rad
        self.human_step = HUMAN_STEP * 0.01
        self.rev_human_step = REV_HUMAN_STEP * 100
        self.p_scene_rot_occupancy_deg = 0
        self.p_tid_map_mat = None
        self.reset_angle_range = None
        self.reset_pot_fwds = None
        self.reset_pot_fwds_num = 0
        self.v_vis_tri = None
        self.v_vis_poly = None
        self.p_vis_grids_ids = None
        self.alpha = 0.2
        self.rot_energy_base = np.exp(-4.5 * REV_PI ** 2)
        self.mov_energy_base = np.exp(-0.5 * self.rev_human_step ** 2)  # cm 2 m

        self.v_weights = None
        self.p_weights = None

        self.v_tiling_visible = None
        self.p_tiling_visited = None
        self.v_tiling_visited = None
        self.p_data_visited = None
        self.v_data_visited = None
        self.p_sur_vecs = None
        self.p_sur_vel_engs = None
        self.p_data_rela = None

        self.tar_v_ids = None
        self.tar_p_ids = None
        self.tar_v_engs = None
        self.tar_p_engs = None
        self.tar_min_prob = 0
        self.tar_max_prob = 0

    def para_setup(self, **kwargs):
        for key in kwargs.keys():
            if key == 'rot_eng_dis':
                if kwargs[key]:
                    self.rot_energy_base = PI_2 ** 0.5
            elif key == 'mov_eng_dis':
                if kwargs[key]:
                    self.mov_energy_base = PI_2 ** 0.5
            elif key == 'alpha':
                self.alpha = kwargs[key]

    def prepare(self):
        self.v_weights = tuple([t.flat_weight * REV_PI_2 for t in self.v_scene.tilings])
        self.p_weights = tuple([t.flat_weight * REV_PI_2 for t in self.p_scene.tilings])

    def reset(self):
        super().reset()
        self.reset_angle_range = tuple(range(-180, 180, 2))
        self.reset_pot_fwds = tuple([geo.rot_vecs(self.init_fwd, ang * DEG2RAD) for ang in self.reset_angle_range])
        self.reset_pot_fwds_num = len(self.reset_pot_fwds)
        self.v_tiling_visited = np.zeros(len(self.v_scene.tilings))
        self.p_tiling_visited = np.zeros(len(self.p_scene.tilings))
        self.p_sur_vecs = [None] * len(self.p_scene.tilings)
        self.p_sur_vel_engs = [None] * len(self.p_scene.tilings)
        self.p_tid_map_mat = -np.ones((len(self.p_scene.tilings), len(self.p_scene.tilings))).astype(np.int32)
        for pt in self.p_scene.tilings:
            for i in range(len(pt.vis_grids_ids)):
                self.p_tid_map_mat[pt.id, pt.vis_grids_ids[i]] = i
        self.p_scene_rot_occupancy_deg = 360 / self.p_scene.tilings_rot_occupancy_num

    def calc_reset_target_fwd(self):
        """
        记录虚拟空间每个tiling在找旋转方向最优解过程中是否被访问过，如果访问过，则其相应的单点能量一定计算过，否则就更新一下该点的能量
        """
        v_vis_id = self.agent.v_cur_tiling.obtain_vis_attr_id(self.v_loc)
        p_vis_id = self.agent.p_cur_tiling.obtain_vis_attr_id(self.p_loc)

        self.v_vis_tri = self.agent.v_cur_tiling.vis_tri[v_vis_id].copy()
        self.v_vis_poly = self.agent.v_cur_tiling.vis_poly[v_vis_id]
        self.p_vis_grids_ids = self.agent.p_cur_tiling.vis_grids_ids[p_vis_id]

        self.v_tiling_visible = [None] * len(self.v_scene.tilings)
        self.v_data_visited = self.v_tiling_visited.copy()
        self.p_data_visited = self.p_tiling_visited.copy()
        self.p_data_rela = self.p_scene.tilings_data[self.p_vis_grids_ids] - self.p_loc

        if len(self.agent.v_cur_tiling.sur_obs_gids) > 0:
            self.agent.v_cur_tiling.sur_obs_bound_tiling_attr = []
            for obs_id in self.agent.v_cur_tiling.sur_bound_gids:
                obs_tiling = self.v_scene.tilings[obs_id]
                obs_vec = (self.v_loc - obs_tiling.center) * 0.01
                obs_d = alg.l2_norm(obs_vec)
                obs_d_rev = 1 / obs_d
                obs_d_rev_2 = obs_d_rev * 0.5
                obs_superscript = self.human_step * obs_d_rev
                self.agent.v_cur_tiling.sur_obs_bound_tiling_attr.append((obs_vec, obs_d_rev_2, obs_superscript))

        v_off = self.v_loc - self.v_scene.tiling_offset
        step = 3
        find_optimal = False
        rough_opt = True
        angle_range = tuple(np.linspace(-180, 179, 120))
        angle_idx = 0
        max_s_eng = -float('inf')
        theta = 0
        fwd = None
        eng_dist = None
        while not find_optimal:
            p_theta = angle_range[angle_idx]
            p_fwd = geo.rot_vecs(self.init_fwd, p_theta * DEG2RAD)
            nx_pot_loc = self.p_loc + p_fwd * self.reset_pred_t
            if self.p_scene.poly_contour_safe.covers(Point(nx_pot_loc)):
                p_eng, p_eng_dist = self.__calc_dir_energy(p_fwd, v_off)
                if p_eng > max_s_eng:
                    max_s_eng = p_eng
                    theta = p_theta
                    fwd = p_fwd
                    eng_dist = p_eng_dist
            if angle_idx < len(angle_range) - 1:
                angle_idx += 1
            else:
                if rough_opt:
                    rough_opt = False
                    angle_range = tuple((np.linspace(-step, step, 2 * step) + theta + 180) % 360 - 180)
                    angle_idx = 0
                else:
                    find_optimal = True

        self.tar_v_ids, self.tar_p_ids, self.tar_v_engs, self.tar_p_engs = eng_dist
        self.reset_target_fwd = fwd

    def __calc_dir_energy(self, pot_fwd, v_off):
        # 按照论文的方法
        v_w, v_h = self.v_scene.tilings_shape
        inter_p, inter_dis = geo.calc_ray_poly_intersection(self.p_loc, pot_fwd, self.p_scene.poly_contour)
        if inter_dis <= self.reset_trigger_t:
            return 0, None
        ## 计算出当前转角前提下，虚实空间相交区域，并进一步计算出虚拟空间相交区域的总能量
        cross_v_energy = 0
        cross_p_energy = 0
        cross_vt_ids = []
        cross_pt_ids = []
        cross_vt_engs = []
        cross_pt_engs = []
        vp_theta_dev = geo.calc_angle_bet_vec(pot_fwd, self.v_fwd)
        # v_col,v_row
        mapped_p_rc = ((geo.rot_vecs(self.p_data_rela, vp_theta_dev) + v_off) // self.v_scene.tiling_w).astype(np.int32)
        for m_id,p_id in enumerate(self.p_vis_grids_ids):
            cross_p_tiling = self.p_scene.tilings[p_id]
            col, row = mapped_p_rc[m_id]
            cross_pt_ids.append(p_id)
            # 如果将运动预测模型加入实际空间能量计算，则开启下面被注释的代码
            if self.p_data_visited[p_id]:
                sur_vec = self.p_sur_vecs[p_id]
                sur_vel_energy = self.p_sur_vel_engs[p_id]
            else:
                self.p_data_visited[p_id] = 1
                sur_vec = cross_p_tiling.center - self.p_loc
                sur_vel_energy = self.p_weights[p_id] * self.mov_energy_base ** alg.l2_norm_square(sur_vec * 0.01)
                self.p_sur_vecs[p_id] = sur_vec
                self.p_sur_vel_engs[p_id] = sur_vel_energy
            theta = geo.calc_angle_bet_vec(sur_vec, pot_fwd)
            p_energy = sur_vel_energy * self.rot_energy_base ** (theta ** 2)
            cross_pt_engs.append(p_energy)
            cross_p_energy += p_energy

            if 0 <= row < v_h and 0 <= col < v_w:
                v_id = row * v_w + col
                cross_v_tiling = self.v_scene.tilings[v_id]  # 物理空间某个tiling在虚拟空间对应的tiling
                if cross_v_tiling.type:
                    if self.v_tiling_visible[v_id] is not None:
                        in_vis_tri = self.v_tiling_visible[v_id]
                    else:
                        in_vis_tri = False
                        if self.v_vis_poly.covers(Point(cross_v_tiling.center)):
                            in_vis_tri = True
                        self.v_tiling_visible[v_id] = in_vis_tri
                    if in_vis_tri:
                        cross_vt_ids.append(v_id)
                        if self.v_data_visited[v_id]:
                            v_energy = cross_v_tiling.prob_energy
                        else:
                            self.v_data_visited[v_id] = 1
                            sur_vec = cross_v_tiling.center - self.v_loc
                            sur_vel = alg.l2_norm(sur_vec)
                            theta = geo.calc_angle_bet_vec(sur_vec, self.v_fwd)
                            v_energy = self.v_weights[v_id] * self.mov_energy_base ** alg.l2_norm_square(
                                sur_vec * 0.01)
                            v_energy *= self.rot_energy_base ** (theta ** 2)
                            obs_coff = 1
                            if sur_vel > 0 and len(self.agent.v_cur_tiling.sur_obs_gids) > 0:
                                norm_vec = sur_vec / sur_vel
                                obs_coff = float('inf')
                                for o_v, o_d_rev_2, o_sup in self.agent.v_cur_tiling.sur_obs_bound_tiling_attr:
                                    epsilon = (0.5 + np.dot(norm_vec, o_v) * o_d_rev_2) ** o_sup
                                    if epsilon < obs_coff:
                                        obs_coff = epsilon
                            v_energy *= obs_coff
                            cross_v_tiling.prob_energy = v_energy
                        cross_vt_engs.append(v_energy)
                        cross_v_energy += v_energy

        cross_scene_energy = (1 - self.alpha) * cross_p_energy + self.alpha * cross_v_energy
        inter_dis *= 0.01
        p1, p2 = np.exp(inter_dis), np.exp(-inter_dis)
        phy_fwd_attenuation_coff = (p1 - p2) / (p1 + p2)
        return phy_fwd_attenuation_coff * cross_scene_energy, (cross_vt_ids, cross_pt_ids, cross_vt_engs, cross_pt_engs)

    def render(self, wdn_obj, default_color):
        super().render(wdn_obj, default_color)
        if not self.enable_draw:
            return
        if self.reset_state:
            # for tri in self.agent.v_cur_tiling.vis_tri:
            #     wdn_obj.draw_vir_poly(tri, fill=False, color=(125, 125, 125, 125))
            # for tri in self.agent.p_cur_tiling.vis_tri:
            #     wdn_obj.draw_phy_poly(tri, fill=False, color=(125, 125, 125, 125))
            min_p, max_p = float('inf'), 0
            for i in range(len(self.tar_v_engs)):
                pvalue = self.tar_v_engs[i]
                if pvalue < min_p:
                    min_p = pvalue
                if pvalue > max_p:
                    max_p = pvalue
            for i in range(len(self.tar_v_ids)):
                sur_t = self.v_scene.tilings[self.tar_v_ids[i]]
                pvalue = self.tar_v_engs[i]
                co = (pvalue - min_p) / (max_p - min_p)
                co = (int(255 * co), 0, 255)
                wdn_obj.draw_vir_circle(sur_t.center, 2, co)
            wdn_obj.draw_vir_line(self.v_loc, geo.norm_vec(self.reset_start_vir_fwd) * 100 + self.v_loc, 2, (0, 255, 0))

            min_p, max_p = float('inf'), 0
            for i in range(len(self.tar_p_engs)):
                pvalue = self.tar_p_engs[i]
                if pvalue < min_p:
                    min_p = pvalue
                if pvalue > max_p:
                    max_p = pvalue

            for i in range(len(self.tar_p_ids)):
                sur_t = self.p_scene.tilings[self.tar_p_ids[i]]
                pvalue = self.tar_p_engs[i]
                co = (pvalue - min_p) / (max_p - min_p)
                co = (int(255 * co), 0, 255)
                wdn_obj.draw_phy_circle(sur_t.center, 5, co)

