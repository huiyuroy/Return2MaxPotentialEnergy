from core import *


class BaseResetter(BaseManager):

    def __init__(self):
        super().__init__()
        self.reset_spec = const_reset
        self.reset_type = 'base'
        self.reset_pre_state = 0
        self.reset_state = 0
        self.reset_angle = 0
        self.reset_scale = 1
        # The minimum distance to the boundary before triggering a reset. (in cm)
        self.reset_trigger_t = 20
        # The minimum angular bias to end the reset state. When the current physical direction is within this value
        # from the target reset direction, the reset process will be terminated (in degree)
        self.reset_terminal_t = 1
        self.reset_pred_t = 20
        self.reset_target_fwd = None
        self.reset_rest_angle = PI
        self.reset_start_vir_fwd = [0, 1]
        self.p_loc, self.p_fwd = np.array([0, 0]), np.array([0, 0])
        self.v_loc, self.v_fwd = np.array([0, 0]), np.array([0, 0])
        self.p_vel, self.p_rot = 0, 0
        self.reset_num = 0
        self.enable_draw = True

    def load_params(self):
        self.reset_trigger_t = self.reset_spec['reset_trigger_dis']
        self.reset_terminal_t = self.reset_spec['reset_finish_ang'] * DEG2RAD
        self.reset_pred_t = self.reset_spec['reset_pred_dis'] * 2

    def prepare(self):
        self.p_scene, self.v_scene = self.agent.p_scene, self.agent.v_scene

    def reset(self):
        self.reset_num = 0

    def is_resetting(self):
        return self.reset_state

    def enable(self):
        self.reset_state = 1

    def disable(self):
        self.reset_state = 0

    def record(self):
        self.reset_pre_state = self.reset_state

    def trigger(self):
        if self.reset_state:
            self.reset_num += 1
            return 1
        else:
            tiling = self.agent.p_cur_tiling
            if tiling.type and (tiling.sur_occu_safe or
                                geo.chk_p_in_bound(self.p_loc, self.p_scene.bounds, self.reset_trigger_t)):
                self.reset_state = 0
            else:
                nx_p_loc = self.p_loc + geo.norm_vec(self.p_fwd) * self.reset_pred_t
                if not geo.chk_p_in_bound(nx_p_loc, self.p_scene.bounds, self.reset_trigger_t) and self.p_vel > 0:
                    self.reset_num += 1
                    self.reset_state = 1
                else:
                    self.reset_state = 0
            return self.reset_state

    def start(self, **kwargs):
        self.calc_reset_target_fwd()
        self.reset_angle = geo.calc_angle_bet_vec(self.p_fwd, self.reset_target_fwd)
        if self.reset_angle == 0:
            self.reset_state = 0
            return
        self.reset_start_vir_fwd = self.v_fwd.copy()
        self.reset_rest_angle = self.reset_angle
        self.reset_scale = PI_2 / self.reset_angle

    @abstractmethod
    def calc_reset_target_fwd(self):
        """
        Used to calculate the reset target direction when initiating a reset. Needs to override this method when
        customizing a new Reset strategy.

        Returns:
            None
         """
        raise NotImplementedError

    def update(self, **kwargs):
        self.reset_rest_angle = geo.calc_angle_bet_vec(self.p_fwd, self.reset_target_fwd)
        head_fwd = geo.norm_vec(geo.rot_vecs(self.reset_start_vir_fwd, self.reset_rest_angle * self.reset_scale))
        if abs(self.reset_rest_angle) < self.reset_terminal_t:
            self.reset_state = 0
            move_fwd = head_fwd * self.p_vel
            self.reset_target_fwd = None
        else:
            self.reset_state = 1
            move_fwd = [0, 0]
        self.agent.v_cur_loc = self.v_loc + move_fwd
        self.agent.v_cur_fwd = head_fwd

    def render(self, wdn_obj, default_color):
        if not self.enable_draw:
            return
        if self.reset_target_fwd is not None:
            wdn_obj.draw_phy_line(self.p_loc, geo.norm_vec(self.reset_target_fwd) * 100 + self.p_loc, 2, (0, 255, 0))

    def copy_target_manager(self, other_mg):
        if other_mg is not None:
            self.reset_trigger_t = other_mg.reset_trigger_t
            self.reset_terminal_t = other_mg.reset_terminal_t
        else:
            self.load_params()


class Turn21Resetter(BaseResetter):
    """
    Basic Return 2:1 Reset. Proposed by https://dl.acm.org/doi/abs/10.1145/1272582.1272590

    """

    def __init__(self):
        super().__init__()
        self.reset_type = 'Turn21'

    def calc_reset_target_fwd(self):
        """
        Target reset direction is calculated by reversing current direction.

        Returns:
            None
        """
        self.reset_target_fwd = geo.rot_vecs(self.p_fwd, PI)


class TurnCenterResetter(BaseResetter):

    def __init__(self):
        super().__init__()
        self.reset_type = 'T2C'
        self.p_center = None

    def reset(self):
        super().reset()
        self.p_center = self.p_scene.bounds[0].center.copy()

    def calc_reset_target_fwd(self):
        self.reset_target_fwd = self.p_center - self.p_loc


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
            for i in range(len(pt.r2mpe_vis_occu_ids)):
                self.p_tid_map_mat[pt.id, pt.r2mpe_vis_occu_ids[i]] = i
        self.p_scene_rot_occupancy_deg = 360 / self.p_scene.tilings_rot_occupancy_num

    def calc_reset_target_fwd(self):
        """
        记录虚拟空间每个tiling在找旋转方向最优解过程中是否被访问过，如果访问过，则其相应的单点能量一定计算过，否则就更新一下该点的能量
        """
        self.v_vis_tri = tuple(np.array(tri) for tri in self.agent.v_cur_tiling.vis_tri)
        self.v_tiling_visible = [None] * len(self.v_scene.tilings)
        self.v_data_visited = self.v_tiling_visited.copy()
        self.p_data_visited = self.p_tiling_visited.copy()
        self.p_data_rela = self.p_scene.tilings_data[self.agent.p_cur_tiling.r2mpe_vis_occu_ids] - self.p_loc

        if len(self.agent.v_cur_tiling.sur_obs_grids_ids) > 0:
            self.agent.v_cur_tiling.sur_obs_bound_tiling_attr = []
            for obs_id in self.agent.v_cur_tiling.sur_obs_bound_grids_ids:
                obs_tiling = self.v_scene.tilings[obs_id]
                obs_vec = (self.v_loc - obs_tiling.center) * 0.01
                obs_d = alg.l2_norm(obs_vec)
                obs_d_rev = 1 / obs_d
                obs_d_rev_2 = obs_d_rev * 0.5
                obs_superscript = self.human_step * obs_d_rev
                self.agent.v_cur_tiling.sur_obs_bound_tiling_attr.append((obs_vec, obs_d_rev_2, obs_superscript))

        v_off = self.v_loc - self.v_scene.tiling_offset
        step = 3
        com_num = 0
        pot_thetas = tuple(range(-180, 181, step))  # 120等分
        max_s_eng = -float('inf')
        theta = 0
        fwd = None
        eng_dist = None
        for p_theta in pot_thetas:
            p_fwd = geo.rot_vecs(self.init_fwd, p_theta * DEG2RAD)
            nx_pot_loc = self.p_loc + p_fwd * self.reset_pred_t
            if geo.chk_p_in_bound(nx_pot_loc, self.p_scene.bounds, self.reset_trigger_t):
                p_eng, p_eng_dist = self.__calc_dir_energy(p_theta, p_fwd, v_off)
                com_num += 1
                if p_eng > max_s_eng:
                    max_s_eng = p_eng
                    theta = p_theta
                    fwd = p_fwd
                    eng_dist = p_eng_dist

        init_theta = theta
        theta_rag = tuple(range(- step, step + 1, 1))
        for t_off in theta_rag:
            p_theta = init_theta + t_off
            if p_theta > 180:
                p_theta -= 360
            if p_theta < -180:
                p_theta += 360
            p_fwd = geo.rot_vecs(self.init_fwd, p_theta * DEG2RAD)
            p_eng, p_eng_dist = self.__calc_dir_energy(p_theta, p_fwd, v_off)
            com_num += 1
            if p_eng > max_s_eng:
                max_s_eng = p_eng
                fwd = p_fwd
                eng_dist = p_eng_dist

        self.tar_v_ids, self.tar_p_ids, self.tar_v_engs, self.tar_p_engs = eng_dist
        self.reset_target_fwd = fwd

    def __calc_dir_energy(self, reset_theta, pot_fwd, v_off):
        # 按照论文的方法
        v_w, v_h = self.v_scene.tilings_shape
        inter_p, inter_dis, _ = geo.calc_ray_bound_intersection(self.p_loc, pot_fwd, self.p_scene.bounds)
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
        t_reset_theta = reset_theta if reset_theta < 180 else reset_theta + 360
        central_grid_idx = int(t_reset_theta // self.p_scene_rot_occupancy_deg)
        h_fov_occu_num = int(self.h_fov // self.p_scene_rot_occupancy_deg)

        for o_i in range(-h_fov_occu_num, h_fov_occu_num + 1):
            occu_id = (central_grid_idx + o_i) % self.p_scene.tilings_rot_occupancy_num
            occu_grid = self.agent.p_cur_tiling.r2mpe_360_partition[occu_id]
            for p_id in occu_grid:
                cross_p_tiling = self.p_scene.tilings[p_id]
                col, row = mapped_p_rc[self.p_tid_map_mat[self.agent.p_cur_tiling.id][p_id]]
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
                            crs_t_dir = cross_v_tiling.center - self.agent.v_cur_tiling.center
                            for tri_id in self.agent.v_cur_tiling.vis_360angle_partition[
                                math.floor((geo.calc_angle_bet_vec(crs_t_dir, (0, 1)) * RAD2DEG + 180) % 360)]:
                                if geo.chk_p_in_conv_simple(cross_v_tiling.center, self.v_vis_tri[tri_id]):
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
                                if sur_vel > 0 and len(self.agent.v_cur_tiling.sur_obs_grids_ids) > 0:
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

    # def __calc_dir_energy(self, reset_theta, pot_fwd, v_off):
    #     # 按照论文的方法
    #     v_w, v_h = self.v_scene.tilings_shape
    #     inter_p, inter_dis, _ = geo.calc_ray_bound_intersection(self.p_loc, pot_fwd, self.p_scene.bounds)
    #     if inter_dis <= self.reset_trigger_t:
    #         return 0, None
    #     ## 计算出当前转角前提下，虚实空间相交区域，并进一步计算出虚拟空间相交区域的总能量
    #     cross_v_energy = 0
    #     cross_p_energy = 0
    #     cross_vt_ids = []
    #     cross_pt_ids = []
    #     cross_vt_engs = []
    #     cross_pt_engs = []
    #     vp_theta_dev = geo.calc_angle_bet_vec(pot_fwd, self.v_fwd)
    #     # v_col,v_row
    #     mapped_p_rc = ((geo.rot_vecs(self.p_data_rela, vp_theta_dev) + v_off) // self.v_scene.tiling_w).astype(np.int32)
    #     t_reset_theta = reset_theta if reset_theta < 180 else reset_theta + 360
    #     central_grid_idx = int(t_reset_theta // self.p_scene_rot_occupancy_deg)
    #     h_fov_occu_num = int(self.h_fov // self.p_scene_rot_occupancy_deg)
    #
    #     for o_i in range(-h_fov_occu_num, h_fov_occu_num + 1):
    #         occu_id = (central_grid_idx + o_i) % self.p_scene.tilings_rot_occupancy_num
    #         occu_grid = self.agent.p_cur_tiling.r2mpe_360_partition[occu_id]
    #         for p_id in occu_grid:
    #             cross_p_tiling = self.p_scene.tilings[p_id]
    #             col, row = mapped_p_rc[self.p_tid_map_mat[self.agent.p_cur_tiling.id][p_id]]
    #             if 0 <= row < v_h and 0 <= col < v_w:
    #                 v_id = row * v_w + col
    #                 cross_v_tiling = self.v_scene.tilings[v_id]  # 物理空间某个tiling在虚拟空间对应的tiling
    #                 if cross_v_tiling.type:
    #                     if self.v_tiling_visible[v_id] is not None:
    #                         in_vis_tri = self.v_tiling_visible[v_id]
    #                     else:
    #                         in_vis_tri = False
    #                         crs_t_dir = cross_v_tiling.center - self.agent.v_cur_tiling.center
    #                         for tri_id in self.agent.v_cur_tiling.vis_360angle_partition[
    #                             math.floor((geo.calc_angle_bet_vec(crs_t_dir, (0, 1)) * RAD2DEG + 180) % 360)]:
    #                             if geo.chk_p_in_conv_simple(cross_v_tiling.center, self.v_vis_tri[tri_id]):
    #                                 in_vis_tri = True
    #                         self.v_tiling_visible[v_id] = in_vis_tri
    #                     if in_vis_tri:
    #                         cross_pt_ids.append(p_id)
    #                         cross_vt_ids.append(v_id)
    #                         # 如果将运动预测模型加入实际空间能量计算，则开启下面被注释的代码
    #                         if self.p_data_visited[p_id]:
    #                             sur_vec = self.p_sur_vecs[p_id]
    #                             sur_vel_energy = self.p_sur_vel_engs[p_id]
    #                         else:
    #                             self.p_data_visited[p_id] = 1
    #                             sur_vec = cross_p_tiling.center - self.p_loc
    #                             sur_vel_energy = self.p_weights[p_id] * self.mov_energy_base ** alg.l2_norm_square(
    #                                 sur_vec * 0.01)
    #                             self.p_sur_vecs[p_id] = sur_vec
    #                             self.p_sur_vel_engs[p_id] = sur_vel_energy
    #                         theta = geo.calc_angle_bet_vec(sur_vec, pot_fwd)
    #                         p_energy = sur_vel_energy * self.rot_energy_base ** (theta ** 2)
    #                         cross_pt_engs.append(p_energy)
    #                         cross_p_energy += p_energy
    #
    #                         if self.v_data_visited[v_id]:
    #                             v_energy = cross_v_tiling.prob_energy
    #                         else:
    #                             self.v_data_visited[v_id] = 1
    #                             sur_vec = cross_v_tiling.center - self.v_loc
    #                             sur_vel = alg.l2_norm(sur_vec)
    #                             theta = geo.calc_angle_bet_vec(sur_vec, self.v_fwd)
    #                             v_energy = self.v_weights[v_id] * self.mov_energy_base ** alg.l2_norm_square(
    #                                 sur_vec * 0.01)
    #                             v_energy *= self.rot_energy_base ** (theta ** 2)
    #                             obs_coff = 1
    #                             if sur_vel > 0 and len(self.agent.v_cur_tiling.sur_obs_grids_ids) > 0:
    #                                 norm_vec = sur_vec / sur_vel
    #                                 obs_coff = float('inf')
    #                                 for o_v, o_d_rev_2, o_sup in self.agent.v_cur_tiling.sur_obs_bound_tiling_attr:
    #                                     epsilon = (0.5 + np.dot(norm_vec, o_v) * o_d_rev_2) ** o_sup
    #                                     if epsilon < obs_coff:
    #                                         obs_coff = epsilon
    #                             v_energy *= obs_coff
    #                             cross_v_tiling.prob_energy = v_energy
    #                         cross_vt_engs.append(v_energy)
    #                         cross_v_energy += v_energy
    #
    #     alpha = 0.2
    #     cross_scene_energy = (1 - alpha) * cross_p_energy + alpha * cross_v_energy
    #     inter_dis *= 0.01
    #     p1, p2 = np.exp(inter_dis), np.exp(-inter_dis)
    #     phy_fwd_attenuation_coff = (p1 - p2) / (p1 + p2)
    #     return phy_fwd_attenuation_coff * cross_scene_energy, (cross_vt_ids, cross_pt_ids, cross_vt_engs, cross_pt_engs)

    def render(self, wdn_obj, default_color):
        super().render(wdn_obj, default_color)
        if not self.enable_draw:
            return
        if self.reset_state:

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
