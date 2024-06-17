from core import *


class BaseRdwManager(BaseManager):

    def __init__(self):
        super().__init__()
        self.rdw_type = None
        self.rdw_spec = const_steer
        self.enable_rdw = True

    def load_params(self):
        pass

    def prepare(self):
        self.p_scene, self.v_scene = self.agent.p_scene, self.agent.v_scene

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calc_vir_state(self):
        raise NotImplementedError

    def copy_target_manager(self, other_mg):
        pass


class NoRdwManager(BaseRdwManager):

    def __init__(self):
        super(NoRdwManager, self).__init__()
        self.rdw_type = 'No Rdw'
        # constant of reactive rdw
        self.steer_vel_c = np.array([0, 0, 0, 0])
        self.steer_dampen = np.array([0, 0])  # dumping terms, distance dumping [0] and rotation dumping [1``]
        self.steer_force = np.array([0, 0])
        self.enable_rdw = False

    def load_params(self):
        self.steer_vel_c = [self.rdw_spec['static_rot'] * DEG2RAD,
                            self.rdw_spec['move_dt'],
                            self.rdw_spec['max_gc_rot'] * DEG2RAD,
                            self.rdw_spec['max_gr_rot'] * DEG2RAD]
        self.steer_dampen = [self.rdw_spec['dampen_dis'], self.rdw_spec['dampen_bear'] * DEG2RAD]

    def reset(self):
        self.enable_rdw = False

    def update(self, **kwargs):
        pass

    def calc_vir_state(self):
        vir_next_fwd = geo.norm_vec(geo.rot_vecs(self.v_fwd, self.p_rot))
        vir_move_fwd = vir_next_fwd * self.p_vel
        self.agent.v_cur_loc = self.v_loc + vir_move_fwd
        self.agent.v_cur_fwd = vir_next_fwd
        self.agent.v_cur_vel = self.p_vel
        self.agent.v_cur_rot = self.p_rot

    def render(self, wdn_obj, default_color):
        pass


class SteerRdwManager(NoRdwManager):

    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()
        self.enable_rdw = True

    def calc_vir_state(self):
        gain_mg = self.agent.gainer
        p_walk_rot = self.p_vel * 0.01 * gain_mg.g_values[2] * RAD2DEG
        if self.enable_rdw:
            rot_dir = 0
            if self.p_rot != 0:
                rot_dir = self.p_rot / abs(self.p_rot)  # + clockwise - anti_clockwise
            desired_rot_dir = 0
            desired_rotation = geo.calc_angle_bet_vec(self.p_fwd, self.steer_force)
            if desired_rotation != 0:
                desired_rot_dir = desired_rotation / abs(desired_rotation)
            else:
                desired_rotation = PI_1_4
            abs_ph_rot = abs(self.p_rot)
            if rot_dir * desired_rot_dir > 0:
                rot_slow = alg.clamp(abs_ph_rot * (1 - gain_mg.gr_range[0][0]), 0,
                                     self.steer_vel_c[3] * self.time_step)
                head_rot_final = abs_ph_rot - rot_slow
            else:
                rotation_accelerate = alg.clamp(abs_ph_rot * (gain_mg.gr_range[0][1] - 1), 0,
                                                self.steer_vel_c[3] * self.time_step)
                head_rot_final = abs_ph_rot + rotation_accelerate
            moving_rot_final = alg.clamp(abs(p_walk_rot), 0, self.steer_vel_c[2] * self.time_step)
            scale_factor = 1
            bearing_to_target = abs(desired_rotation)
            length_to_target = alg.l2_norm(self.steer_force)
            if bearing_to_target < self.steer_dampen[1]:
                scale_factor *= math.sin(math.pi * bearing_to_target / (2 * self.steer_dampen[1]))

            '''if length_to_target < self.steer_dampen[0]:
                # print("moving scale trigger:", length_to_target / self.steer_mg.steer_dampen[0])
                scale_factor *= length_to_target / self.steer_mg.steer_dampen[0]'''
            selected_rot = max(moving_rot_final, head_rot_final) * scale_factor
            # 当前正在行走
            if self.p_vel > self.steer_vel_c[1] * self.time_step:
                if rot_dir == 0:
                    clk_angle = abs(geo.calc_angle_bet_vec(self.steer_force, geo.rot_vecs(self.p_fwd, selected_rot)))
                    anti_clk_angle = abs(
                        geo.calc_angle_bet_vec(self.steer_force, geo.rot_vecs(self.p_fwd, -selected_rot)))
                    rot_dir = 1 if clk_angle > anti_clk_angle else -1
            vir_rot_vel = selected_rot * rot_dir
            vir_mov_vel = self.p_vel * gain_mg.g_values[0]
        else:
            vir_rot_vel = self.p_rot
            vir_mov_vel = self.p_vel
        vir_next_fwd = geo.norm_vec(geo.rot_vecs(self.v_fwd, vir_rot_vel))
        vir_move_fwd = vir_next_fwd * vir_mov_vel
        self.agent.v_cur_loc = self.v_loc + vir_move_fwd
        self.agent.v_cur_fwd = vir_next_fwd
        self.agent.v_cur_vel = vir_mov_vel
        self.agent.v_cur_rot = vir_rot_vel

    def render(self, wdn_obj, default_color):
        wdn_obj.draw_phy_line(self.p_loc, self.steer_force + self.p_loc, 2, (255, 0, 0))

    def copy_target_manager(self, other_mg):
        if other_mg is not None:
            self.steer_vel_c = pickle.loads(pickle.dumps(other_mg.steer_vel_c))
            self.steer_dampen = pickle.loads(pickle.dumps(other_mg.steer_dampen))
        else:
            self.load_params()


class S2CRdwManager(SteerRdwManager):
    """
    refer to https://ieeexplore.ieee.org/abstract/document/6479192

    """

    def __init__(self):
        super().__init__()
        self.rdw_type = 'S2C Rdw'

    def update(self, **kwargs):
        self.steer_force = self.agent.p_cur_conv.center - self.p_loc


class S2ORdwManager(SteerRdwManager):
    """
    refer to https://ieeexplore.ieee.org/abstract/document/6479192


    """

    def __init__(self):
        super().__init__()
        self.rdw_type = 'S2O Rdw'
        self.steer_circle = None

    def update(self, **kwargs):
        self.steer_circle = self.agent.p_cur_conv.in_circle
        r = self.steer_circle[1] * 0.8
        center = np.array(self.steer_circle[0])
        p2center = center - self.p_loc
        m, n = p2center
        dis2center = alg.l2_norm(p2center)
        if dis2center > r:
            a = dis2center ** 2
            b = 2 * r * n
            c = r * r - m * m
            sin_theta1 = (-b + (b * b - 4 * a * c) ** 0.5) / (2 * a)
            sin_theta2 = (-b - (b * b - 4 * a * c) ** 0.5) / (2 * a)
            cx, cy = self.steer_circle[0]
            y1 = sin_theta1 * r + cy
            y2 = sin_theta2 * r + cy
            x1 = (1 - sin_theta1 ** 2) ** 0.5 * r + cx
            x2 = -(1 - sin_theta1 ** 2) ** 0.5 * r + cx
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y1])
            vt1 = self.p_loc - p1
            vt2 = center - p1
            vt3 = self.p_loc - p2
            vt4 = center - p2
            cos_t1 = abs(np.dot(vt1, vt2) / (alg.l2_norm(vt1) * alg.l2_norm(vt2)))
            cos_t2 = abs(np.dot(vt3, vt4) / (alg.l2_norm(vt3) * alg.l2_norm(vt4)))
            if cos_t1 < cos_t2:
                tar1 = np.array([x1, y1])
            else:
                tar1 = np.array([x2, y1])
            x1 = (1 - sin_theta2 ** 2) ** 0.5 * r + cx
            x2 = -(1 - sin_theta2 ** 2) ** 0.5 * r + cx
            p1 = np.array([x1, y2])
            p2 = np.array([x2, y2])
            vt1 = self.p_loc - p1
            vt2 = center - p1
            vt3 = self.p_loc - p2
            vt4 = center - p2
            cos_t1 = abs(np.dot(vt1, vt2) / (alg.l2_norm(vt1) * alg.l2_norm(vt2)))
            cos_t2 = abs(np.dot(vt3, vt4) / (alg.l2_norm(vt3) * alg.l2_norm(vt4)))
            if cos_t1 < cos_t2:
                tar2 = np.array([x1, y2])
            else:
                tar2 = np.array([x2, y2])
        else:
            a = 4
            b = 6 * dis2center
            c = 3 * dis2center ** 2 - r ** 2
            ml = (-b + (b * b - 16 * c) ** 0.5) / 8
            nl = (r * r - ml * ml) ** 0.5

            v = geo.norm_vec(p2center)
            pm = center + v * ml
            v_nl = v * nl
            tar1 = np.array([pm[0] - v_nl[1], pm[1] + v_nl[0]])
            tar2 = np.array([pm[0] + v_nl[1], pm[1] - v_nl[0]])
        ap1 = abs(geo.calc_angle_bet_vec(tar1 - self.p_loc, self.p_fwd))
        ap2 = abs(geo.calc_angle_bet_vec(tar2 - self.p_loc, self.p_fwd))
        tar = tar1 if ap1 < ap2 else tar2
        self.steer_force = tar - self.p_loc
