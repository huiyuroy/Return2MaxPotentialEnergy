import time

import numpy as np
from core import *
from abc import abstractmethod, ABC


class Logger(ABC):

    def __init__(self, agent):
        self.agent = agent
        self.delta_time = 0
        self.rdw_type = 'none'
        self.reset_type = 'none'
        self.walk_state = 'normal'
        self.v_scene_size = None
        self.p_scene_size = None
        self.p_walk_dis = 0
        self.p_cur_vel = np.array([0, 0])
        self.v_cur_vel = np.array([0, 0])
        self.rdw_cur_rate = np.array([0, 0])
        self.rdw_total_rate = np.array([0, 0])
        self.rdw_mean_rate = np.array([0, 0])
        self.g_rot_range = [0, 0]
        self.g_tran_range = [0, 0]
        self.g_curv_range = [0, 0]
        self.gains = [0, 0, 0, 0]
        self.gains_rate = [0, 0, 0, 0]
        self.p_pos_record = []
        self.reset_pos_record = []
        self.past_reset_num = []
        self.reset_num = 0
        self.max_reset_num = 0
        self.total_reset_num = 0
        self.cur_step = 0
        self.episode_num = 0
        self.max_record_len = 100
        self.past_epi_frames = None

    def set_agent(self, agent):
        self.agent = agent

    def set_epi_data_buffer(self, max_rec_len):
        self.max_record_len = max_rec_len
        self.past_reset_num = [0] * self.max_record_len
        self.past_epi_frames = [0] * self.max_record_len

    @abstractmethod
    def prepare_log(self):
        raise NotImplementedError

    @abstractmethod
    def reset_epi_data(self):
        raise NotImplementedError

    @abstractmethod
    def record_step_data(self):
        raise NotImplementedError

    @abstractmethod
    def log_epi_data(self):
        raise NotImplementedError


class RDWBaseLogger(Logger):

    def __init__(self, env):
        super().__init__(env)
        self.agents_log = {}

    def prepare_log(self):
        # for ag in self.rdw_env.agents:
        #     self.agents_log[ag.name] = {'total_resets': 0}
        pass

    def reset_epi_data(self):
        self.rdw_type = self.agent.rdwer.rdw_type
        self.v_scene_size = self.agent.v_scene.max_size[:]
        self.p_scene_size = self.agent.p_scene.max_size[:]
        self.cur_step = 0
        self.p_walk_dis = 0
        self.p_cur_vel = np.array([0, 0])
        self.v_cur_vel = np.array([0, 0])
        self.rdw_cur_rate = np.array([0, 0])
        self.rdw_total_rate = np.array([0, 0])
        self.rdw_mean_rate = np.array([0, 0])
        self.g_rot_range = self.agent.gainer.gr_range[0].tolist()
        self.g_tran_range = self.agent.gainer.gt_range[0].tolist()
        self.g_curv_range = self.agent.gainer.gc_range[0].tolist()
        self.p_pos_record = []
        self.reset_pos_record = []
        self.reset_num = 0
        self.delta_time = 1 / TIME_STEP

    def record_step_data(self):
        self.cur_step += 1

        self.p_cur_vel = np.array([self.agent.p_cur_vel, self.agent.p_cur_rot]) * self.delta_time
        self.v_cur_vel = np.array([self.agent.v_cur_vel, self.agent.v_cur_rot]) * self.delta_time
        if self.p_cur_vel[0] > 0:
            self.p_walk_dis += self.p_cur_vel[0]
            self.p_pos_record.append(self.agent.p_cur_loc)
        self.gains = self.agent.gainer.g_values
        self.gains_rate = self.agent.gainer.g_rates
        if not self.agent.resetter.reset_state:
            self.walk_state = 'normal'
            self.reset_type = 'none'
            self.rdw_cur_rate[0] = self.v_cur_vel[0] - self.p_cur_vel[0]  # 当前速度变化率
            self.rdw_cur_rate[1] = abs(self.v_cur_vel[1]) - abs(self.p_cur_vel[1])  # 当前旋转变化率（重定向率）
            self.rdw_total_rate += self.rdw_cur_rate
            self.rdw_mean_rate = self.rdw_total_rate / self.cur_step
        else:
            self.walk_state = 'reset'
            self.reset_type = self.agent.resetter.reset_type
            if not self.agent.resetter.reset_pre_state:
                self.reset_num += 1
                reset_info = []
                reset_info.extend(self.agent.p_cur_loc)
                reset_info.extend(self.agent.resetter.reset_target_fwd)
                reset_info.append(self.agent.resetter.reset_angle)
                self.reset_pos_record.append(reset_info)

    def log_epi_data(self):
        data = {'v_name': self.agent.v_scene.name,
                'p_name': self.agent.p_scene.name,
                'traj_type': self.agent.inputer.cur_traj.type,
                'alg_name': self.agent.name,
                'reset_num': self.reset_num,
                'walk_dis': self.p_walk_dis * 0.01,
                'walk_pos': np.array(self.p_pos_record).tolist(),
                'reset_pos': np.array(self.reset_pos_record).tolist()
                }
        return data


class RDWTrajectoryWalkerLogger(RDWBaseLogger):

    def __init__(self, env):
        super().__init__(env)

        self.cur_traj_type = None
        self.total_traj_num = 0
        self.cur_traj_idx = 0
        self.total_tar_num = 0
        self.cur_tar_idx = 0

    def record_step_data(self):
        super().record_step_data()
