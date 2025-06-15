from pyrdw.core import *
from pyrdw.core.agent.base import BaseAgent


class Logger(ABC):

    def __init__(self, agent):
        self.agent: BaseAgent = agent
        self.delta_time = const_env['time_step']
        self.rdw_type = 'none'
        self.reset_type = 'none'
        self.reset_state = 'normal'
        self.v_scene_size = None
        self.p_scene_size = None
        self.p_walk_dis = 0
        self.v_walk_dis = 0
        self.p_cur_vel = np.array([0, 0])
        self.v_cur_vel = np.array([0, 0])
        self.rdw_total_rate = np.array([0, 0])
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

    def set_agent(self, agent: BaseAgent):
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

    def __init__(self, agent):
        super().__init__(agent)
        self.epi_states = []

    def prepare_log(self):
        """
        clean up total reset information in prepare.

        Returns:

        """
        self.total_reset_num = 0

    def reset_epi_data(self):
        self.epi_states = []
        self.rdw_type = self.agent.rdwer.rdw_type
        self.v_scene_size = self.agent.v_scene.max_size[:]
        self.p_scene_size = self.agent.p_scene.max_size[:]
        self.cur_step = 0
        self.p_walk_dis = 0
        self.v_walk_dis = 0
        self.p_cur_vel = np.array([0, 0])
        self.v_cur_vel = np.array([0, 0])
        self.rdw_total_rate = []
        self.g_rot_range = self.agent.gainer.gr_range[0].tolist()
        self.g_tran_range = self.agent.gainer.gt_range[0].tolist()
        self.g_curv_range = self.agent.gainer.gc_range[0].tolist()
        self.p_pos_record = []
        self.reset_pos_record = []
        self.reset_num = 0

    def record_step_data(self):
        step_state = self.agent.state_log()
        self.epi_states.append(step_state)
        self.cur_step = step_state['t']
        if not step_state['rs_s']:
            self.rdw_total_rate.append(step_state['rdw_r'])
            self.p_walk_dis += step_state['pwv']
            self.v_walk_dis += step_state['vwv']

    def log_epi_data(self):
        self.reset_num = self.epi_states[-1]['rs_n']
        self.total_reset_num += self.reset_num
        states = []
        for s in self.epi_states:
            new_s = {
                't': s['t'],  # current step
                'pwv': s['pwv'],  # p walking velocity
                'prv': s['prv'],  # p rotating velocity
                'p_loc': np.round(s['p_loc'], decimals=1).tolist(),  # p location
                'p_fwd': np.round(s['p_fwd'], decimals=1).tolist(),  # p forward
                'v_loc': np.round(s['v_loc'], decimals=1).tolist(),
                'v_fwd': np.round(s['v_fwd'], decimals=1).tolist(),
                'vwv': s['vwv'],
                'vrv': s['vrv'],
                'rdw_r': s['rdw_r'],
                'rs_s': s['rs_s'],
                'rs_t': s['rs_t'],
                'rs_n': s['rs_n'],
                'rs_info': s['rs_info'],
            }
            states.append(new_s)

        data = {
            'v_name': self.agent.v_scene.name,
            'p_name': self.agent.p_scene.name,
            'traj_type': self.agent.inputer.cur_traj.type if self.agent.inputer.cur_traj is not None else 'live',
            'alg_name': self.agent.name,
            'reset_num': self.reset_num,
            'total_reset_num': self.total_reset_num,  # reset number since prepare function
            'mean_rdw_rate': np.array(self.rdw_total_rate).mean() / self.delta_time * RAD2DEG,
            'walk_dis': self.p_walk_dis * 0.01,
            'avg_dis_btw_resets': self.p_walk_dis * 0.01 / (self.reset_num + 1),
            'v_walk_dis': self.v_walk_dis * 0.01,
            'vdis_btw_resets': self.v_walk_dis * 0.01 / (self.reset_num + 1),
            'state_traj': states  # 该回合完整的状态轨迹
        }
        return data


class RDWTrajectoryWalkerLogger(RDWBaseLogger):

    def __init__(self, agent):
        super().__init__(agent)
        self.cur_traj_type = None
        self.total_traj_num = 0
        self.cur_traj_idx = 0
        self.total_tar_num = 0
        self.cur_tar_idx = 0

    def record_step_data(self):
        super().record_step_data()