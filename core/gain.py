from core import *


class BaseGainManager(BaseManager):

    def __init__(self):
        super().__init__()
        self.gain_spec = const_gain
        # --------------------------重定向增益相关变量-----------------------------------
        # [[min, max, default_min, default_max], [min, max, default_min, default_max]]
        # |------------最严格约束范围-----------|---|-----------最宽松约束范围------------|
        self.gt_const = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])  # translation gain
        self.gr_const = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])  # rotation gain
        self.gc_const = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])  # curvature gain
        self.gb_const = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])  # bending gain
        # [[gt_r,gr_r,gc_r,gb_r],[dgt_r,dgr_r,dgc_r,dgb_r]],当前增益变化率,默认变化率
        self.g_rate_const = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        # [[gt,gr,gc,gb],[dgt,dgr,dgc,dgb]],当前重定向增益值
        self.g_pri = np.array([[1, 1, 0, 1], [1, 1, 0, 1]])
        self.gt_range = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.gr_range = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.gc_range = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.gb_range = np.array([[0.0, 0.0], [0.0, 0.0]])
        # [gt,gr,gc,gb]
        self.g_values = np.array([0.0, 0.0, 0.0, 0.0])
        self.g_rates = np.array([0.0, 0.0, 0.0, 0.0])

    def load_params(self):
        self.gt_const = np.array(self.gain_spec['trans_gain'])
        self.gr_const = np.array(self.gain_spec['rot_gain'])
        self.gc_const = np.array(self.gain_spec['cur_gain'])
        self.gb_const = np.array(self.gain_spec['bend_gain'])
        self.g_rate_const = np.array(self.gain_spec['gains_rate'])
        self.g_pri = np.array(self.gain_spec['pri_gains'])

    def setup_gains(self):
        self.gt_range[0:2, 0:2] = self.gt_const[0:2, 0:2]
        self.gr_range[0:2, 0:2] = self.gr_const[0:2, 0:2]
        self.gc_range[0:2, 0:2] = self.gc_const[0:2, 0:2]
        self.gb_range[0:2, 0:2] = self.gb_const[0:2, 0:2]
        self.g_values = self.g_pri[0]
        self.g_rates = self.g_rate_const[0]

    def prepare(self):
        self.p_scene, self.v_scene = self.agent.p_scene, self.agent.v_scene

    @abstractmethod
    def reset(self):
        """
        重置重定向增益值

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs):
        """
        动态更新重定向增益

        Returns:

        """
        raise NotImplementedError

    def copy_target_manager(self, other_mg):
        if other_mg is not None:
            self.gt_const = other_mg.gt_const.copy()
            self.gr_const = other_mg.gr_const.copy()
            self.gc_const = other_mg.gc_const.copy()
            self.gb_const = other_mg.gb_const.copy()
            self.g_rate_const = other_mg.g_rate_const.copy()
            self.g_pri = other_mg.g_pri.copy()
        else:
            self.load_params()
        self.setup_gains()


class SimpleGainManager(BaseGainManager):

    def __init__(self):
        super(SimpleGainManager, self).__init__()

    def reset(self):
        self.g_values = self.g_pri[0]

    def update(self, **kwargs):
        self.g_values += self.g_rates

    def render(self, wdn_obj, default_color):
        pass
