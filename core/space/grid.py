from core.space import *


class Tiling:

    def __init__(self):
        self.id: int = 0
        self.corr_scene = None
        self.mat_loc: Tuple = tuple()  # row,col
        self.center: np.ndarray = np.array([0, 0])
        self.cross_bound: np.ndarray = np.array([])  # 穿过这个tiling的边界
        self.rect: Tuple = (0,)
        self.x_min: float = 0
        self.x_max: float = 0
        self.y_min: float = 0
        self.y_max: float = 0
        self.type: int = 1  # 1 totally inside space, 0 - some or all part collides with space
        self.nei_ids: Tuple = ()
        self.rela_patch = None
        self.corr_conv_ids = []  # 相交的凸多边形区域id
        self.corr_conv_inters = []  # 相交区域
        self.corr_conv_cin = -1  # tiling center 所在的凸多边形区域id

        self.vis_tri: Tuple = ()
        self.vis_rays: Tuple = ()
        self.vis_360angle_partition: Tuple = ()  # 记录当前tiling360度每个角度分区对应的可见三角形编号，哈希

        self.flat_weight = 0  # 本权重用于显示可行区域，所有可行区域阈值相近
        self.flat_grad = np.array([0, 0])
        self.nearst_obs_grid_id = 0  # 距离最近障碍物tiling的id（此处采用occupancy grid方法，求与最近障碍物tiling的距离）

        self.sur_grids_ids: Tuple = ()  # 指定半径范围内的其他tiling的ids,指定半径使用scene对象的human_step_single数值
        self.sur_obs_grids_ids = []  # 指定半径范围内的障碍物tiling的ids
        self.sur_obs_bound_grids_ids = []  # 指定半径范围内的障碍物边界tiling的ids（1个human step 范围内）
        self.sur_occu_safe = True  # 周围1步内无遮挡则为true
        # 以[0,1]旋转角度划分若干区域，每个区域内的可达tiling，目前将角度划分为120个区间，每个区间代表3度（见space对象）
        self.sur_360_partition = []  # 周围360度可见区域
        self.sur_vis_occu_ids = []

        self.prob_energy = 0
        self.rot_max_weight_runtime = 0
        self.sur_obs_bound_tiling_attr = []
        self.rot_occu_grid_attr_runtime = []

    def calc_sur_tiling_prob(self, sur_tiling, loc, fwd, effect_r, enable_obs_coff=True):
        """
        已知用户所在tiling，求某个指定tiling的能量

        Args:

            loc:用户当前位置
            fwd:
            enable_obs_coff:
        Returns:

        """
        sur_prob = 0
        if sur_tiling.type:
            sur_vec = sur_tiling.center - loc
            sur_vel = alg.l2_norm(sur_vec)
            theta = geo.calc_angle_bet_vec(sur_vec, fwd)
            # 将[-pi,pi]作为99%置信区间，防止角度大时能量太小 https://blog.csdn.net/kaede0v0/article/details/113790060
            # rot_prob = np.exp(-0.5 * (theta / np.pi) ** 2 * 4) / ((2 * np.pi) ** 0.5)
            # 将3*human step作为99%置信区间 https://blog.csdn.net/kaede0v0/article/details/113790060
            # mov_prob = np.exp(-0.5 * (sur_vel / self.human_step_single) ** 2 * 9 / 4) / ((2 * np.pi) ** 0.5)
            # sur_prob = rot_prob * mov_prob
            sur_prob = np.exp(-(4.5 * (theta * REV_PI) ** 2 + 0.5 * (sur_vel / effect_r) ** 2)) * REV_PI_2
            obs_coff = 1
            if enable_obs_coff and sur_vel > 0 and len(self.sur_obs_grids_ids) > 0:
                norm_vec = sur_vec / sur_vel
                obs_coff = float('inf')
                for obs_vec, obs_d, obs_d_rev in self.sur_obs_bound_tiling_attr:
                    epsilon = (0.5 + np.dot(norm_vec, obs_vec) * obs_d_rev * 0.5) ** (effect_r * obs_d_rev)
                    if epsilon < obs_coff:
                        obs_coff = epsilon
            sur_prob *= obs_coff
            sur_prob *= sur_tiling.flat_weight
        return sur_prob
