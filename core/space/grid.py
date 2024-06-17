from core.space import *


class Tiling:

    def __init__(self):
        self.id: int = 0
        self.corr_scene = None
        self.mat_loc: Tuple = tuple()  # row,col
        self.center: np.ndarray = np.array([0, 0])
        self.cross_bound: np.ndarray = np.array([])  # the boundary across this tiling
        self.rect: Tuple = (0,)
        self.x_min: float = 0
        self.x_max: float = 0
        self.y_min: float = 0
        self.y_max: float = 0
        self.type: int = 1  # 1 totally inside space, 0 - some or all part collides with space
        self.nei_ids: Tuple = ()
        self.rela_patch = None
        self.corr_conv_ids = []
        self.corr_conv_inters = []
        self.corr_conv_cin = -1

        self.vis_tri: Tuple = ()
        self.vis_rays: Tuple = ()
        self.vis_360angle_partition: Tuple = ()

        self.flat_weight = 0
        self.flat_grad = np.array([0, 0])
        self.nearst_obs_grid_id = 0

        self.sur_grids_ids: Tuple = ()
        self.sur_obs_grids_ids = []
        self.sur_obs_bound_grids_ids = []
        self.sur_occu_safe = True
        self.sur_360_partition = []
        self.sur_vis_occu_ids = []

        self.prob_energy = 0
        self.rot_max_weight_runtime = 0
        self.sur_obs_bound_tiling_attr = []
        self.rot_occu_grid_attr_runtime = []

    def calc_sur_tiling_prob(self, sur_tiling, loc, fwd, effect_r, enable_obs_coff=True):
        """


        Args:
            sur_tiling:
            loc:用户当前位置
            fwd:
            effect_r:
            enable_obs_coff:

        Returns:

        """
        sur_prob = 0
        if sur_tiling.type:
            sur_vec = sur_tiling.center - loc
            sur_vel = alg.l2_norm(sur_vec)
            theta = geo.calc_angle_bet_vec(sur_vec, fwd)
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
