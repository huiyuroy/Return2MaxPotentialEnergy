import lib.math.geometry as geo
from vis.ui import *

update_freq = 1000
draw_scale = const_env["file_scene_scale"]  # 场景数据中1个单位为现实1cm
human_radius = const_env["human_radius"]
scene_board = const_env["scene_board"]
v_obs_width = 600
vir_obs_sp = [0, 0, v_obs_width, v_obs_width]
p_obs_width = 600
p_obs_sp = [0, 0, p_obs_width, p_obs_width]


class DrawColor(Enum):
    Black = (0, 0, 0)
    White = (255, 255, 255)
    Red = (255, 0, 0)
    Green = (0, 255, 0)
    Blue = (0, 0, 255)
    Yellow = (0, 255, 255)
    DarkGray = (64, 64, 64)
    User = (72, 72, 72)
    LightGray = (225, 225, 225)


class RDWWindow:
    def __init__(self, env, ui_spec):
        self.p_max_size = (10, 30, 920, 920)
        self.v_max_size = (935, 30, v_obs_width, v_obs_width)
        self.c_max_size = (945, 35 + v_obs_width, v_obs_width, 925 - v_obs_width)
        self.screen_size = (self.v_max_size[0] + self.v_max_size[2] + 5, 960)
        self.env = env
        self.agents = None
        self.main_ui_spec = ui_spec
        self.uis_spec = self.main_ui_spec["DRL_Render_Screen"]
        self.f_label_spec = self.uis_spec["fix_labels"]
        self.v_label_spec = self.uis_spec["var_labels"]
        self.v_label_txt = {}
        self.back_surf = None
        self.back_clr = (0, 0, 0)

        self.v_shp_ivs = ()
        self.p_shp_ivs = ()
        self.v_obs_render_sp = None
        self.v_render_pos = None
        self.p_render_pos = None
        self.v_render_center = None
        self.p_render_center = None

        # --------------------------虚实场景绘制相关变量-----------------------------------
        self.v_scene, self.p_scene = None, None
        self.v_surf_size, self.p_surf_size = np.array([500.0, 500.0]), np.array([500.0, 500.0])  # 虚实surface大小
        self.v_surf_size_2, self.p_surf_size_2 = np.array([250.0, 250.0]), np.array([250.0, 250.0])  # 虚实surface大小
        self.v_surf, self.v_surf_bk = None, None  # 虚拟空间surface区域，1像素对应1cm
        self.p_surf, self.p_surf_bk = None, None
        self.v_draw_size, self.v_scale = None, 0
        self.p_draw_size, self.p_scale = None, 0
        self.v_draw_loc, self.p_draw_loc = None, None
        self.v_surf_data, self.p_surf_data = None, None
        # 虚拟空间滑动观察窗口大小，默认800cm * 800cm。主要原因是虚拟surface太大，难以直接全部显示
        self.v_obs_sp = vir_obs_sp
        self.user_r = 0  # 用户标识的半径
        # --------------------------虚实场景绘制控制变量-----------------------------------
        self.enable_vir_render, self.enable_phy_render = False, False
        self.enable_ptiling_render = False
        self.enable_pconv_render = False
        self.enable_path_render, self.enable_steer_render = True, True
        self.enable_simu_tar_render, self.enable_walk_traj_render = True, True
        self.enable_reset_render = False
        # --------------------------绘制队列-----------------------------------------------
        """
           seq_circles-> 要绘制的位置，记录位置点，每个点记录为(x, y)，即((x1,y1), (x2,y2), .....) n*2
           seq_lines-> 要绘制的向量，记录每对位置点，即((x1s,y1s), (x1e,y1e), (x2s,y2s), (x2e,y2e), ......) 2n*2
           seq_polys-> 要绘制的区域，记录一系列位置点，即(((x1s,y1s), (x1e,y1e), (x2s,y2s), (x2e,y2e), ...), (...), ...)，每个区域
                       用一个包含n个点的列表表示，多个区域最终包含在一个tuple里作为返回值给出。
        """
        self.seq_v_circles = []
        self.seq_v_circles_attri = []  # radius, color
        self.seq_v_lines = []
        self.seq_v_lines_attri = []  # width, color
        self.seq_v_polys = []
        self.seq_v_polys_attri = []  # width, color

        self.seq_p_circles = []
        self.seq_p_circles_attri = []  # radius, color
        self.seq_p_lines = []
        self.seq_p_lines_attri = []  # width, color
        self.seq_p_polys = []
        self.seq_p_polys_attri = []  # width, color
        # -------------------------交互属性-------------------------------------
        self.dragging = False
        self.click_pos = ()

        # -------------------------ui组件初始化---------------------------------
        self.__init_ui()

    def __init_ui(self):
        pygame.init()
        pygame.display.set_caption(self.uis_spec["title"])
        self.ui_surface_clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("timesnewroman", 20)
        self.back_surf = pygame.display.set_mode(self.screen_size, flags=pygame.DOUBLEBUF, depth=32, display=0, vsync=0)
        self.back_surf.fill((255, 255, 255))
        pygame.draw.rect(self.back_surf, self.back_clr, self.v_max_size, 1)
        pygame.draw.rect(self.back_surf, self.back_clr, self.p_max_size, 1)
        pygame.draw.rect(self.back_surf, self.back_clr, self.c_max_size, 1)

        blits_seq = (
            (self.ui_font.render('Physical Space', True, self.back_clr), (10, 5)),
            (self.ui_font.render('Virtual Space', True, self.back_clr), (935, 5)),
            (self.ui_font.render('Log', True, self.back_clr), (945, 45 + v_obs_width))
        )

        self.back_surf.blits(blits_seq)

        # for k, v in self.v_label_spec.items():
        #     if k == "pre_train_label" or k == "cur_train_label" or k == "general_train_label" \
        #             or k == "cur_rdw_label" or k == "cur_walk_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text']
        #     elif k == "pre_train_index_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.episode_num - 1)
        #     elif k == "pre_train_total_rwd_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(
        #             self.rdw_logger.get_last_epi_total_rwd())
        #     elif k == "pre_train_mean_ls_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(
        #             self.rdw_logger.get_last_epi_mean_ls())
        #     elif k == "pre_train_total_step_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(
        #             self.rdw_logger.get_last_epi_frames())
        #     elif k == "cur_train_index_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(
        #             self.rdw_logger.get_cur_epi_idx())
        #     elif k == "cur_train_step_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.cur_step)
        #     elif k == "cur_train_step_rwd_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.cur_rwd)
        #     elif k == "cur_train_step_ls_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.cur_ls)
        #     elif k == "cur_train_total_rwd_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.cur_total_reward)
        #     elif k == "gen_train_max_step_rwd_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.max_single_rwd)
        #     elif k == "gen_train_max_total_rwd_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.max_total_rwd)
        #     elif k == "gen_train_fps_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.ui_surface_clock.get_fps())
        #     elif k == "cur_rdw_alg_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'] + self.rdw_logger.rdw_type
        #     elif k == "cur_rdw_state_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'] + self.rdw_logger.walk_state
        #     elif k == "cur_rdw_rot_gain_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.gains[1],
        #                                                                             self.rdw_logger.gains_rate[1],
        #                                                                             self.rdw_logger.gr_range[0],
        #                                                                             self.rdw_logger.gr_range[1])
        #     elif k == "cur_rdw_trans_gain_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.gains[0],
        #                                                                             self.rdw_logger.gains_rate[0],
        #                                                                             self.rdw_logger.gt_range[0],
        #                                                                             self.rdw_logger.gt_range[1])
        #     elif k == "cur_rdw_curv_gain_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.gains[2],
        #                                                                             self.rdw_logger.gains_rate[2],
        #                                                                             self.rdw_logger.gc_range[0],
        #                                                                             self.rdw_logger.gc_range[1])
        #     elif k == "cur_rdw_mov_vel_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.phy_cur_vel[0],
        #                                                                             self.rdw_logger.vir_cur_vel[0])
        #     elif k == "cur_rdw_rot_vel_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.phy_cur_vel[1],
        #                                                                             self.rdw_logger.vir_cur_vel[1])
        #     elif k == "cur_rdw_vel_rate_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.rdw_cur_rate[0])
        #     elif k == "cur_rdw_rot_rate_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.rdw_cur_rate[1])
        #     elif k == "cur_rdw_reset_type_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'] + self.rdw_logger.reset_type
        #     elif k == "cur_rdw_reset_num_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.reset_num)
        #     elif k == "cur_walk_v_size_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.v_scene_size[0],
        #                                                                             self.rdw_logger.v_scene_size[1])
        #     elif k == "cur_walk_p_size_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.p_scene_size[0],
        #                                                                             self.rdw_logger.p_scene_size[1])
        #     elif k == "cur_walk_traj_type_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'] + self.rdw_logger.cur_traj_type
        #     elif k == "cur_walk_traj_index_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.cur_traj_idx,
        #                                                                             self.rdw_logger.total_traj_num)
        #     elif k == "cur_walk_tar_index_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.cur_tar_idx,
        #                                                                             self.rdw_logger.total_tar_num)
        #     elif k == "cur_walk_total_reset_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.total_reset_num)
        #     elif k == "cur_walk_max_epi_reset_label":
        #         self.v_label_txt[k] = lambda k: self.v_label_spec[k]['text'].format(self.rdw_logger.max_reset_num)

    def prepare_surfs(self):
        self.v_scene, self.p_scene = self.env.v_scene, self.env.p_scene
        v_w, v_h = list(map(lambda x: math.ceil((x + scene_board * 2) * draw_scale), self.v_scene.max_size))
        p_w, p_h = list(map(lambda x: math.ceil((x + scene_board * 2) * draw_scale), self.p_scene.max_size))
        v_w = v_w if v_w > vir_obs_sp[2] else vir_obs_sp[2] + 10
        v_h = v_h if v_h > vir_obs_sp[3] else vir_obs_sp[3] + 10
        self.v_surf_size = np.array([v_w, v_h], dtype=np.int32)
        self.v_surf_size_2 = self.v_surf_size * 0.5
        self.v_scale = v_obs_width / self.v_surf_size.max()
        self.v_draw_size = self.v_surf_size * self.v_scale
        self.v_draw_loc = (0, 0, *self.v_draw_size)

        self.p_surf_size = np.array([p_w, p_h], dtype=np.int32)
        self.p_surf_size_2 = self.p_surf_size * 0.5
        self.p_scale = p_obs_width / self.p_surf_size.max()
        self.p_draw_size = self.p_surf_size * self.p_scale
        self.p_draw_loc = (0, 0, *self.p_draw_size)
        self.user_r = int(human_radius * draw_scale * self.v_scale)

    def prepare_agents(self):
        ag_colors = ((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                     range(len(self.env.agents)))
        self.agents = tuple(zip(self.env.agents, ag_colors))
        for ag, ag_color in self.agents:
            setattr(ag, 'draw_color', ag_color)

    def render_surfs(self):
        self.v_surf.blit(self.v_surf_bk, (0, 0))
        self.p_surf.blit(self.p_surf_bk, (0, 0))
        for ag, ag_color in self.agents:
            ag.inputer.render(self, ag.draw_color)
            ag.gainer.render(self, ag.draw_color)
            ag.rdwer.render(self, ag.draw_color)
            ag.resetter.render(self, ag.draw_color)
        self.v_surf_data, self.p_surf_data = None, None
        if self.enable_vir_render:
            v_pre_s = pygame.transform.flip(self.v_surf.subsurface(self.v_draw_loc), False, True)
            self.v_surf_data = np.array(pygame.surfarray.pixels3d(v_pre_s))
        if self.enable_phy_render:
            p_ori_s = pygame.transform.flip(self.p_surf.subsurface(self.p_draw_loc), False, True)
            self.p_surf_data = np.array(pygame.surfarray.pixels3d(p_ori_s))  # w,h,c

    def reset_surfs(self):
        vsc = self.v_scene.scene_center
        psc = self.p_scene.scene_center
        self.v_surf = pygame.Surface(self.v_surf_size)
        self.p_surf = pygame.Surface(self.p_surf_size)
        self.v_surf.fill(DrawColor.Black.value)
        self.p_surf.fill(DrawColor.Black.value)
        for vb in self.v_scene.bounds:  # 虚拟空间整体surface
            b_ps = (np.array(vb.points) - vsc) * draw_scale + self.v_surf_size_2
            color = DrawColor.White.value if vb.is_out_bound else DrawColor.Black.value
            pygame.gfxdraw.aapolygon(self.v_surf, b_ps, color)
            pygame.gfxdraw.filled_polygon(self.v_surf, b_ps, color)

        for pb in self.p_scene.bounds:  # 物理空间整体surface
            b_ps = (np.array(pb.points) - psc) * draw_scale + self.p_surf_size_2
            color = DrawColor.White.value if pb.is_out_bound else DrawColor.Black.value
            pygame.gfxdraw.aapolygon(self.p_surf, b_ps, color)
            pygame.gfxdraw.filled_polygon(self.p_surf, b_ps, color)
        if self.enable_pconv_render:
            for conv in self.p_scene.conv_polys:
                c_v = (np.array(conv.vertices) - psc) * draw_scale + self.p_surf_size_2
                pygame.gfxdraw.aapolygon(self.p_surf, c_v, DrawColor.DarkGray.value)
                pygame.gfxdraw.filled_polygon(self.p_surf, c_v, DrawColor.LightGray.value)
        if self.enable_ptiling_render:
            for t in self.p_scene.tilings:
                c = (255, 255, 255) if t.type else (0, 0, 0)
                t_pos = ((t.center - psc) * draw_scale + self.p_surf_size_2).astype(np.int32)
                pygame.gfxdraw.filled_circle(self.p_surf, t_pos[0], t_pos[1], 1, c)

        self.v_surf = pygame.transform.smoothscale(self.v_surf, self.v_draw_size)
        self.p_surf = pygame.transform.smoothscale(self.p_surf, self.p_draw_size)
        self.v_surf_bk = self.v_surf.copy()
        self.p_surf_bk = self.p_surf.copy()
        self.render_surfs()

        if self.v_surf_data is not None:
            self.v_surf_data = np.transpose(self.v_surf_data, axes=(1, 0, 2))
            v_shape = self.v_surf_data.shape
            self.v_render_center = (self.v_max_size[0] + self.v_max_size[2] * 0.5,
                                    self.v_max_size[1] + self.v_max_size[3] * 0.5)
            self.v_render_pos = (self.v_render_center[0] - v_shape[1] * 0.5,
                                 self.v_render_center[1] - v_shape[0] * 0.5)
            self.v_shp_ivs = tuple(self.v_surf_data.shape[1::-1])
        if self.p_surf_data is not None:
            self.p_surf_data = np.transpose(self.p_surf_data, axes=(1, 0, 2))
            p_shape = self.p_surf_data.shape  # h,w of image
            self.p_render_center = (self.p_max_size[0] + self.p_max_size[2] * 0.5,
                                    self.p_max_size[1] + self.p_max_size[3] * 0.5)
            self.p_render_pos = (self.p_render_center[0] - p_shape[1] * 0.5,
                                 self.p_render_center[1] - p_shape[0] * 0.5)
            self.p_shp_ivs = tuple(self.p_surf_data.shape[1::-1])

    def update_surfs(self):
        """
        draw virtual and physical spaces on the surface, and update other elements of the surface.

        Returns:
            None
        """
        # -------monitor surface events----------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 鼠标按下，开始拖拽
                self.dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                # 鼠标释放，停止拖拽
                self.dragging = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.enable_vir_render = not self.enable_vir_render
                    self.enable_phy_render = not self.enable_phy_render

        p_scene_tar_pos = self.trans_pos_from_psurf(pygame.mouse.get_pos()) if self.dragging else None
        keys_pressed = pygame.key.get_pressed()
        vel = 0
        max_vel = 100
        if keys_pressed[pygame.K_w]:
            vel = max_vel
        elif keys_pressed[pygame.K_s]:
            vel = -max_vel
        fps = self.ui_surface_clock.get_fps()
        if fps != 0:
            delta_time = 1 / self.ui_surface_clock.get_fps()
        else:
            delta_time = 1 / 500

        for ag, ag_color in self.agents:
            if p_scene_tar_pos is not None:
                ag.inputer.p_fwd = np.array(p_scene_tar_pos) - ag.p_cur_loc
            ag.inputer.p_loc = ag.inputer.p_loc + geo.norm_vec(ag.inputer.p_fwd) * vel * delta_time

        # ------ui update-----------------------------------------------------------------------
        self.render_surfs()
        blits_seq = []
        if self.v_surf_data is not None:
            self.v_surf_data = np.transpose(self.v_surf_data, axes=(1, 0, 2))
            tpv_img = pygame.image.frombuffer(bytearray(self.v_surf_data.flatten()), self.v_shp_ivs, "RGB")
            blits_seq.append((tpv_img, self.v_render_pos))
        if self.p_surf_data is not None:
            self.p_surf_data = np.transpose(self.p_surf_data, axes=(1, 0, 2))
            tp_img = pygame.image.frombuffer(bytearray(self.p_surf_data.flatten()), self.p_shp_ivs, "RGB")
            blits_seq.append((tp_img, self.p_render_pos))

        self.back_surf.fill((255, 255, 255), self.c_max_size)
        blits_seq.append(
            (self.ui_font.render('fps:{}'.format(self.ui_surface_clock.get_fps()), True, self.back_clr),
             (945, 55 + v_obs_width)))
        # for k, v in self.v_label_spec.items():
        #     blits_seq.append((self.ui_font.render(self.v_label_txt[k](k), True, self.back_clr), v['loc']))
        self.back_surf.blits(blits_seq)
        pygame.display.flip()
        self.ui_surface_clock.tick(update_freq)

    def enable_render(self):
        self.enable_vir_render = True
        self.enable_phy_render = True

    """
    To render other contents on the screen, use these call back methods.
    Note: all color are in RGBA mode, e.g., red->(255,0,0,255). To set a transport color, set the last value a between
    [0, 255].
    
    """

    def draw_vir_circle(self, c, r, color=None):
        pos = (((c - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(int)
        pygame.gfxdraw.aacircle(self.v_surf, pos[0], pos[1], r, color if color is not None else (0, 0, 0))

    def draw_vir_circle_bg(self, c, r, color=None):
        pos = (((c - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(int)
        pygame.gfxdraw.aacircle(self.v_surf_bk, pos[0], pos[1], r, color if color is not None else (0, 0, 0))

    def draw_phy_circle(self, c, r, color=None):
        pos = (((c - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(int)
        pygame.gfxdraw.aacircle(self.p_surf, pos[0], pos[1], int(r * draw_scale * self.p_scale),
                                color if color is not None else (0, 0, 0))

    def draw_phy_circle_bg(self, c, r, color=None):
        pos = (((c - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(int)
        pygame.gfxdraw.aacircle(self.p_surf, pos[0], pos[1], int(r * draw_scale * self.p_scale),
                                color if color is not None else (0, 0, 0))

    def draw_vir_line(self, s, e, w, color=None):
        s_pos = (((s - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(int)
        e_pos = (((e - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(int)
        pygame.draw.line(self.v_surf, color if color is not None else (0, 0, 0), s_pos, e_pos, w)

    def draw_vir_line_bg(self, s, e, w, color=None):
        s_pos = (((s - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(int)
        e_pos = (((e - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(int)
        pygame.draw.line(self.v_surf_bk, color if color is not None else (0, 0, 0), s_pos, e_pos, w)

    def draw_phy_line(self, s, e, w, color=None):
        s_pos = (((s - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(int)
        e_pos = (((e - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(int)
        pygame.draw.aaline(self.p_surf, color if color is not None else (0, 0, 0), s_pos, e_pos, 1)

    def draw_phy_line_bg(self, s, e, w, color=None):
        s_pos = (((s - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(int)
        e_pos = (((e - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(int)
        pygame.draw.aaline(self.p_surf_bk, color if color is not None else (0, 0, 0), s_pos, e_pos, 2)

    def draw_vir_poly(self, vertexes, fill, color=None):
        """

        Args:
            vertexes: 顶点序列，e.g., ((x1,y1),(x2,y2),...)
            fill:
            color:

        Returns:

        """
        vertexes = (((np.array(
            vertexes) - self.v_scene.scene_center) * draw_scale + self.v_surf_size_2) * self.v_scale).astype(int)
        pygame.gfxdraw.aapolygon(self.v_surf, vertexes, color)
        pygame.gfxdraw.filled_polygon(self.v_surf, vertexes, color)

    def draw_phy_poly(self, vertexes, fill, color=None):
        """

        Args:
            vertexes: 顶点序列，e.g., ((x1,y1),(x2,y2),...)
            fill:
            color: (0~255,0~255,0~255, alpha)

        Returns:

        """

        vertexes = (((np.array(
            vertexes) - self.p_scene.scene_center) * draw_scale + self.p_surf_size_2) * self.p_scale).astype(int)
        pygame.gfxdraw.aapolygon(self.p_surf, vertexes, color)
        pygame.gfxdraw.filled_polygon(self.p_surf, vertexes, color)

    def trans_pos_from_psurf(self, pos) -> Tuple:
        x_off = (pos[0] - self.p_render_center[0]) / self.p_scale
        y_off = (-pos[1] + self.p_render_center[1]) / self.p_scale
        px = (self.p_surf_size_2[0] + x_off) / draw_scale - scene_board
        py = (self.p_surf_size_2[1] + y_off) / draw_scale - scene_board
        return px, py
