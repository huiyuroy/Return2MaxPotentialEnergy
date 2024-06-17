from core.space import *
from core.space.boundary import Boundary
from core.space.grid import Tiling


def calc_axis_x_angle(p):
    """
    计算目标点的x轴转角，逆时针为正，v=(1,0)为 0度

    Args:
        p:

    Returns:
        in degree


    """
    return math.atan2(p[1], p[0]) * RAD2DEG % 360


class Ray:
    def __init__(self):
        self.origin = (0, 0)
        self.dir_point = (0, 0)
        self.fwd = (0, 0)
        self.rot_angle = 0

        self.hit = None
        self.hit_dis = 0

        self.rela_vertex: Vertex = None
        self.pot_coll_walls: List[Segment] = []  # 潜在碰撞墙
        self.exceed_vertex = False  # 是否超越轮廓顶点，延伸向远方
        self.excluded = False


class Vertex:
    def __init__(self, data):
        self.data: Tuple = data
        self.rela_data: np.ndarray = np.array(data)  # 与当前观察点的相对位置，每次更新可见性时更新
        self.rot_angle: float = 0  # 以观察点为原点时，绕x轴转过的角度
        self.quadrant: int = 0  # 0,1,2,3,4 象限
        self.dis2obs_pos = 0
        self.start_seg: Segment = Segment()  # 以vertex为起点的segment
        self.end_seg: Segment = Segment()  # 以vertex为终点的segment
        self.excluded = False


class Segment:
    def __init__(self):
        self.id: int = 0
        self.vtx_start: Vertex = None
        self.vtx_end: Vertex = None
        self.pre: Segment = None
        self.post: Segment = None
        self.length: float = 0
        self.center: Tuple = (0, 0)
        self.out_contour = True
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0
        self.dis2obs_pos = 0
        self.lighted = False


class Observer:
    """
    Realized by Rotational sweep visibility. see:
    https://www.redblobgames.com/articles/visibility/
    https://github.com/akapkotel/light_raycasting

    Idea: Observer is a point which represents a source of light or an observer in field-of-view simulation.
    """

    def __init__(self):
        self.pos = (0, 0)  # position of the light/observer
        self.scene: DiscreteScene = None
        # The algorithm does not check against whole polygons-obstacles, but against each of their edges:
        # objects considered as blocking FOV/light. Each obstacle is a polygon consisting a list of points -
        # it's vertices.
        self.fov_fwd = (0, 0)
        self.fov_angle = 0
        self.half_fov = 85  # degree

        self.segments: List[Segment] = []
        self.vertexes: List[Vertex] = []
        self.outer_segments: List[Segment] = []
        self.inner_segments: List[Segment] = []
        self.face_segments: Set[Segment] = set()  # 面向光源的墙壁
        self.rays: Sequence[Ray] = []
        # this would be used to draw our visible/lit-up area:

    def load_walls(self):
        self.vertexes = []
        self.segments = []
        self.outer_segments = []
        self.inner_segments = []
        seg_id = 0
        for bound in self.scene.bounds:
            if bound.is_out_bound:
                b_points, _ = geo.calc_adjust_poly_order(bound.points, order=0)  # 必须全部方向调整为顺时针
            else:
                b_points, _ = geo.calc_adjust_poly_order(bound.points, order=1)  # 必须全部方向调整为逆时针
            vertexes = [Vertex(p) for p in b_points]
            segments = []

            for i in range(bound.points_num):
                s_id = i
                e_id = i + 1 if i < bound.points_num - 1 else 0
                wall = Segment()
                wall.id = seg_id
                wall_s, wall_e = tuple(b_points[s_id]), tuple(b_points[e_id])
                wall.vtx_start = vertexes[s_id]
                wall.vtx_end = vertexes[e_id]
                wall.vtx_start.start_seg = wall
                wall.vtx_end.end_seg = wall
                wall.out_contour = bound.is_out_bound
                wall.length = alg.l2_norm(np.array(wall_s) - np.array(wall_e))
                wall.center = tuple((np.array(wall_s) + np.array(wall_e)) * 0.5)
                wall.min_x = min(wall_s[0], wall_e[0])
                wall.min_y = min(wall_s[1], wall_e[1])
                wall.max_x = max(wall_s[0], wall_e[0])
                wall.max_y = max(wall_s[1], wall_e[1])
                if bound.is_out_bound:
                    self.outer_segments.append(wall)
                else:
                    self.inner_segments.append(wall)
                segments.append(wall)
                seg_id += 1
            self.vertexes.extend(vertexes)
            self.segments.extend(segments)

        for wall in self.segments:
            w_s = wall.vtx_start
            wall.pre = w_s.end_seg
            w_s.end_seg.post = wall

    def update_visible_polygon(self, pos) -> Sequence[Ray]:
        """
        Field of view or lit area is represented by polygon which is basically a list of points. Each frame list is
        updated accordingly to the position of the Observer

        Args:
            pos: point from which we will shot rays

        Returns:

        """
        self.pos = pos
        self.update_entire_relative_relation()
        vertexes = self.vertexes[::]  # [v for v in self.vertexes]
        vertexes.sort(key=lambda v: v.rot_angle)
        walls = self.segments[::]  # [s for s in self.segments]
        # sorting walls according to their distance to origin allows for faster finding rays intersections and avoiding
        # iterating through whole list of the walls:
        for w in walls:
            w_s, w_e = w.vtx_start.data, w.vtx_end.data
            w.dis2obs_pos = geo.calc_point_mindis2line(self.pos, w_s, w_e)
        walls.sort(key=lambda w: w.dis2obs_pos)
        # to avoid issue with border-walls when wall-rays are preceding obstacle-rays:
        walls.sort(key=lambda w: w.out_contour)
        # s = time.perf_counter()
        self.generate_rays_for_walls()
        # e = time.perf_counter()
        # print('find rays in {}, fps {}'.format(e - s, 1 / (e - s)))
        # s = time.perf_counter()
        self.intersect_rays_w_walls()
        # e = time.perf_counter()
        # print('obtain vis polys in {}, fps {}'.format(e - s, 1 / (e - s)))
        # need to sort rays by their ending angle again because offset_rays are unsorted and pushed at the end of the
        # list: finally, we build a visibility polygon using endpoint of each ray:

        return self.rays

    def update_entire_relative_relation(self):
        n_pos = np.array(self.pos)
        for vertex in self.vertexes:
            dis_vec = np.array(vertex.data) - n_pos
            vertex.rela_data = tuple(dis_vec)
            vertex.rot_angle = calc_axis_x_angle(vertex.rela_data)
            vertex.dis2obs_pos = alg.l2_norm(dis_vec)
            vertex.excluded = False

    def generate_rays_for_walls(self):
        """
        Create a 'ray' connecting origin with each corner (obstacle vertex) on the screen. Ray is a tuple of two (x, y) coordinates used later to
        find which segment obstructs visibility.
        TODO: find way to emit less offset rays [x][ ]
        :param origin: Tuple -- point from which 'light' is emitted
        :param corners: List -- vertices of obstacles
        :return: List -- rays to be tested against obstacles edges

        Args:
            vertexes:

        """
        self.rays = set()
        self.face_segments = set()
        for wall in self.outer_segments:
            wall.lighted = False
            w_s, w_e = wall.vtx_start, wall.vtx_end
            if geo.chk_points_clockwise((self.pos, w_s.data, w_e.data)) >= 0:  # 如果顺时针或共线，代表这面外墙正面朝着光源
                self.face_segments.add(wall)
                wall.lighted = True
        for wall in self.inner_segments:
            wall.lighted = False
            w_s, w_e = wall.vtx_start, wall.vtx_end
            if not geo.chk_points_clockwise((self.pos, w_s.data, w_e.data)) <= 0:  # 如果逆时针或共线，代表这面内墙正面朝着光源
                self.face_segments.add(wall)
                wall.lighted = True

        dropout_segs = []
        for face_seg in self.face_segments:
            ws, we = face_seg.vtx_start.data, face_seg.vtx_end.data
            for other_seg in self.face_segments:
                if face_seg.id != other_seg.id:  # 另一个面向光源的墙
                    o_ws, o_we = other_seg.vtx_start.data, other_seg.vtx_end.data
                    if geo.chk_lines_cross(self.pos, ws, o_ws, o_we) and geo.chk_lines_cross(self.pos, we, o_ws, o_we):
                        dropout_segs.append(face_seg)
                        break
        for seg in dropout_segs:
            self.face_segments.remove(seg)
        for wall in self.face_segments:
            w_s, w_e = wall.vtx_start, wall.vtx_end
            if not w_s.excluded:
                w_s.excluded = True
                self.rays.add(self.generate_vertex_ray(w_s))
            if not w_e.excluded:
                w_e.excluded = True
                self.rays.add(self.generate_vertex_ray(w_e))

        # for wall in self.face_segments:
        extend_rays = []
        for ray in self.rays:
            wall1 = ray.rela_vertex.start_seg
            wall2 = ray.rela_vertex.end_seg
            if wall1.lighted != wall2.lighted:
                r2v1 = self.generate_extend_ray(ray.rot_angle + EPS)
                r2v2 = self.generate_extend_ray(ray.rot_angle - EPS)
                extend_rays.append(r2v1)
                extend_rays.append(r2v2)
        self.rays = list(self.rays)
        self.rays.extend(extend_rays)
        # sorted by rotation angle between x-axis, which means rays sorted by counterclockwise
        self.rays.sort(key=lambda r: r.rot_angle)

    def intersect_rays_w_walls(self):
        for ray in self.rays:
            r_s, r_e = np.array(ray.origin), np.array(ray.dir_point)
            for wall in self.face_segments:
                if not geo.chk_ray_rect_AABB(ray.origin, ray.fwd, (wall.min_x, wall.min_y, wall.max_x, wall.max_y)):
                    continue
                w_s, w_e = wall.vtx_start, wall.vtx_end
                _, ray_end = geo.calc_ray_line_intersect(np.array(w_s.data), np.array(w_e.data), r_s, r_e)
                if ray_end is not None:
                    if ray.hit is None:
                        ray.hit = ray_end[1]
                        ray.hit_dis = alg.l2_norm(np.array(ray.hit) - np.array(ray.origin))
                    else:
                        ray.pot_coll_walls.append(wall)
                        hit_d = ray.hit_dis
                        end_d = alg.l2_norm(np.array(ray_end[1]) - np.array(ray.origin))
                        if end_d < hit_d:
                            ray.hit = ray_end[1]
                            ray.hit_dis = end_d
        self.rays = [r for r in self.rays if r.hit is not None]

    def generate_vertex_ray(self, vertex: Vertex):
        r2v = Ray()
        r2v.origin = self.pos
        r2v.dir_point = vertex.data
        r2v.fwd = geo.norm_vec((r2v.dir_point[0] - r2v.origin[0], r2v.dir_point[1] - r2v.origin[1]))
        r2v.rot_angle = calc_axis_x_angle(np.array(r2v.dir_point) - np.array(self.pos))
        r2v.rela_vertex = vertex
        return r2v

    def generate_extend_ray(self, angle):
        r2v = Ray()
        r2v.origin = self.pos
        r2v.dir_point = self.scaled_rot_vec(self.pos, angle, 10000)
        r2v.fwd = geo.norm_vec((r2v.dir_point[0] - r2v.origin[0], r2v.dir_point[1] - r2v.origin[1]))
        r2v.rot_angle = angle
        r2v.rela_vertex = None
        return r2v

    @staticmethod
    def scaled_rot_vec(start, angle, length):
        rad = angle * DEG2RAD
        return start[0] + math.cos(rad) * length, start[1] + math.sin(rad) * length


class Scene:
    def __init__(self):
        self.name = None
        self.scene_type = 'vir'
        self.bounds = []
        self.max_size = [0, 0]  # w,h
        self.out_bound_conv = geo.ConvexPoly()  # 外包围盒凸包-矩形
        self.out_conv_hull = geo.ConvexPoly()  # 外边界凸包
        self.scene_center = np.array([0, 0])
        self.nodes = []
        self.patches = []
        self.patches_mat = []  # 记录patch彼此关系的matrix
        self.tris = []
        self.tris_nei_ids = []
        self.conv_polys = []
        self.conv_nei_ids = []
        self.conv_area_priority = []
        self.conv_collision_priority = []
        self.conv_connection_priority = []

    def update_contours(self, name, contours_points):
        """

        Args:
            name:
            contours_points: 轮廓点集，必须是二维数组

        Returns:

        """
        self.name = name
        self.bounds = []
        for contour in contours_points:
            bound = Boundary()
            bound.is_out_bound = False
            bound.points = contour
            bound.clean_repeat()
            bound.points_num = len(bound.points)
            np_contour = np.array(bound.points).copy()
            bound.center = (np_contour.sum(axis=0) / bound.points_num).tolist()
            bound.barycenter = geo.calc_poly_barycenter(np_contour).tolist()
            bound.cir_rect = geo.calc_cir_rect(np_contour)[0].tolist()
            self.bounds.append(bound)
        out_bound = self.bounds[0]
        out_bound.is_out_bound = True
        out_bound_points = np.array(out_bound.points).copy().tolist()
        self.max_size = np.array(out_bound.cir_rect).max(axis=0)
        mx, my = self.max_size
        self.scene_center = np.array([mx * 0.5, my * 0.5])
        self.out_conv_hull = geo.ConvexPoly()
        self.out_conv_hull.generate_from_poly(geo.calc_convex_graham_scan(out_bound_points))
        self.out_bound_conv = geo.ConvexPoly()
        self.out_bound_conv.generate_from_poly([[0, 0], [mx, 0], [mx, my], [0, my]])
        print('contours: done')

    def update_segmentation(self):
        """
            1. 耳分法生成三角剖分
            2. 基于三角剖分集进行区域生长，将三角形合并为多个凸多边形。每次生长都先选择面积最大的三角形作为生长中心，尽量将区域中最大的凸多边形生成出来
            注：边界轮廓需要按照特定顺序，外轮廓必须在列表首位

            Returns:

        """
        poly_bounds = [bound.points for bound in self.bounds]
        tris = geo.calc_poly_triangulation(poly_bounds)
        tris_num = len(tris)
        tris_nei_ids = [[] for _ in range(tris_num)]
        for i in range(tris_num):
            tar_tri = tris[i]
            for j in range(tris_num - 1):
                other_i = (j + i + 1) % tris_num
                other_tri = tris[other_i]
                if tar_tri.det_common_edge(other_tri) is not None:
                    tris_nei_ids[i].append(other_i)
        conv_polys = []
        total_area = 0
        total_perimeter = 0
        visit_mat = pickle.loads(pickle.dumps(tris_nei_ids))
        tris_ids = [i for i in range(len(tris))]
        grown_deque = deque()
        while len(tris_ids) > 0:
            start_tri_id = 0
            min_area = float('inf')
            max_area = 0
            for id in tris_ids:
                t_a = geo.calc_tri_area(tris[id].vertices)

                if t_a > max_area:
                    start_tri_id = id
                    max_area = t_a
            cur_poly = np.array(tris[start_tri_id].vertices).copy().tolist()
            cur_out_edges = []
            cur_in_edges = []
            for e in tris[start_tri_id].out_edges:
                v1_id, v2_id = e
                v1 = tris[start_tri_id].vertices[v1_id]
                v2 = tris[start_tri_id].vertices[v2_id]
                e = [np.array(v1).copy().tolist(), np.array(v2).copy().tolist()]
                cur_out_edges.append(e)
            for e in tris[start_tri_id].in_edges:
                v1_id, v2_id = e
                v1 = tris[start_tri_id].vertices[v1_id]
                v2 = tris[start_tri_id].vertices[v2_id]
                e = [np.array(v1).copy().tolist(), np.array(v2).copy().tolist()]
                cur_in_edges.append(e)
            tris_ids.remove(start_tri_id)  # 选定生长中心，则从三角形集中移除当前三角形
            grown_deque.clear()
            for n_id in visit_mat[start_tri_id]:
                if n_id in tris_ids:
                    grown_deque.append(n_id)
                    if start_tri_id in visit_mat[n_id]:
                        visit_mat[n_id].remove(start_tri_id)
            while len(grown_deque) > 0:
                cur_tri_id = grown_deque.popleft()
                cur_tri = tris[cur_tri_id].vertices
                find_con = False
                for nt_i in range(-1, 2):
                    tv1 = cur_tri[nt_i]
                    tv2 = cur_tri[nt_i + 1]
                    for p_i in range(-1, len(cur_poly) - 1):
                        pv1 = cur_poly[p_i]
                        pv2 = cur_poly[p_i + 1]
                        if geo.chk_edge_same([pv1, pv2], [tv1, tv2]):  # 当前多边形与目标三角形邻接
                            find_con = True
                            temp_poly = np.array(cur_poly).copy().tolist()
                            temp_poly.insert(p_i + 1, cur_tri[(nt_i + 2) % 3])
                            temp_poly, _ = geo.calc_adjust_poly_order(temp_poly, 1)
                            if geo.chk_poly_concavity(temp_poly):  # 能够成凸多边形
                                cur_poly = temp_poly
                                common_edge = [pv1, pv2]
                                for e in tris[cur_tri_id].out_edges:
                                    v1_id, v2_id = e
                                    v1 = cur_tri[v1_id]
                                    v2 = cur_tri[v2_id]
                                    e = [np.array(v1).copy().tolist(), np.array(v2).copy().tolist()]
                                    cur_out_edges.append(e)
                                for e in tris[cur_tri_id].in_edges:
                                    v1_id, v2_id = e
                                    v1 = cur_tri[v1_id]
                                    v2 = cur_tri[v2_id]
                                    e = [np.array(v1).copy().tolist(), np.array(v2).copy().tolist()]
                                    if not geo.chk_edge_same(e, common_edge):
                                        cur_in_edges.append(e)
                                for e in cur_in_edges:
                                    if geo.chk_edge_same(e, common_edge):
                                        cur_in_edges.remove(e)
                                tris_ids.remove(cur_tri_id)
                                nei_ids = visit_mat[cur_tri_id]
                                for n_id in nei_ids:
                                    visit_mat[n_id].remove(cur_tri_id)
                                    is_in_queue = False
                                    for q_id in grown_deque:
                                        if q_id == n_id:
                                            is_in_queue = True
                                    if not is_in_queue:
                                        grown_deque.append(n_id)
                            break
                    if find_con:
                        break

            conv_poly = geo.ConvexPoly()
            conv_poly.vertices = geo.cmp_convex_vertex_order(cur_poly)
            conv_poly.center = geo.calc_convex_centroid(np.array(cur_poly))
            conv_poly.barycenter = geo.calc_poly_barycenter(np.array(cur_poly)).tolist()
            conv_poly.cir_circle = geo.calc_poly_min_cir_circle(cur_poly)
            conv_poly.in_circle = geo.calc_poly_max_in_circle(cur_poly)
            conv_poly.out_edges_perimeter = 0
            conv_poly.out_edges = []
            for e in cur_out_edges:
                v1_id = conv_poly.find_vertex_idx(e[0])
                v2_id = conv_poly.find_vertex_idx(e[1])
                conv_poly.out_edges.append([v1_id, v2_id])
                conv_poly.out_edges_perimeter += alg.l2_norm(
                    np.array(conv_poly.vertices[v1_id]) - np.array(conv_poly.vertices[v2_id]))
            conv_poly.in_edges = []
            for e in cur_in_edges:
                v1_id = conv_poly.find_vertex_idx(e[0])
                v2_id = conv_poly.find_vertex_idx(e[1])
                conv_poly.in_edges.append([v1_id, v2_id])
            conv_poly.area = geo.calc_poly_area(np.array(conv_poly.vertices))
            conv_polys.append(conv_poly)

        poly_num = len(conv_polys)
        conv_nei_ids = [[] for _ in range(poly_num)]
        conv_area_priority = [0] * poly_num
        conv_collision_priority = [0] * poly_num
        conv_connection_priority = [0] * poly_num
        for i in range(poly_num):
            tar_p = conv_polys[i]
            total_area += tar_p.area
            total_perimeter += tar_p.out_edges_perimeter

            for j in range(poly_num - 1):
                other_i = (j + i + 1) % poly_num
                other_p = conv_polys[other_i]
                if tar_p.det_common_edge(other_p) is not None:
                    conv_nei_ids[i].append(other_i)
        for i in range(poly_num):
            tar_p = conv_polys[i]
            conv_area_priority[i] = tar_p.area / total_area
            conv_collision_priority[i] = 1 - tar_p.out_edges_perimeter / total_perimeter
            conv_connection_priority[i] = len(conv_nei_ids[i]) / (2 * poly_num)

        self.tris = tris
        self.tris_nei_ids = tris_nei_ids
        self.conv_polys = conv_polys
        self.conv_nei_ids = conv_nei_ids
        self.conv_area_priority = conv_area_priority
        self.conv_collision_priority = conv_collision_priority
        self.conv_connection_priority = conv_connection_priority
        print("segmentation: done")


class DiscreteScene(Scene):

    def __init__(self):
        super().__init__()
        self.tiling_h_width: float = 0
        self.tiling_h_diag = 0
        self.tilings: Sequence[Tiling] = []
        self.tilings_shape: Tuple = ()  # tilings 行列分布 [w,h]
        self.tiling_w: float = GRID_WIDTH  # tiling块的宽度大小，以m计算
        self.tiling_w_inv: float = 1 / GRID_WIDTH
        self.tiling_offset: np.ndarray = np.array([0, 0])
        self.tilings_data: np.ndarray = np.array([])
        self.tilings_weights: np.ndarray = np.array([])
        self.tilings_weights_grad: np.ndarray = np.array([])  # 梯度方向为weight下降方向，要用需要取反
        self.tilings_nei_ids = []
        self.tilings_walkable = []
        self.tilings_visitable = []

        self.tilings_rot_occupancy_num = 360
        self.tilings_rot_occupancy_grids = None

        self.vispoly_observer: Observer = None

    # belows are offline calculations, used for program boosting.
    def update_grids_base_attr(self):
        """
        generate base attributes of scene grids, including:
            - tiling width
            - tiling offset to the scene center
            - tiling number shape: (w, h) w: horizontal number of tilings, h: vertical number of tilings
            - all tilings, each tiling contain: id, idx_loc, center, xy min and max, nei_ids


        Returns:

        """
        m_wd, m_wh = self.max_size
        w, h = math.ceil(m_wd / self.tiling_w), math.ceil(m_wh / self.tiling_w)
        self.tiling_h_width = self.tiling_w * 0.5
        self.tiling_h_diag = self.tiling_h_width * (2 ** 0.5)
        self.tiling_offset = np.array([(m_wd - w * self.tiling_w) * 0.5, (m_wh - h * self.tiling_w) * 0.5])
        self.tilings_shape = (w, h)
        self.tilings = [None] * (w * h)
        nei_offset = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        count = 0
        for i in range(h):
            for j in range(w):
                tiling = Tiling()
                tiling.id = i * w + j
                tiling.mat_loc = (i, j)
                tiling.center = np.array([(j + (1 - w) * 0.5) * self.tiling_w + 0.5 * m_wd,
                                          (i + (1 - h) * 0.5) * self.tiling_w + 0.5 * m_wh])
                tiling.rect = ((tiling.center[0] - self.tiling_h_width, tiling.center[1] - self.tiling_h_width),
                               (tiling.center[0] + self.tiling_h_width, tiling.center[1] - self.tiling_h_width),
                               (tiling.center[0] + self.tiling_h_width, tiling.center[1] + self.tiling_h_width),
                               (tiling.center[0] - self.tiling_h_width, tiling.center[1] + self.tiling_h_width))
                tiling.x_min, tiling.x_max = tiling.center[0] - self.tiling_h_width, tiling.center[
                    0] + self.tiling_h_width
                tiling.y_min, tiling.y_max = tiling.center[1] - self.tiling_h_width, tiling.center[
                    1] + self.tiling_h_width
                nei_ids = []
                for off in nei_offset:
                    row, col = i + off[0], j + off[1]
                    if 0 <= row < h and 0 <= col < w:
                        nei_ids.append(row * w + col)
                tiling.nei_ids = tuple(nei_ids)
                self.tilings[tiling.id] = tiling
                count += 1
                print('\rtilings generate: {}%'.format(count / (w * h) * 100), end='')
        print()
        count = 0
        for tiling in self.tilings:
            tiling.rela_patch = None
            tiling.type = 0
            find_intersect = False
            if self.patches is not None and len(self.patches) > 0:
                for patch in self.patches:
                    for i in range(patch.nodes_num - 1):
                        line_s = patch.nodes[i]
                        line_e = patch.nodes[i + 1]
                        l_pos = [line_s.pos[0], line_s.pos[1], line_e.pos[0], line_e.pos[1]]
                        if geo.chk_line_rect_cross(l_pos, tiling.rect):
                            tiling.rela_patch = patch.id
                            find_intersect = True
                            break
                    if find_intersect:
                        break
            if not geo.chk_square_bound_cross(tiling.center, self.tiling_h_width * 2, self.bounds) and \
                    geo.chk_p_in_bound(tiling.center, self.bounds, 0):
                tiling.type = 1
            else:
                tiling.cross_bound = geo.calc_square_bound_cross(tiling.center, self.tiling_h_width * 2, self.bounds)
            tiling.corr_conv_ids = []
            tiling.corr_conv_inters = []
            tiling.corr_conv_cin = -1
            for i in range(len(self.conv_polys)):
                conv = self.conv_polys[i]
                inter_poly = geo.calc_con_polys_intersect_simple(conv.vertices, tiling.rect)
                if inter_poly:
                    tiling.corr_conv_ids.append(i)
                    tiling.corr_conv_inters.append(np.array(inter_poly))
                if geo.chk_p_in_conv(tiling.center, conv.vertices, conv.in_circle):
                    tiling.corr_conv_cin = i
            count += 1
            print('\rtiling attribute building: {}%'.format(count / len(self.tilings) * 100), end='')
        print()

    def update_grids_visibility(self):
        count = 0
        self.enable_visibility()
        for tiling in self.tilings:
            if tiling.type or tiling.cross_bound.shape[0] > 0:
                tiling.vis_tri, _ = self.update_visibility(tiling.center)
            count += 1
            print('\rscene vis building: {}%'.format(count / len(self.tilings) * 100), end='')
        print()

    def update_grids_weights(self):
        count = 0
        w, h = self.tilings_shape
        flat_sig = 0.1
        flat_weights = np.zeros((h, w))
        for tiling in self.tilings:
            r, c = tiling.mat_loc
            if not tiling.type:
                tiling.nearst_obs_grid_id = tiling.id
                tiling.flat_weight = 0
            else:
                min_obs_dist = float('inf')
                for ntiling in self.tilings:
                    if ntiling.id == tiling.id or ntiling.type:
                        continue
                    else:
                        dist = alg.l2_norm(tiling.center - ntiling.center)
                        if dist < min_obs_dist:
                            min_obs_dist = dist
                            tiling.nearst_obs_grid_id = ntiling.id
                nearst = self.tilings[tiling.nearst_obs_grid_id]
                dist_square = alg.l2_norm_square((nearst.center - tiling.center) * 0.01)  # cm to m
                tiling.flat_weight = np.exp(-flat_sig / dist_square)
            flat_weights[r, c] = tiling.flat_weight
            count += 1
            print('\rtiling weight building: {}%'.format(count / len(self.tilings) * 100), end='')
        print()

        flat_weights = img.blur_image(flat_weights, h, w)
        flat_grads = img.calc_img_grad(flat_weights, h, w)
        for i in range(h):
            for j in range(w):
                tiling = self.tilings[i * w + j]
                tiling.flat_weight = flat_weights[i, j]
                tiling.flat_grad = flat_grads[i, j]

        self.tilings_weights = flat_weights
        self.tilings_weights_grad = flat_grads

    def update_grids_rot_occupancy(self, enable_vis_range_precompute=False):
        """


        Returns:

        """
        count = 0
        total_c = len(self.tilings) * 2
        self.enable_visibility()
        w, h = self.tilings_shape
        for tiling in self.tilings:
            r, c = tiling.mat_loc
            tiling.sur_grids_ids = []
            tiling.sur_obs_grids_ids = []
            tiling.sur_obs_bound_grids_ids = []
            radius_idx = math.ceil(HUMAN_STEP / self.tiling_w)
            sur1, sur2 = self.calc_tiling_diffusion((c, r), (0, 1), 360, radius_idx, 0)
            sur_tiling_ids = sur1 + sur2
            for sur_id_c, sur_id_r in sur_tiling_ids:
                sur_id = sur_id_r * w + sur_id_c
                sur_tiling = self.tilings[sur_id]
                tiling.sur_grids_ids.append(sur_id)
                if not sur_tiling.type:
                    tiling.sur_obs_grids_ids.append(sur_id)
                    if sur_tiling.cross_bound.shape[0] > 0:
                        tiling.sur_obs_bound_grids_ids.append(sur_id)
            tiling.sur_grids_ids = tuple(tiling.sur_grids_ids)
            tiling.sur_obs_grids_ids = tuple(tiling.sur_obs_grids_ids)
            tiling.sur_obs_bound_grids_ids = tuple(tiling.sur_obs_bound_grids_ids)
            for sur_id in tiling.sur_grids_ids:
                sur_grid = self.tilings[sur_id]
                if not sur_grid.type:
                    tiling.sur_occu_safe = False
                    break
            count += 1
            print("\rsurround visibility precompute processed in {}%".format(count / total_c * 100), end="")

        if enable_vis_range_precompute:
            self.tilings_visitable = [1 if t.type or t.cross_bound.shape[0] > 0 else 0 for t in self.tilings]

            for tiling in self.tilings:
                count += 1
                tiling.sur_360_partition = [[] for _ in range(360)]
                if tiling.type or tiling.cross_bound.shape[0] > 0:
                    vis_tris, vis_grids = self.update_visibility(tiling.center, fwd=(0, 1), fov=360, grids_comp=True,
                                                                 realtime_comp=True)
                    tiling.sur_360_partition = vis_grids
                print("\rsurround visibility precompute processed in {}%".format(count / total_c * 100), end="")

        print("\rsurround visibility precompute processed in done")

    # online calculations, called in env prepare, including base attributes calculation or other specific rdw
    # controllers preparation.
    def update_grids_runtime_attr(self):
        """
        calculate the runtime attributes of grids, used for supply offline setups

        Returns:

        """
        for bound in self.bounds:
            if bound.is_out_bound:
                bound.points, _ = geo.calc_adjust_poly_order(bound.points, order=0)  # 必须全部方向调整为顺时针
            else:
                bound.points, _ = geo.calc_adjust_poly_order(bound.points, order=1)  # 必须全部方向调整为逆时针

        m_wd, m_wh = self.max_size
        self.tiling_h_width = self.tiling_w * 0.5
        self.tiling_h_diag = self.tiling_h_width * (2 ** 0.5)
        self.tilings_data = []
        self.tilings_nei_ids = [[] for _ in range(len(self.tilings))]
        self.tilings_walkable = []
        self.tilings_visitable = np.zeros(len(self.tilings))
        w, h = self.tilings_shape
        flat_weights = np.zeros((h, w))
        flat_grad = np.zeros((h, w, 2))
        max_f_grad_len, max_s_grad_len = 0, 0
        nei_offset = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        vis_norm_fwd = (0, 1)
        for tiling in self.tilings:
            i, j = tiling.mat_loc
            tiling.center = np.array([(j + (1 - w) * 0.5) * self.tiling_w + 0.5 * m_wd,
                                      (i + (1 - h) * 0.5) * self.tiling_w + 0.5 * m_wh])
            tiling.rect = ((tiling.center[0] - self.tiling_h_width, tiling.center[1] - self.tiling_h_width),
                           (tiling.center[0] + self.tiling_h_width, tiling.center[1] - self.tiling_h_width),
                           (tiling.center[0] + self.tiling_h_width, tiling.center[1] + self.tiling_h_width),
                           (tiling.center[0] - self.tiling_h_width, tiling.center[1] + self.tiling_h_width))
            tiling.x_min, tiling.x_max = tiling.center[0] - self.tiling_h_width, tiling.center[0] + self.tiling_h_width
            tiling.y_min, tiling.y_max = tiling.center[1] - self.tiling_h_width, tiling.center[1] + self.tiling_h_width
            nei_ids = []
            for off in nei_offset:
                row, col = i + off[0], j + off[1]
                if 0 <= row < h and 0 <= col < w:
                    nei_ids.append(row * w + col)
            tiling.nei_ids = tuple(nei_ids)

            self.tilings_data.append(tiling.center)
            self.tilings_nei_ids[tiling.id].extend(tiling.nei_ids)

            tiling.vis_rays = []
            tiling.vis_360angle_partition = [[] for _ in range(360)]
            tc = tiling.center
            for idx, tri in enumerate(tiling.vis_tri):
                _, s_rp, e_rp = tri
                ray = Ray()
                ray.hit = s_rp
                ray.rot_angle = calc_axis_x_angle(np.array(ray.hit) - tc)
                tiling.vis_rays.append(ray)

                s_ang = math.floor(geo.calc_angle_bet_vec(np.array(s_rp) - tc, vis_norm_fwd) * RAD2DEG + 180) % 360
                e_ang = math.ceil(geo.calc_angle_bet_vec(np.array(e_rp) - tc, vis_norm_fwd) * RAD2DEG + 180) % 360

                if s_ang <= e_ang:
                    ang_idxes = tuple(range(s_ang, e_ang + 1))
                else:
                    ang_idxes = list(range(s_ang, 360))
                    ang_idxes.extend(range(0, e_ang + 1))
                    ang_idxes = tuple(ang_idxes)

                for a_idx in ang_idxes:
                    tiling.vis_360angle_partition[a_idx].append(idx)
            tiling.vis_rays = tuple(tiling.vis_rays)

            if tiling.type or tiling.cross_bound.shape[0] > 0:
                self.tilings_walkable.append(tiling)
                self.tilings_visitable[tiling.id] = 1
            tiling.sur_vis_occu_ids = np.array([t_id for vis_g in tiling.sur_360_partition for t_id in vis_g])

            r, c = tiling.mat_loc
            flat_weights[r, c] = tiling.flat_weight
            flat_grad[r, c] = tiling.flat_grad
            dir_f_len = alg.l2_norm(flat_grad[r, c])
            if dir_f_len > max_f_grad_len:
                max_f_grad_len = dir_f_len
        flat_grad = flat_grad / max_f_grad_len
        self.tilings_weights = flat_weights
        self.tilings_weights_grad = flat_grad
        self.tilings_data = np.array(self.tilings_data)

    def calc_r2mpe_precomputation(self):
        """
        如采用R2mpe，必须在创建物理场景后调用

        Returns:

        """
        total_count = len(self.tilings)
        count = 0
        th = (HUMAN_STEP * 3) ** 2
        for tiling in self.tilings:
            setattr(tiling, 'r2mpe_360_partition', [])
            setattr(tiling, 'r2mpe_vis_occu_ids', [])
            count += 1
            print('\rprocessing r2mpe visibility precompute: {}%'.format(count / total_count * 100), end='')
            if tiling.type or tiling.cross_bound.shape[0] > 0:
                for g_ids in tiling.sur_360_partition:
                    r2mpe_grid_part = []
                    for gi in g_ids[::2]:  # 全部都检测太多，隔一个点取一个点，降低计算复杂度
                        if alg.l2_norm_square(self.tilings[gi].center - tiling.center) <= th:
                            r2mpe_grid_part.append(gi)
                    tiling.r2mpe_360_partition.append(r2mpe_grid_part)
                tiling.r2mpe_vis_occu_ids = np.array([t_id for vis_g in tiling.r2mpe_360_partition for t_id in vis_g])
        print()

    def calc_user_tiling(self, pos) -> Tuple[Tiling, geo.ConvexPoly]:
        xc, yc = ((pos - self.tiling_offset) // self.tiling_w).astype(np.int32)
        cur_tiling = self.tilings[yc * self.tilings_shape[0] + xc]  # 对应虚拟网格的id
        corr_conv_len = len(cur_tiling.corr_conv_ids)
        cur_conv = None
        if corr_conv_len == 0:
            cur_conv = None
        elif corr_conv_len == 1:
            cur_conv = self.conv_polys[cur_tiling.corr_conv_ids[0]]
        else:
            for i in range(corr_conv_len):
                if geo.chk_p_in_conv_simple(pos, cur_tiling.corr_conv_inters[i]):
                    cur_conv = self.conv_polys[cur_tiling.corr_conv_ids[i]]
                    break
        return cur_tiling, cur_conv

    def enable_visibility(self):
        self.vispoly_observer = Observer()
        self.vispoly_observer.scene = self
        self.vispoly_observer.load_walls()

    def update_visibility(self, pos, fwd=None, fov=360, grids_comp=False, realtime_comp=True) -> Tuple[
        Tuple, List[List[int]]]:
        """

        Args:
            pos:
            fwd:
            fov: 可见区域左右两边视线夹角 in degree
            grids_comp:计算可见区域内离散网格点，当采用离线计算可见区域时，此设置无意义，算法直接返回存储的可见区域离散网格点；反之，根据设置
            实时计算处在可见区域内的所有网格点

            realtime_comp: 采用实时计算可见性，反之使用离线计算的可见区域（基于网格图）

        Returns:

            vis_tris: 可见三角形，每个三角形第一个点一定是可见原点，整体按照逆时针排序

            vis_grids: 以1度划分360个扇区，记录每个扇区中可见的采样点的id，当grids_comp启用时返回实际值。反之返回空

        """
        pos = np.array(pos)
        xc, yc = ((pos - self.tiling_offset) // self.tiling_w).astype(np.int32)
        tiling = self.tilings[yc * self.tilings_shape[0] + xc]
        vis_grids = []
        if realtime_comp:
            vis_rays = self.vispoly_observer.update_visible_polygon(pos)
            vis_rays, vis_tris = self.__calc_fov_visibility(pos, fwd, fov, vis_rays)
            if grids_comp:
                h_fov = fov * DEG2RAD * 0.5
                if fov == 360:  # 如果视野范围是360度，则将朝向标准化为(0,1)，目的是用于预计算可见区域时的方向标准化
                    fwd = (0, 1)
                # 以可见原点提供360个旋转分区，每个分区记录该分区可能跨越的可见三角的下标。每个分区记录1度圆心角的扇形与fwd的夹角，本质是从-180度
                # 开始，到180度结束。0号位置表示-180~-179
                ang_partition = [[] for _ in range(360)]
                vis_grids = [[] for _ in range(360)]
                # 记录当前三角形是否在其射线间边后面有隐藏边，若有，处于射线夹角内的点还需要检测是否处于三角形内，若没有，则必定在三角形内
                tris_hidden_back = []
                npvis_tris = []
                for idx, tri in enumerate(vis_tris):
                    sray, eray = tri
                    np_s, np_e = np.array(sray.hit), np.array(eray.hit)
                    npvis_tris.append(np.array((pos, np_s, np_e)))

                    s_ang = math.floor(geo.calc_angle_bet_vec(np_s - pos, fwd) * RAD2DEG + 180) % 360
                    e_ang = math.ceil(geo.calc_angle_bet_vec(np_e - pos, fwd) * RAD2DEG + 180) % 360

                    if s_ang <= e_ang:
                        ang_idxes = tuple(range(s_ang, e_ang + 1))
                    else:
                        ang_idxes = list(range(s_ang, 360))
                        ang_idxes.extend(range(0, e_ang + 1))
                        ang_idxes = tuple(ang_idxes)

                    for a_idx in ang_idxes:
                        ang_partition[a_idx].append(idx)
                    tris_hidden_back.append(len(sray.pot_coll_walls) > 0 or len(eray.pot_coll_walls) > 0)
                dfs_deque = deque([(tiling, 0)])
                tiling_visitable = self.tilings_visitable.copy()
                while len(dfs_deque) > 0:
                    cur_t, ang_idx = dfs_deque.popleft()
                    vis_grids[ang_idx].append(cur_t.id)
                    for ni in cur_t.nei_ids:
                        if tiling_visitable[ni]:
                            n_tiling = self.tilings[ni]
                            n_dir = n_tiling.center - pos
                            n_ang = geo.calc_angle_bet_vec(n_dir, fwd)
                            possible_in = True if abs(n_ang) <= h_fov else False
                            if possible_in:
                                ang_idx = math.floor(n_ang * RAD2DEG + 180) % 360
                                tri_ids = ang_partition[ang_idx]
                                if len(tri_ids) == 1:
                                    if not tris_hidden_back[tri_ids[0]]:
                                        dfs_deque.append((n_tiling, ang_idx))
                                    elif geo.chk_p_in_conv_simple(n_tiling.center, npvis_tris[tri_ids[0]]):
                                        dfs_deque.append((n_tiling, ang_idx))
                                else:
                                    for tri_id in tri_ids:
                                        if geo.chk_p_in_conv_simple(n_tiling.center, npvis_tris[tri_id]):
                                            dfs_deque.append((n_tiling, ang_idx))
                                            break
                            tiling_visitable[ni] = 0
        else:
            vis_rays = tiling.vis_rays
            pos = tiling.center
            vis_rays, vis_tris = self.__calc_fov_visibility(pos, fwd, fov, vis_rays)
            if grids_comp:
                if fov == 360:
                    vis_grids = tiling.sur_360_partition
                else:
                    fwd = (0, 1)
                    vis_grids = [[] for _ in range(360)]
                    s_ray = vis_rays[0]
                    e_ray = vis_rays[-1]
                    s_idx = math.floor(geo.calc_angle_bet_vec(np.array(s_ray.hit) - pos, fwd) * RAD2DEG + 180) % 360
                    e_idx = math.ceil(geo.calc_angle_bet_vec(np.array(e_ray.hit) - pos, fwd) * RAD2DEG + 180) % 360
                    if s_idx <= e_idx:
                        ang_idxes = tuple(range(s_idx, e_idx))
                    else:
                        ang_idxes = list(range(s_idx, 360))
                        ang_idxes.extend(range(0, e_idx))
                        ang_idxes = tuple(ang_idxes)

                    for bound_idx in (0, -1):
                        test_tri = (pos, vis_tris[bound_idx][0].hit, vis_tris[bound_idx][1].hit)
                        a_idx = ang_idxes[bound_idx]
                        for t_id in tiling.sur_360_partition[a_idx]:
                            if geo.chk_p_in_conv_simple(self.tilings[t_id].center, np.array(test_tri)):
                                vis_grids[a_idx].append(t_id)

                    for a_idx in ang_idxes[1:len(ang_idxes) - 1]:
                        vis_grids[a_idx] = tiling.sur_360_partition[a_idx]
        vis_tris = tuple((tuple(pos), tri[0].hit, tri[1].hit) for tri in vis_tris)
        return vis_tris, vis_grids

    @staticmethod
    def __calc_fov_visibility(pos, fwd, fov, vis_rays: Sequence[Ray]) -> tuple[tuple[Ray], tuple[tuple[Ray, Ray], ...]]:
        """

        Args:
            pos:
            fwd:
            fov:
            vis_rays:

        Returns:
            vis_vertexes: 可见区域的外轮廓点
            vis_triangles:可见三角形，记录三角形两边（沿可见原点发出），整体按照逆时针排序，三角形按照射线与x轴夹角排序（逆时针）
        """
        if fov < 360 and fwd is not None:
            vis_rays = list(vis_rays)
            h_fov = fov * DEG2RAD * 0.5
            ray_r = geo.rot_vecs(fwd, h_fov)  # 最右侧视线
            ray_l = geo.rot_vecs(fwd, -h_fov)  # 最左侧视线
            r_find, l_find = False, False
            r_pos, l_pos = None, None
            side_rays = []
            for vi in range(-1, len(vis_rays) - 1):
                if not r_find:
                    inter_r_type, inter_r = geo.calc_ray_line_intersect(np.array(vis_rays[vi].hit),
                                                                        np.array(vis_rays[vi + 1].hit),
                                                                        pos,
                                                                        pos + ray_r)
                    if inter_r_type:
                        r_pos = inter_r[1]
                        r_find = True
                if not l_find:
                    inter_l_type, inter_l = geo.calc_ray_line_intersect(np.array(vis_rays[vi].hit),
                                                                        np.array(vis_rays[vi + 1].hit),
                                                                        pos,
                                                                        pos + ray_l)
                    if inter_l_type:
                        l_pos = inter_l[1]
                        l_find = True

                if r_find and l_find:
                    break
            for side_pos in (r_pos, l_pos):
                side_ray = Ray()
                side_ray.rot_angle = calc_axis_x_angle(side_pos - pos)
                side_ray.hit = tuple(side_pos)
                vis_rays.append(side_ray)
                side_rays.append(side_ray)
            vis_rays.sort(key=lambda r: r.rot_angle)
            start_idx = vis_rays.index(side_rays[0])
            end_idx = vis_rays.index(side_rays[1])
            if start_idx < end_idx:
                vis_rays = vis_rays[start_idx:end_idx + 1:1]
            else:
                vis_rays = vis_rays[start_idx::] + vis_rays[:end_idx + 1]
            tri_range = range(len(vis_rays) - 1)
        else:
            tri_range = range(-1, len(vis_rays) - 1)
        vis_tris = []
        for ti in tri_range:
            tri_s = vis_rays[ti]
            tri_e = vis_rays[ti + 1]
            vis_tris.append((tri_s, tri_e))
        return tuple(vis_rays), tuple(vis_tris)  # vis_vertexes逆时针排序，所以vis_tris里三角形顶点排序也是逆时针

    def calc_tiling_diffusion(self, tiling_idx: Tuple, init_fwd, fov=120, depth=6, rot_theta=0) -> Tuple[Tuple, Tuple]:
        """
        计算以像素为单位，fov视角范围内，depth深度内某像素的邻域像素，返回结果以最近向最外，逐层扩散的方式返回邻域像素。例：
            fov = 120
            depth = 6
            rot_theta = 0

            - - - - - p - - - - -
            - - p p p p p p p - -
            - p p p p p p p p p -
            - - p p p p p p p - -
            - - - - p p p - - - -
            - - - - - c - - - - -
            - - - - - - - - - - -
            - - - - - - - - - - -

            其中，p是范围内像素，-是范围外像素

        Args:
            tiling_idx: 中心tiling的行列号（w,h）
            fov:
            init_fwd: 初始正方向
            pixel_width: 像素宽度
            depth: 可观测到的最深像素深度（中心像素depth-1范围）
            rot_theta: 正方向旋转角（角度制），默认以[0,1] 为正方向

        Returns: 左右两个视野半区的tiling编号，每个编号(col,row)

        """

        fov = fov * DEG2RAD
        h_fov = fov * 0.5
        fwd = geo.rot_vecs(init_fwd, rot_theta)
        range_layer1 = [tiling_idx]
        range_layer2 = []
        for i in range(1, depth):
            rans = ((-i, i + 1, 1), (1 - i, i, 1), (i, -i - 1, -1), (i - 1, -i, -1))
            for ri, ran in enumerate(rans):
                for k in range(*ran):
                    if ri == 0:
                        idx = (-i, k)
                    elif ri == 1:
                        idx = (k, i)
                    elif ri == 2:
                        idx = (i, k)
                    else:
                        idx = (k, -i)
                    col = idx[0] + tiling_idx[0]
                    row = idx[1] + tiling_idx[1]
                    if 0 <= col < self.tilings_shape[0] and 0 <= row < self.tilings_shape[1]:
                        tar_vec = np.array(idx)
                        if alg.l2_norm(tar_vec) <= (depth - 1):
                            tar_ang = geo.calc_angle_bet_vec(tar_vec, fwd)
                            if abs(tar_ang) <= h_fov:
                                if tar_ang >= 0:
                                    range_layer1.append((col, row))
                                else:
                                    range_layer2.append((col, row))

        return tuple(range_layer1), tuple(range_layer2)

    def calc_diffusion_dist(self, user_loc, user_fwd, enable_obs_coff=True):
        cur_tiling, _ = self.calc_user_tiling(user_loc)
        sur_tiling_centers = self.tilings_data.copy()
        sur_tiling_probs = np.zeros(len(self.tilings))
        for sur_tiling in self.tilings:
            sur_vec = sur_tiling.center - user_loc
            sur_vel = alg.l2_norm(sur_vec)
            sur_prob = 0
            if sur_tiling.type:
                theta = geo.calc_angle_bet_vec(sur_vec, user_fwd)
                # 将[-pi,pi]作为99%置信区间，防止角度大时能量太小 https://blog.csdn.net/kaede0v0/article/details/113790060
                # rot_prob = np.exp(-0.5 * (theta / np.pi) ** 2 * 4) / ((2 * np.pi) ** 0.5)
                # 将3*human step作为99%置信区间 https://blog.csdn.net/kaede0v0/article/details/113790060
                # mov_prob = np.exp(-0.5 * (sur_vel / self.human_step_single) ** 2 * 9 / 4) / ((2 * np.pi) ** 0.5)
                # sur_prob = rot_prob * mov_prob
                sur_prob = np.exp(
                    -(4.5 * (theta * REV_PI) ** 2 + 0.5 * (sur_vel * REV_HUMAN_STEP) ** 2)) * REV_PI_2
                obs_coff = 1
                if enable_obs_coff and sur_vel > 0 and len(cur_tiling.sur_obs_grids_ids) > 0:
                    norm_vec = sur_vec / sur_vel
                    obs_coff = float('inf')
                    for obs_id in cur_tiling.sur_obs_bound_grids_ids:
                        obs_tiling = self.tilings[obs_id]
                        obs_vec = user_loc - obs_tiling.center
                        obs_d = alg.l2_norm(obs_vec)
                        epsilon = ((obs_d + np.dot(norm_vec, obs_vec)) / obs_d * 0.5) ** (HUMAN_STEP / obs_d)
                        if epsilon < obs_coff:
                            obs_coff = epsilon
                sur_prob *= obs_coff
                sur_prob *= sur_tiling.flat_weight
            sur_tiling_probs[sur_tiling.id] = sur_prob

        return sur_tiling_centers, sur_tiling_probs
