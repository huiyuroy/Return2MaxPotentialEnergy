import numpy as np

from lib.math import *
import lib.math.algebra as alg
from typing import List, Tuple, Set, Sequence, Dict, Optional, Iterator


class Triangle:

    def __init__(self):
        self.vertices = []
        self.barycenter = []
        self.in_circle = []
        self.out_edges = []
        self.in_edges = []

    def set_points(self, p1, p2, p3):
        self.vertices = [p1, p2, p3]
        self.barycenter = [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3]
        self.in_circle = [0, 0]
        self.out_edges = []
        self.in_edges = []

    def det_common_edge(self, other_t):
        for i in range(-1, len(self.vertices) - 1):
            e = [self.vertices[i], self.vertices[i + 1]]
            for j in range(-1, len(other_t.vertices) - 1):
                o_e = [other_t.vertices[j], other_t.vertices[j + 1]]
                if chk_edge_same(e, o_e):
                    return e
        return None

    def find_vertex_idx(self, v):
        for i in range(len(self.vertices)):
            if chk_p_same(self.vertices[i], v):
                return i

    def clone(self):
        p1, p2, p3 = self.vertices
        cp1, cp2, cp3 = p1[:], p2[:], p3[:]
        c_tri = Triangle()
        c_tri.set_points(cp1, cp2, cp3)
        c_tri.vertices = pickle.loads(pickle.dumps(self.vertices))
        c_tri.barycenter = pickle.loads(pickle.dumps(self.barycenter))
        c_tri.in_circle = pickle.loads(pickle.dumps(self.in_circle))
        c_tri.out_edges = pickle.loads(pickle.dumps(self.out_edges))
        c_tri.in_edges = pickle.loads(pickle.dumps(self.in_edges))
        return c_tri


class ConvexPoly:

    def __init__(self):
        self.vertices = None
        self.center = np.array([0, 0])
        self.barycenter = None
        self.cir_circle = []
        self.in_circle = []
        self.cir_rect = []
        self.out_edges = []
        self.in_edges = []
        self.area = 0
        self.out_edges_perimeter = 0

    def generate_from_poly(self, poly_points):
        self.vertices = cmp_convex_vertex_order(poly_points)
        self.center = calc_convex_centroid(np.array(poly_points))
        self.barycenter = calc_poly_barycenter(np.array(poly_points)).tolist()
        self.cir_circle = calc_poly_min_cir_circle(poly_points)
        self.in_circle = calc_poly_max_in_circle(poly_points)

    def det_common_edge(self, other_p):
        for i in range(-1, len(self.vertices) - 1):
            e = [self.vertices[i], self.vertices[i + 1]]
            for j in range(-1, len(other_p.vertices) - 1):
                o_e = [other_p.vertices[j], other_p.vertices[j + 1]]
                if chk_edge_same(e, o_e):
                    return e
        return None

    def find_vertex_idx(self, v):
        for i in range(len(self.vertices)):
            if chk_p_same(self.vertices[i], v):
                return i

    def calc_intersection2other(self, other_conv):
        c1_c, c1_r = self.cir_circle
        c2_c, c2_r = other_conv.cir_circle
        if alg.l2_norm(np.array(c1_c) - np.array(c2_c)) < (c1_r + c2_r):
            return calc_con_polys_intersect(self.vertices, self.in_circle, other_conv.vertices, other_conv.in_circle)
        else:
            return None

    def check_intersection2other(self, o_conv):
        c1_c, c1_r = self.cir_circle
        c2_c, c2_r = o_conv.cir_circle
        if alg.l2_norm(np.array(c1_c) - np.array(c2_c)) < (c1_r + c2_r):
            return chk_convs_intersect(self.vertices, self.in_circle, o_conv.vertices, o_conv.in_circle)
        else:
            return False

    def clone(self):
        c_poly = ConvexPoly()
        c_poly.vertices = pickle.loads(pickle.dumps(self.vertices))
        c_poly.in_circle = pickle.loads(pickle.dumps(self.in_circle))
        c_poly.center = self.center.copy()
        c_poly.barycenter = pickle.loads(pickle.dumps(self.barycenter))
        c_poly.out_edges = pickle.loads(pickle.dumps(self.out_edges))
        c_poly.in_edges = pickle.loads(pickle.dumps(self.in_edges))
        return c_poly


def rot_vecs(v, ang) -> np.ndarray:
    """
    rotate around (0,0)


    Args:
        v:
        ang: in radian, +clockwise, -anti_clockwise

    Returns: 旋转后向量

    """

    sin_t, cos_t = np.sin(ang), np.cos(ang)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return np.dot(v, rot_mat)


def norm_vec(v):
    v_o = np.array(v)
    v_l = (v_o[0] ** 2 + v_o[1] ** 2) ** 0.5
    if v_l == 0:
        return np.array([0, 0])
    return np.divide(v_o, v_l)


def calc_tri_area(tri):
    p1x, p1y = tri[0]
    p2x, p2y = tri[1]
    p3x, p3y = tri[2]
    return (p1x * (p2y - p3y) + p2x * (p3y - p1y) + p3x * (p1y - p2y)) * 0.5


def calc_poly_area(poly):
    row, col = poly.shape
    index = range(row - 2)
    area_sum = 0
    for i in index:
        area_sum += calc_tri_area(poly[[0, i + 1, i + 2]])
    return abs(area_sum)


def calc_poly_barycenter(poly):
    row, col = poly.shape
    index = range(row - 2)
    area_sum = 0
    center = np.array([0, 0])
    for i in index:
        tri_p = poly[[0, i + 1, i + 2]]
        tri_a = calc_tri_area(tri_p)
        area_sum += tri_a
        tri_c = np.sum(tri_p, axis=0)
        center = center + tri_c * tri_a

    return np.divide(center, 3 * area_sum)


def calc_convex_centroid(convex_set):
    row, col = convex_set.shape
    if row < 2:
        return None
    else:
        return np.divide(np.sum(convex_set, axis=0), row)


def calc_angle_bet_vec(v_base, v_target):
    if (v_base[0] == 0 and v_base[1] == 0) or (v_target[0] == 0 and v_target[1] == 0):
        return 0
    vbtan = np.arctan2(v_base[1], v_base[0])
    vttan = np.arctan2(v_target[1], v_target[0])
    ang_base = vbtan if vbtan > 0 else vbtan + PI_2
    ang_tar = vttan if vttan > 0 else vttan + PI_2
    turn_ang = ang_base - ang_tar
    if turn_ang > PI:
        turn_ang -= PI_2
    elif turn_ang < -PI:
        turn_ang += PI_2
    return turn_ang


def calc_turn_angle_in_gain(velocity_v, gain):
    return (velocity_v[0] ** 2 + velocity_v[1] ** 2) ** 0.5 * gain * RAD2DEG


def calc_turn_angle(velocity_v, radius):
    if radius == -1:
        return 0
    else:
        return (velocity_v[0] ** 2 + velocity_v[1] ** 2) ** 0.5 * RAD2DEG / radius


def calc_cir_rect(poly):
    [x_min, y_min] = np.min(poly, axis=0)
    [x_max, y_max] = np.max(poly, axis=0)
    min_rect = np.array([[x_min, y_min],
                         [x_max, y_min],
                         [x_max, y_max],
                         [x_min, y_max]])
    return min_rect, x_max - x_min, y_max - y_min


def calc_poly_min_cir_rect(poly):
    N, d = poly.shape
    if N < 3 or d != 2:
        raise ValueError
    rect_min, w_min, h_min = calc_cir_rect(poly)
    rad_min = 0.
    area_min = w_min * h_min
    rad = []
    for i in range(N):
        vector = poly[i - 1] - poly[i]
        rad.append(np.arctan(vector[1] / (vector[0] + EPS)))
    for r in rad:
        new_poly = rot_vecs(poly, r)
        rect, w, h = calc_cir_rect(new_poly)
        area = w * h
        if area < area_min:
            rect_min, area_min, w_min, h_min, rad_min = rect, area, w, h, -r
    min_rect_r = rot_vecs(rect_min, rad_min)
    return min_rect_r, w_min, h_min, rad_min


def calc_poly_min_cir_circle(poly: list):
    tar_p = pickle.loads(pickle.dumps(poly))
    cur_p_set = []
    p1, p2 = tar_p.pop(), tar_p.pop()
    cur_p_set.append(p1)
    cur_p_set.append(p2)
    cur_circle = [[(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5], alg.l2_norm(np.array(p1) - np.array(p2)) * 0.5]
    while len(tar_p) > 0:
        pt = tar_p.pop()
        if not chk_p_in_circle(pt, cur_circle[0], cur_circle[1]):
            p_num = len(cur_p_set)
            min_r = float('inf')
            min_cir = []
            for i in range(p_num - 1):
                p1 = cur_p_set[i]
                for j in range(i + 1, p_num):
                    p2 = cur_p_set[j]
                    temp_circle = calc_tri_min_cir_circle(np.array(p1), np.array(p2), np.array(pt))
                    in_temp = True
                    for tp in cur_p_set:
                        if not chk_p_in_circle(tp, temp_circle[0], temp_circle[1]):
                            in_temp = False
                            break
                    if in_temp:
                        if temp_circle[1] <= min_r:
                            min_cir = temp_circle
                            min_r = temp_circle[1]
            cur_circle = min_cir
        cur_p_set.append(pt)
    return cur_circle


def calc_tri_min_cir_circle(p1, p2, p3):
    d1 = alg.l2_norm(p1 - p2)
    d2 = alg.l2_norm(p1 - p3)
    d3 = alg.l2_norm(p2 - p3)
    d_max = max(d1, d2, d3)
    if d1 == d_max:
        xt, yt, rt = (p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, d1 * 0.5
        other_p = p3
    elif d2 == d_max:
        xt, yt, rt = (p1[0] + p3[0]) * 0.5, (p1[1] + p3[1]) * 0.5, d2 * 0.5
        other_p = p2
    else:
        xt, yt, rt = (p2[0] + p3[0]) * 0.5, (p2[1] + p3[1]) * 0.5, d3 * 0.5
        other_p = p1
    d2other = alg.l2_norm(np.array([xt, yt]) - other_p)
    if d2other <= rt:
        x0, y0, r = xt, yt, rt
    else:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x0 = (y3 - y2) * (x2 ** 2 + y2 ** 2 - x1 ** 2 - y1 ** 2) - (y2 - y1) * (x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2)
        x0 = x0 / ((y3 - y2) * (x2 - x1) - (y2 - y1) * (x3 - x2)) * 0.5
        y0 = (x2 - x1) * (x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2) - (x3 - x2) * (x2 ** 2 + y2 ** 2 - x1 ** 2 - y1 ** 2)
        y0 = y0 / ((y3 - y2) * (x2 - x1) - (y2 - y1) * (x3 - x2)) * 0.5
        r = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return [[x0, y0], r]


def calc_poly_max_in_circle(poly: list):
    tar_poly = pickle.loads(pickle.dumps(poly))
    x_sub, y_sub = 20, 20
    stop_thre = 0.1
    x_min, x_max = float('inf'), 0
    y_min, y_max = float('inf'), 0
    for p in tar_poly:
        x, y = p
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    bound = [x_min, x_max, y_min, y_max]
    interval = (2 ** 0.5) * 2

    while True:
        c_center, radius = calc_poly_in_circle(tar_poly, bound, x_sub, y_sub)
        cx, cy = c_center
        fit_tmp = (bound[1] - bound[0]) / interval
        bound[1] = cx + fit_tmp
        bound[0] = cx - fit_tmp
        fit_tmp = (bound[3] - bound[2]) / interval
        bound[3] = cy + fit_tmp
        bound[2] = cy - fit_tmp
        if (bound[1] - bound[0]) < stop_thre or (bound[3] - bound[2]) < stop_thre:
            break

    return c_center, radius


def calc_poly_in_circle(poly: list, out_bound, x_sub, y_sub):
    poly = np.array(poly)
    c_center = [0, 0]
    x_inc = (out_bound[1] - out_bound[0]) / x_sub
    y_inc = (out_bound[3] - out_bound[2]) / y_sub
    max_dis = 0
    for i in range(x_sub):
        x_temp = out_bound[0] + i * x_inc
        for j in range(y_sub):
            y_temp = out_bound[2] + j * y_inc
            if chk_p_in_conv_simple([x_temp, y_temp], poly):
                dis_temp, _ = calc_point_mindis2poly([x_temp, y_temp], poly)
                if dis_temp > max_dis:
                    max_dis = dis_temp
                    c_center = [x_temp, y_temp]
    return c_center, max_dis


def calc_point_mindis2bound(pos, bounds):
    min_dis = float('inf')
    in_bound = True
    for bound in bounds:
        in_cur_bound = False
        b_ps = bound.points
        b_ps_num = len(b_ps)
        intersect_num = 0
        for i in range(-1, b_ps_num - 1):
            b_s, b_e = b_ps[i], b_ps[i + 1]
            if chk_right_ray_line_cross(pos, b_s, b_e):
                intersect_num = intersect_num + 1
            p2curb = calc_point_mindis2line(pos, b_s, b_e)
            # 如果距离当前边界过进，则认为发生边界碰撞，直接触发出界
            if p2curb <= min_dis:
                min_dis = p2curb
        if intersect_num % 2 != 0:
            in_cur_bound = True
        if in_cur_bound and not bound.is_out_bound:
            in_bound = False
            break
        elif not in_cur_bound and bound.is_out_bound:
            in_bound = False
            break
    if in_bound:
        return min_dis
    else:
        return None


def calc_point_mindis2poly(pos, poly: np.ndarray):
    row, col = poly.shape
    min_dis = float('inf')
    mp1, mp2 = None, None
    for i in range(-1, row - 1):
        p1, p2 = poly[i], poly[i + 1]
        dis, r, t = calc_point_pro2line(pos, p1, p2)
        if dis < min_dis:
            min_dis = dis
            mp1, mp2 = p1, p2
    return min_dis, [mp1, mp2]


def calc_point_pro2line(p, l_s, l_e):
    point_loc = np.array(p)
    line_s_loc = np.array(l_s)
    line_e_loc = np.array(l_e)
    line_vec = line_e_loc - line_s_loc
    r_dir_normal = norm_vec(line_vec)
    t = np.dot(point_loc - line_s_loc, r_dir_normal)
    result = np.multiply(t, r_dir_normal) + l_s
    v = result - point_loc
    distance = (v[0] ** 2 + v[1] ** 2) ** 0.5
    if line_vec[0] != 0:
        t_t = t * r_dir_normal[0] / line_vec[0]
    elif line_vec[1] != 0:
        t_t = t * r_dir_normal[1] / line_vec[1]
    else:
        t_t = None
        distance = None
        result = None
    return distance, result, t_t


def calc_point_mindis2line(p, l_s, l_e):
    p_loc = np.array(p)
    s_loc = np.array(l_s)
    e_loc = np.array(l_e)
    cos1 = np.dot(p_loc - s_loc, e_loc - s_loc)
    cos2 = np.dot(p_loc - e_loc, s_loc - e_loc)
    if cos1 * cos2 >= 0:
        a = e_loc[1] - s_loc[1]
        b = s_loc[0] - e_loc[0]
        c = e_loc[0] * s_loc[1] - s_loc[0] * e_loc[1]
        d = (a ** 2 + b ** 2) ** 0.5
        return np.fabs(a * p_loc[0] + b * p_loc[1] + c) / d
    else:
        ps = p_loc - s_loc
        d1 = (ps[0] ** 2 + ps[1] ** 2) ** 0.5
        es = p_loc - e_loc
        d2 = (es[0] ** 2 + es[1] ** 2) ** 0.5
        if d1 < d2:
            return d1
        else:
            return d2


def calc_lines_intersect(l1_s, l1_e, l2_s, l2_e):
    l1_s = np.array(l1_s)
    l1_e = np.array(l1_e)
    l2_s = np.array(l2_s)
    l2_e = np.array(l2_e)
    l1_v = l1_e - l1_s
    l2_v = l2_e - l2_s
    i_type = -1  # -2 para erro,  -1 - no intersect， 0 - collinear， 1 - 1 intersect
    intersect = []
    extend_info = []
    if (l1_v[0] == 0 and l1_v[1] == 0) or (l2_v[0] == 0 and l2_v[1] == 0):
        i_type = -2
        intersect = None
        extend_info = None
    elif alg.cross(l1_v, l2_v) != 0:
        a, b = l1_v[0], l1_v[1]
        c, d = l2_v[0], l2_v[1]
        e, f = l2_s[0] - l1_s[0], l2_s[1] - l1_s[1]
        s = (e * d - c * f) / (a * d - c * b)
        t = (e * b - a * f) / (a * d - c * b)
        if 0 <= s <= 1 and 0 <= t <= 1:
            i_type = 1
            intersect = np.multiply(l1_v, s) + l1_s
            extend_info = (s, t)
        else:
            i_type = -1
            intersect = None
            extend_info = None
    else:
        l3_v = l2_e - l1_s
        if alg.cross(l1_v, l3_v) != 0:
            i_type = -1
            intersect = None
            extend_info = None
        else:
            pass

    return i_type, intersect, extend_info


def calc_nonparallel_lines_intersect(l1_s, l1_e, l2_s, l2_e):
    l1_s = np.array(l1_s)
    l1_e = np.array(l1_e)
    l2_s = np.array(l2_s)
    l2_e = np.array(l2_e)
    l1_v = l1_e - l1_s
    l2_v = l2_e - l2_s

    if alg.cross(l1_v, l2_v) != 0:
        a, b = l1_v
        c, d = l2_v
        e, f = l2_s - l1_s
        m = a * d - c * b
        s = (e * d - c * f) / m
        t = (e * b - a * f) / m
        if 0 <= s <= 1 and 0 <= t <= 1:
            return 1, (np.multiply(l1_v, s) + l1_s).tolist(), (s, t)
    return 0, None, None


def calc_ray_line_intersect(l_s, l_e, r_s, r_p) -> Tuple[int, Tuple[np.ndarray, np.ndarray] |
                                                              Tuple[None, np.ndarray] |
                                                              None]:
    l_left = l_s - r_s
    l_right = l_e - r_s

    if alg.cross(l_left, l_right) == 0:
        r_dir = r_p - r_s
        if np.dot(l_left, l_right) <= 0:
            if np.dot(r_dir, l_right) >= 0:
                return 2, (r_s, l_e)
            else:
                return 2, (r_s, l_s)
        elif np.dot(r_dir, l_left) >= 0:
            return 2, (l_s, l_e)
        else:
            return 0, None
    else:
        center = (l_s + l_e) * 0.5
        v1, v2 = r_p - r_s, center - r_s
        rela1 = alg.cross(v1, r_p - l_s) * alg.cross(v2, center - l_s)
        rela2 = alg.cross(v1, r_p - l_e) * alg.cross(v2, center - l_e)

        if rela1 >= 0 and rela2 >= 0:
            xa, ya = l_s
            xb, yb = l_e
            xe, ye = r_p
            xd, yd = r_s
            t = (xa * yb - xb * ya + (ya - yb) * xd + (xb - xa) * yd) / ((xe - xd) * (yb - ya) + (ye - yd) * (xa - xb))
            return 1, (None, t * v1 + r_s)
        else:
            return 0, None


def calc_ray_bound_intersection(pos: np.ndarray,
                                fwd: np.ndarray,
                                bounds) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    min_dis = float('inf')
    min_inter = None
    min_bound = []
    for b in bounds:
        for i in range(b.points_num):
            contour_s = np.array(b.points[i - 1])
            contour_e = np.array(b.points[i])
            inter_type, inter_point = calc_ray_line_intersect(contour_s, contour_e, pos, pos + fwd)
            if inter_type == 1:
                d = alg.l2_norm(inter_point[1] - pos)
                if d < min_dis:
                    min_dis = d
                    min_inter = inter_point[1]
                    min_bound = (contour_s.copy(), contour_e.copy())
            elif inter_type == 2:
                inter1, inter2 = inter_point
                d1 = alg.l2_norm(inter1 - pos)
                d2 = alg.l2_norm(inter2 - pos)
                d, inter = (d1, inter1) if d1 < d2 else (d2, inter2)
                if d < min_dis:
                    min_dis = d
                    min_inter = inter
                    min_bound = (contour_s.copy(), contour_e.copy())
    return min_inter, min_dis, min_bound


def calc_square_bound_cross(center, width, boundaries):
    c_x, c_y = center
    h_w = 0.5 * width
    s_points = [[c_x - h_w, c_y - h_w],
                [c_x + h_w, c_y - h_w],
                [c_x + h_w, c_y + h_w],
                [c_x - h_w, c_y + h_w]]
    cross = []
    for boundary in boundaries:
        walls = boundary.points
        walls_num = boundary.points_num
        for i in range(-1, walls_num - 1):
            sx, sy = walls[i]
            ex, ey = walls[i + 1]
            line = [sx, sy, ex, ey]
            if chk_line_rect_cross(line, s_points):
                cross.append(np.array([walls[i], walls[i + 1]]))
    return np.array(cross)


def chk_p_in_tilings(pos, tilings):
    target = None
    for tiling in tilings:
        if tiling.x_min <= pos[0] <= tiling.x_max and tiling.y_min <= pos[1] <= tiling.y_max:
            target = tiling
        else:
            continue
    return target


def chk_ps_on_line_side(p1, p2, l_s, l_e):
    return alg.cross(p1 - l_s, p1 - l_e) * alg.cross(p2 - l_s, p2 - l_e) > 0


def chk_lines_cross(l1_s, l1_e, l2_s, l2_e) -> bool:
    if max(l1_s[0], l1_e[0]) >= min(l2_s[0], l2_e[0]) and max(l2_s[0], l2_e[0]) >= min(l1_s[0], l1_e[0]) and \
            max(l1_s[1], l1_e[1]) >= min(l2_s[1], l2_e[1]) and max(l2_s[1], l2_e[1]) >= min(l1_s[1], l1_e[1]):
        nl1_s = np.array(l1_s)
        nl1_e = np.array(l1_e)
        nl2_s = np.array(l2_s)
        nl2_e = np.array(l2_e)
        return np.cross(nl2_s - nl1_s, nl2_e - nl1_s) * np.cross(nl2_s - nl1_e, nl2_e - nl1_e) <= 0 and \
            np.cross(nl1_s - nl2_s, nl1_e - nl2_s) * np.cross(nl1_s - nl2_e, nl1_e - nl2_e) <= 0
    else:
        return False


def chk_line_bound_cross(l1_s, l1_e, bounds):
    for bound in bounds:
        b_ps = bound.points
        b_ps_num = len(b_ps)
        for i in range(-1, b_ps_num - 1):
            wall_s = b_ps[i]
            wall_e = b_ps[i + 1]
            intersects = chk_lines_cross(l1_s, l1_e, wall_s, wall_e)
            if intersects:
                return True
    return False


def chk_line_rect_cross(line, rect):
    l_s = np.array(line[:2])
    l_e = np.array(line[2:4])
    left = rect[0][0]
    right = rect[2][0]
    bottom = rect[0][1]
    top = rect[2][1]
    if (left <= l_s[0] <= right and bottom <= l_s[1] <= top) or (left <= l_e[0] <= right and bottom <= l_e[1] <= top):
        return 1
    else:
        if chk_lines_cross(l_s, l_e, rect[0], rect[2]) or chk_lines_cross(l_s, l_e, rect[1], rect[3]):
            return 1
        else:
            return 0


def chk_right_ray_line_cross(r_s, l_s, l_e):
    if l_s[1] == l_e[1]:
        return False
    if l_s[1] > r_s[1] and l_e[1] > r_s[1]:
        return False
    if l_s[1] < r_s[1] and l_e[1] < r_s[1]:
        return False
    if l_s[1] == r_s[1] and l_e[1] > r_s[1]:
        return False
    if l_e[1] == r_s[1] and l_s[1] > r_s[1]:
        return False
    if l_s[0] < r_s[0] and l_e[0] < r_s[0]:
        return False
    x_seg = l_e[0] - (l_e[0] - l_s[0]) * (l_e[1] - r_s[1]) / (l_e[1] - l_s[1])  # 求交
    if x_seg < r_s[0]:
        return False
    else:
        return True


def chk_p_rect_quick(p, rect_w, rect_h):
    return 0 <= p[0] <= rect_w and 0 <= p[1] <= rect_h


def chk_p_in_bound(pos, bounds, danger_dis=0):
    in_bound = True
    for bound in bounds:
        in_cur_bound = False
        b_ps = bound.points
        b_ps_num = len(b_ps)
        intersect_num = 0
        for i in range(-1, b_ps_num - 1):
            b_s, b_e = b_ps[i], b_ps[i + 1]
            if chk_right_ray_line_cross(pos, b_s, b_e):
                intersect_num = intersect_num + 1
            # 如果距离当前边界过进，则认为发生边界碰撞，直接触发出界
            if calc_point_mindis2line(pos, b_s, b_e) <= danger_dis:
                return False
        if intersect_num % 2 != 0:
            in_cur_bound = True
        if in_cur_bound and not bound.is_out_bound:
            in_bound = False
            break
        elif not in_cur_bound and bound.is_out_bound:
            in_bound = False
            break

    return in_bound


def chk_p_in_conv(pos, poly: list, poly_in_circle):
    c, r = poly_in_circle
    c = np.array(c)
    v = pos - c
    if v[0] ** 2 + v[1] ** 2 <= r ** 2:
        return True
    else:
        n_poly = np.array(poly)
        n_poly = n_poly - pos
        poly_num = len(poly)
        for i in range(-1, poly_num - 1):
            pv1, pv2 = n_poly[i], n_poly[i + 1]
            cross = alg.cross(pv1, pv2)
            if cross < 0:  # 点在多边形边右侧，代表在多边形之外
                return False
            elif cross == 0 and np.dot(pv1, pv2) <= 0:  # 点p在直线v1v2上，并且在线段v1v2之间，则直接判定在多边形内
                return True
        return True


def chk_p_in_conv_simple(pos: np.ndarray, poly: np.ndarray):
    row, col = poly.shape
    p_poly = poly - pos
    for i in range(-1, row - 1):
        pv1, pv2 = p_poly[i], p_poly[i + 1]
        cross = alg.cross(pv1, pv2)
        if cross < 0:
            return False
        elif cross == 0 and np.dot(pv1, pv2) <= 0:
            return True
    return True


def chk_p_in_tiling_simple(pos, tiling):
    return tiling.x_min <= pos[0] <= tiling.x_max and tiling.y_min <= pos[1] <= tiling.y_max


def chk_p_in_circle(pos, c_center, c_r):
    vec = np.array(pos) - np.array(c_center)
    return (alg.l2_norm(vec) - c_r) <= EPS


def chk_square_bound_cross(center, width, boundaries):
    intersect_boundary = False
    c_x = center[0]
    c_y = center[1]
    h_w = 0.5 * width
    s_points = [[c_x - h_w, c_y - h_w], [c_x + h_w, c_y - h_w],
                [c_x + h_w, c_y - h_w], [c_x + h_w, c_y + h_w],
                [c_x + h_w, c_y + h_w], [c_x - h_w, c_y + h_w],
                [c_x - h_w, c_y + h_w], [c_x - h_w, c_y - h_w]]
    for boundary in boundaries:
        walls = boundary.points
        walls_num = boundary.points_num
        for i in range(-1, walls_num - 1):
            wall_s = walls[i]
            wall_e = walls[i + 1]
            if chk_lines_cross(s_points[0], s_points[1], wall_s, wall_e):
                intersect_boundary = True
                break
            elif chk_lines_cross(s_points[2], s_points[3], wall_s, wall_e):
                intersect_boundary = True
                break
            elif chk_lines_cross(s_points[4], s_points[5], wall_s, wall_e):
                intersect_boundary = True
                break
            elif chk_lines_cross(s_points[6], s_points[7], wall_s, wall_e):
                intersect_boundary = True
                break
        if intersect_boundary:
            break
    return intersect_boundary


def chk_poly_concavity(poly: list):
    poly_num = len(poly)
    for i in range(0, poly_num):
        v_f = poly[i - 1]
        v = poly[i]
        v_n = poly[(i + 1) % poly_num]
        vec1 = np.array([v[0] - v_f[0], v[1] - v_f[1]])
        vec2 = np.array([v_n[0] - v[0], v_n[1] - v[1]])
        if calc_angle_bet_vec(vec2, vec1) <= 0:
            return False
    return True


def chk_convs_intersect(poly1: list, poly1_in_cir, poly2: list, poly2_in_cir):
    if len(poly1) < 3 or len(poly2) < 3:
        return False
    else:
        poly1_num, poly2_num = len(poly1), len(poly2)
        total_set = []
        for p in poly2:
            if chk_p_in_conv(np.array(p), poly1, poly1_in_cir):
                total_set.append(p)
        if len(total_set) >= 3:
            return True
        for i in range(-1, poly1_num - 1):
            po1v1, po1v2 = poly1[i], poly1[i + 1]
            if chk_p_in_conv(np.array(po1v2), poly2, poly2_in_cir):
                can_add = True
                for ep in total_set:
                    if abs(ep[0] - po1v2[0]) < EPS and abs(ep[1] - po1v2[1]) < EPS:
                        can_add = False
                        break
                if can_add:
                    total_set.append(po1v2)
                    if len(total_set) >= 3:
                        return True
            for j in range(-1, poly2_num - 1):
                po2v1, po2v2 = poly2[j], poly2[j + 1]
                # 这里特别注意，由于交点坐标精度问题，可能出现极小的误差，这里需要进一步处理一下，忽略误差
                i_t, i_p, _ = calc_nonparallel_lines_intersect(po1v1, po1v2, po2v1, po2v2)
                # 仅考察poly1和poly2非共线边的交叉点,原因是若两边共线，无非就是重合或完全不相交
                # 若重合，则重合点在之前判断多边形顶点是否在另一多边形内时就会找到
                if i_t > 0:
                    can_add = True
                    for ep in total_set:
                        if abs(ep[0] - i_p[0]) < EPS and abs(ep[1] - i_p[1]) < EPS:  # 判断两点是否相同，去重复
                            can_add = False
                            break
                    if can_add:
                        total_set.append(i_p)
                        if len(total_set) >= 3:
                            return True
        return False


def chk_points_clockwise(points) -> int:
    p1, p2, p3 = points[0], points[1], points[2]
    clockwise = ((p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1]))
    if clockwise > 0:
        return 1
    elif clockwise == 0:
        return 0
    else:
        return -1


def chk_ray_rect_AABB(r_s: Tuple, r_dir: Tuple, rect: Tuple[float, float, float, float]):
    min_x, min_y, max_x, max_y = rect
    rs_x, rs_y = r_s
    if r_dir[0] == 0:
        if rs_x < min_x or rs_x > max_x:
            return False
        elif min_y <= rs_y <= max_y:
            return True
        elif rs_y < min_y and r_dir[1] > 0:
            return True
        elif rs_y > max_y and r_dir[1] < 0:
            return True
        else:
            return False
    elif r_dir[1] == 0:
        if rs_y < min_y or rs_y > max_y:
            return False
        elif min_x <= rs_x <= max_x:
            return True
        elif rs_x < min_x and r_dir[0] > 0:
            return True
        elif rs_x > max_x and r_dir[0] < 0:
            return True
        else:
            return False
    else:
        slab_x_t1 = (min_x - rs_x) / r_dir[0]
        slab_x_t2 = (max_x - rs_x) / r_dir[0]
        slab_y_t1 = (min_y - rs_y) / r_dir[1]
        slab_y_t2 = (max_y - rs_y) / r_dir[1]
        slab_x_tmin = min(slab_x_t1, slab_x_t2)
        slab_x_tmax = max(slab_x_t1, slab_x_t2)
        slab_y_tmin = min(slab_y_t1, slab_y_t2)
        slab_y_tmax = max(slab_y_t1, slab_y_t2)
        if (slab_x_tmin < 0 and slab_x_tmax < 0) or (slab_y_tmin < 0 and slab_y_tmax < 0):
            return False
        elif slab_x_tmax >= slab_y_tmin or slab_x_tmin >= slab_y_tmax:
            return True
        else:
            all_t = sorted([slab_x_tmin, slab_x_tmax, slab_y_tmin, slab_y_tmax])
            if abs(all_t[1] - all_t[2]) < EPS:
                return True
            else:
                return False


def chk_ray_line_cross(r_s: Tuple, r_dir: Tuple, l_s: Tuple, l_e: Tuple):
    r_e = (r_s[0] + r_dir[0], r_s[1] + r_dir[1])


def calc_poly_angle(p, inner_p):
    vec = np.array(p) - inner_p
    ang = np.arctan2(vec[1], vec[0])
    ang = ang if ang >= 0 else ang + PI_2
    return ang


def quick_sort_poly_points(inner_p, poly_points: list, low, high):
    if low >= high:
        return poly_points
    i = low
    j = high
    pivot = poly_points[low]
    ang_piv = calc_poly_angle(pivot, inner_p)
    while i < j:
        while i < j and calc_poly_angle(poly_points[j], inner_p) > ang_piv:
            j -= 1
        poly_points[i] = poly_points[j]
        while i < j and calc_poly_angle(poly_points[i], inner_p) < ang_piv:
            i += 1
        poly_points[j] = poly_points[i]
    poly_points[j] = pivot
    quick_sort_poly_points(inner_p, poly_points, low, j - 1)
    quick_sort_poly_points(inner_p, poly_points, j + 1, high)
    return poly_points


def reorder_conv_vertex(poly_points: list):
    tri_ps = []
    for p in poly_points:
        if len(tri_ps) == 0:
            tri_ps.append(p)
        else:
            can_add = True
            for tp in tri_ps:
                if abs(tp[0] - p[0]) < EPS and abs(tp[1] - p[1]) < EPS:
                    can_add = False
                    break
            if can_add:
                tri_ps.append(p)
            if len(tri_ps) == 3:
                break
    inner_p = calc_convex_centroid(np.array(tri_ps))

    reordered_poly = []
    for p in poly_points:
        if len(reordered_poly) == 0:
            reordered_poly.append(p)
        else:
            r_num = len(reordered_poly)
            insert_i = 0
            for i in range(r_num):
                rp = reordered_poly[i]
                if abs(rp[0] - p[0]) < EPS and abs(rp[1] - p[1]) < EPS:
                    insert_i = None
                    break
                else:
                    insert_i = i
                    vec1, vec2 = np.array(rp) - inner_p, np.array(p) - inner_p
                    ang1, ang2 = np.arctan2(vec1[1], vec1[0]), np.arctan2(vec2[1], vec2[0])
                    ang1 = ang1 if ang1 >= 0 else ang1 + PI_2
                    ang2 = ang2 if ang2 >= 0 else ang2 + PI_2
                    if ang1 > ang2:
                        break
            if insert_i is not None:
                reordered_poly = reordered_poly[0:insert_i] + [p] + reordered_poly[insert_i:r_num + 1]

    return reordered_poly


def cmp_convex_vertex_order(points_set: list):
    centroid = calc_convex_centroid(np.array(points_set[0:3]))
    point_num = len(points_set)
    point_angs = []
    reordered_poly = []
    reordered_angs = []
    for p in points_set:
        vec = np.array(p) - centroid
        ang_p = np.arctan2(vec[1], vec[0])
        ang_p = ang_p if ang_p >= 0 else ang_p + PI_2
        point_angs.append(ang_p)

    for i in range(0, point_num - 1):
        for j in range(0, point_num - i - 1):
            ang1 = point_angs[j]
            ang2 = point_angs[j + 1]
            if ang1 > ang2:
                temp = np.array(points_set[j]).tolist()
                points_set[j] = points_set[j + 1]
                points_set[j + 1] = temp
                temp_p = point_angs[j]
                point_angs[j] = point_angs[j + 1]
                point_angs[j + 1] = temp_p
    return points_set


def calc_con_polys_intersect(poly1: list, poly1_in_cir, poly2: list, poly2_in_cir):
    if len(poly1) < 3 or len(poly2) < 3:
        inter_poly = None
    else:
        poly1_num, poly2_num = len(poly1), len(poly2)
        total_set = []
        for p in poly2:
            if chk_p_in_conv(np.array(p), poly1, poly1_in_cir):
                total_set.append(p)
        for i in range(-1, poly1_num - 1):
            po1v1, po1v2 = poly1[i], poly1[i + 1]
            if chk_p_in_conv(np.array(po1v2), poly2, poly2_in_cir):
                can_add = True
                for ep in total_set:
                    if abs(ep[0] - po1v2[0]) < EPS and abs(ep[1] - po1v2[1]) < EPS:  # 判断两点是否相同，去重复
                        can_add = False
                        break
                if can_add:
                    total_set.append(po1v2)
            for j in range(-1, poly2_num - 1):
                po2v1, po2v2 = poly2[j], poly2[j + 1]
                i_t, i_p, _ = calc_nonparallel_lines_intersect(po1v1, po1v2, po2v1, po2v2)
                if i_t:
                    can_add = True
                    for ep in total_set:
                        if abs(ep[0] - i_p[0]) < EPS and abs(ep[1] - i_p[1]) < EPS:
                            can_add = False
                            break
                    if can_add:
                        total_set.append(i_p)

        inter_poly = cmp_convex_vertex_order(total_set) if len(total_set) >= 3 else None

    return inter_poly


def calc_con_polys_intersect_simple(poly1: Sequence, poly2: Sequence):
    if len(poly1) < 3 or len(poly2) < 3:
        inter_poly = None
    else:
        poly1_num, poly2_num = len(poly1), len(poly2)
        total_set = []
        for p in poly2:
            if chk_p_in_conv_simple(p, np.array(poly1)):
                total_set.append(p)
        for i in range(-1, poly1_num - 1):
            po1v1, po1v2 = poly1[i], poly1[i + 1]
            if chk_p_in_conv_simple(po1v2, np.array(poly2)):
                can_add = True
                for ep in total_set:
                    if abs(ep[0] - po1v2[0]) < EPS and abs(ep[1] - po1v2[1]) < EPS:
                        can_add = False
                        break
                if can_add:
                    total_set.append(po1v2)
            for j in range(-1, poly2_num - 1):
                po2v1, po2v2 = poly2[j], poly2[j + 1]
                i_t, i_p, _ = calc_nonparallel_lines_intersect(po1v1, po1v2, po2v1, po2v2)
                if i_t > 0:
                    can_add = True
                    for ep in total_set:
                        if abs(ep[0] - i_p[0]) < EPS and abs(ep[1] - i_p[1]) < EPS:
                            can_add = False
                            break
                    if can_add:
                        total_set.append(i_p)

        inter_poly = cmp_convex_vertex_order(total_set) if len(total_set) >= 3 else None

    return inter_poly


def calc_sort_point_cos(points, center_point):
    n = len(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points[i]
        point = [point_[0] - center_point[0], point_[1] - center_point[1]]
        rank.append(i)
        norm_value = np.sqrt(point[0] * point[0] + point[1] * point[1])
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)

    for i in range(0, n - 1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index - 1] or (
                    cos_value[index] == cos_value[index - 1] and norm_list[index] > norm_list[index - 1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index - 1]
                rank[index] = rank[index - 1]
                norm_list[index] = norm_list[index - 1]
                cos_value[index - 1] = temp
                rank[index - 1] = temp_rank
                norm_list[index - 1] = temp_norm
                index = index - 1
            else:
                break
    sorted_points = []
    for i in rank:
        sorted_points.append(points[i])

    return sorted_points


def calc_convex_graham_scan(points):
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1] < points[min_index][1] or (
                points[i][1] == points[min_index][1] and points[i][0] < points[min_index][0]):
            min_index = i

    bottom_point = points.pop(min_index)
    sorted_points = calc_sort_point_cos(points, bottom_point)

    m = len(sorted_points)
    if m < 2:
        print("error")
        return

    stack = []
    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])

    for i in range(2, m):
        length = len(stack)
        top = stack[length - 1]
        next_top = stack[length - 2]
        v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
        v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        while alg.cross(v1, v2) >= 0:
            stack.pop()
            length = len(stack)
            top = stack[length - 1]
            next_top = stack[length - 2]
            v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
            v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        stack.append(sorted_points[i])

    return stack


def calc_poly_triangulation(poly_bounds: list):
    total_bound, total_edges = calc_poly_out_bound(poly_bounds)
    tris = calc_out_bound_triangulation(total_bound, total_edges)
    return tris


def calc_poly_out_bound(poly_bounds: list):
    target_bounds = pickle.loads(pickle.dumps(poly_bounds))
    if target_bounds is None:
        return None, None
    elif len(target_bounds) == 1:
        total_bound = target_bounds[0]
        ver_num = len(total_bound)
        total_bound, _ = calc_adjust_poly_order(total_bound)
        total_edges = []
        for i in range(-1, ver_num - 1):
            total_edges.append([total_bound[i], total_bound[i + 1], 1])
    else:
        bounds_num = len(target_bounds)
        out_bound = pickle.loads(pickle.dumps(target_bounds[0]))
        out_bound, _ = calc_adjust_poly_order(out_bound)
        added_edges = []
        inner_bounds = pickle.loads(pickle.dumps(target_bounds[1:bounds_num]))
        while len(inner_bounds) > 0:
            selected_idx = 0
            max_x = 0
            for i in range(len(inner_bounds)):
                temp_bound = inner_bounds[i]
                max_v_x = 0
                for iv in temp_bound:
                    if iv[0] > max_v_x:
                        max_v_x = iv[0]
                if max_v_x > max_x:
                    selected_idx = i
                    max_x = max_v_x
            inner_bound = inner_bounds[selected_idx]
            inner_bound, _ = calc_adjust_poly_order(inner_bound, order=0)
            in_num = len(inner_bound)
            in_idx, out_idx = find_visible_vertex(inner_bound, out_bound)
            out_num = len(out_bound)
            out1 = out_bound[0:out_idx + 1]
            out2 = out_bound[out_idx:out_num]
            in1 = inner_bound[in_idx:in_num]
            in2 = inner_bound[0:in_idx + 1]
            out_bound = out1 + in1 + in2 + out2
            added_edges.append([out_bound[out_idx], inner_bound[in_idx]])
            added_edges.append([inner_bound[in_idx], out_bound[out_idx]])
            inner_bounds.pop(selected_idx)

        total_bound = out_bound
        ver_num = len(total_bound)
        total_edges = []
        for i in range(-1, ver_num - 1):
            is_out_e = 1
            for added_e in added_edges:
                if chk_dir_edge_same([total_bound[i], total_bound[i + 1]], added_e):
                    is_out_e = 0
                    break
            total_edges.append([total_bound[i], total_bound[i + 1], is_out_e])

    return total_bound, total_edges


def calc_out_bound_triangulation(poly_bounds: list, poly_edges: list):
    """
    calculate the triangulation of a polygon, for the outer contour of polygon, the order of vertex must be
    anti-clockwise, and the inner contours are all in clockwise order.

    Args:
        poly_bounds: a list contains all bounds, [0] must be outer bound.

        poly_edges:

    Returns:
        list of triangulation
    """

    total_bound = pickle.loads(pickle.dumps(poly_bounds))
    total_edges = pickle.loads(pickle.dumps(poly_edges))
    total_num = len(total_bound)
    if total_num < 3:
        return None
    else:
        tris = []
        while len(total_bound) > 3:
            total_bound, total_edges, tri = ear_clip_poly_opti(total_bound, total_edges)  # or use 'ear_clip_poly'
            tris.append(tri)
        tri = Triangle()
        tri.set_points(total_bound[0], total_bound[1], total_bound[2])
        tri.vertices = cmp_convex_vertex_order(tri.vertices)
        tri.in_circle = calc_poly_max_in_circle(tri.vertices)
        e1 = remove_edge_from_edgeset(total_edges, [total_bound[0], total_bound[1]])
        e2 = remove_edge_from_edgeset(total_edges, [total_bound[1], total_bound[2]])
        e3 = remove_edge_from_edgeset(total_edges, [total_bound[0], total_bound[2]])
        if e1[2]:
            tri.out_edges.append([tri.find_vertex_idx(e1[0]), tri.find_vertex_idx(e1[1])])
        else:
            tri.in_edges.append([tri.find_vertex_idx(e1[0]), tri.find_vertex_idx(e1[1])])
        if e2[2]:
            tri.out_edges.append([tri.find_vertex_idx(e2[0]), tri.find_vertex_idx(e2[1])])
        else:
            tri.in_edges.append([tri.find_vertex_idx(e2[0]), tri.find_vertex_idx(e2[1])])
        if e3[2]:
            tri.out_edges.append([tri.find_vertex_idx(e3[0]), tri.find_vertex_idx(e3[1])])
        else:
            tri.in_edges.append([tri.find_vertex_idx(e3[0]), tri.find_vertex_idx(e3[1])])
        tris.append(tri)
        return tris


def chk_poly_clockwise(poly_bound: list):
    s = 0
    poly_num = len(poly_bound)
    for i in range(-1, poly_num - 1):
        s += (poly_bound[i + 1][1] + poly_bound[i][1]) * (poly_bound[i][0] - poly_bound[i + 1][0])
    ori_order = 1 if s > 0 else 0
    return ori_order


def calc_adjust_poly_order(poly_bound: list, order=1):
    adjusted = pickle.loads(pickle.dumps(poly_bound))
    ori_order = chk_poly_clockwise(adjusted)
    if (order and not ori_order) or (not order and ori_order):
        adjusted.reverse()
    return adjusted, ori_order


def ear_clip_poly(bound, edges):
    cut_ear = None
    bound_num = len(bound)

    for i in range(0, bound_num):
        is_ear = True
        v_f = bound[i - 1]  # vn-1
        v = bound[i]  # vn
        v_n = bound[(i + 1) % bound_num]  # vn+1
        vec1 = np.array([v[0] - v_f[0], v[1] - v_f[1]])
        vec2 = np.array([v_n[0] - v[0], v_n[1] - v[1]])
        if calc_angle_bet_vec(vec2, vec1) < 0:
            is_ear = False
        else:
            tri_c = np.array([(v_f[0] + v[0] + v_n[0]) / 3, (v_f[1] + v[1] + v_n[1]) / 3])
            for j in range(0, bound_num - 3):
                other_v = np.array(bound[(i + j + 2) % bound_num])
                in_tri = False
                nv_f = np.array(v_f)
                nv = np.array(v)
                nv_n = np.array(v_n)
                if chk_ps_on_line_side(tri_c, other_v, nv_f, nv):
                    if chk_ps_on_line_side(tri_c, other_v, nv, nv_n):
                        if chk_ps_on_line_side(tri_c, other_v, nv_n, nv_f):
                            in_tri = True
                if in_tri:
                    is_ear = False
                    break
        if is_ear:
            bound.pop(i)
            e1 = remove_edge_from_edgeset(edges, [v_f, v], is_dir=1)
            e2 = remove_edge_from_edgeset(edges, [v, v_n], is_dir=1)
            e3 = [v_f, v_n, 0]
            edges.append(e3)
            cut_ear = Triangle()
            cut_ear.set_points(v_f, v, v_n)
            cut_ear.vertices = cmp_convex_vertex_order(cut_ear.vertices)
            cut_ear.in_circle = calc_poly_max_in_circle(cut_ear.vertices)
            if e1[2]:
                cut_ear.out_edges.append([cut_ear.find_vertex_idx(e1[0]), cut_ear.find_vertex_idx(e1[1])])
            else:
                cut_ear.in_edges.append([cut_ear.find_vertex_idx(e1[0]), cut_ear.find_vertex_idx(e1[1])])
            if e2[2]:
                cut_ear.out_edges.append([cut_ear.find_vertex_idx(e2[0]), cut_ear.find_vertex_idx(e2[1])])
            else:
                cut_ear.in_edges.append([cut_ear.find_vertex_idx(e2[0]), cut_ear.find_vertex_idx(e2[1])])
            cut_ear.in_edges.append([cut_ear.find_vertex_idx(e3[0]), cut_ear.find_vertex_idx(e3[1])])
            break

    return bound, edges, cut_ear


def ear_clip_poly_opti(bound, edges):
    cut_ear = None
    bound_num = len(bound)
    ears = []
    for i in range(0, bound_num):
        is_ear = True
        v_f = bound[i - 1]  # vn-1
        v = bound[i]  # vn
        v_n = bound[(i + 1) % bound_num]  # vn+1
        nv_f = np.array(v_f)
        nv = np.array(v)
        nv_n = np.array(v_n)
        vec1 = np.array([v[0] - v_f[0], v[1] - v_f[1]])
        vec2 = np.array([v_n[0] - v[0], v_n[1] - v[1]])
        if calc_angle_bet_vec(vec2, vec1) <= 0:
            is_ear = False
        else:
            tri_c = np.array([(v_f[0] + v[0] + v_n[0]) / 3, (v_f[1] + v[1] + v_n[1]) / 3])
            for j in range(0, bound_num - 3):
                other_v = np.array(bound[(i + j + 2) % bound_num])
                in_tri = False
                if chk_ps_on_line_side(tri_c, other_v, nv_f, nv):
                    if chk_ps_on_line_side(tri_c, other_v, nv, nv_n):
                        if chk_ps_on_line_side(tri_c, other_v, nv_n, nv_f):
                            in_tri = True
                if in_tri:
                    is_ear = False
                    break
        if is_ear:
            ears.append(i)
    target_idx = 0
    tar_vf, tar_v, tar_vn = None, None, None
    max_area = 0
    for idx in ears:
        v_f = bound[idx - 1]  # vn-1
        v = bound[idx]  # vn
        v_n = bound[(idx + 1) % bound_num]  # vn+1

        area = calc_tri_area([v_f, v, v_n])
        if area > max_area:
            target_idx = idx
            tar_vf, tar_v, tar_vn = v_f, v, v_n
            max_area = area
    bound.pop(target_idx)
    e1 = remove_edge_from_edgeset(edges, [tar_vf, tar_v], is_dir=1)
    e2 = remove_edge_from_edgeset(edges, [tar_v, tar_vn], is_dir=1)
    e3 = [tar_vf, tar_vn, 0]
    edges.append(e3)
    cut_ear = Triangle()
    cut_ear.set_points(tar_vf, tar_v, tar_vn)
    cut_ear.vertices = cmp_convex_vertex_order(cut_ear.vertices)
    cut_ear.in_circle = calc_poly_max_in_circle(cut_ear.vertices)
    if e1[2]:
        cut_ear.out_edges.append([cut_ear.find_vertex_idx(e1[0]), cut_ear.find_vertex_idx(e1[1])])
    else:
        cut_ear.in_edges.append([cut_ear.find_vertex_idx(e1[0]), cut_ear.find_vertex_idx(e1[1])])
    if e2[2]:
        cut_ear.out_edges.append([cut_ear.find_vertex_idx(e2[0]), cut_ear.find_vertex_idx(e2[1])])
    else:
        cut_ear.in_edges.append([cut_ear.find_vertex_idx(e2[0]), cut_ear.find_vertex_idx(e2[1])])
    cut_ear.in_edges.append([cut_ear.find_vertex_idx(e3[0]), cut_ear.find_vertex_idx(e3[1])])

    return bound, edges, cut_ear


def find_visible_vertex(inner_bound, outer_bound):
    M = None
    m_idx = 0
    mx_max = 0
    in_num = len(inner_bound)
    for j in range(0, in_num):
        iv = inner_bound[j]
        if iv[0] >= mx_max:
            M = iv
            m_idx = j
            mx_max = iv[0]
    out_num = len(outer_bound)
    intersect_t = float('inf')
    I, cor_b1, cor_b2 = None, None, None
    cor_b1_idx, cor_b2_idx = 0, 0
    for i in range(-1, out_num - 1):
        ov1 = outer_bound[i]
        ov2 = outer_bound[i + 1]
        i_p, i_t = find_ray_edge_intersect(M, ov1, ov2)
        if i_p is not None:
            if 0 <= i_t <= intersect_t:
                I, cor_b1, cor_b2 = i_p, ov1, ov2
                cor_b1_idx, cor_b2_idx = i, i + 1
                intersect_t = i_t
    if chk_p_same(I, cor_b1):
        return m_idx, cor_b1_idx
    elif chk_p_same(I, cor_b2):
        return m_idx, cor_b2_idx
    else:
        if cor_b1[0] > cor_b2[0]:
            P = cor_b1
            p_idx = cor_b1_idx
        else:
            P = cor_b2
            p_idx = cor_b2_idx
        p_in_MIP = []
        tri_c = np.array([(M[0] + I[0] + P[0]) / 3, (M[1] + I[1] + P[1]) / 3])
        nM, nI, nP = np.array(M), np.array(I), np.array(P)
        for r_i in range(0, out_num - 2):
            idx = (cor_b2_idx + r_i + 1) % out_num
            other_v = np.array(outer_bound[idx])
            if chk_ps_on_line_side(tri_c, other_v, nM, nI):
                if chk_ps_on_line_side(tri_c, other_v, nI, nP):
                    if chk_ps_on_line_side(tri_c, other_v, nP, nM):
                        p_in_MIP.append(idx)
        if len(p_in_MIP) == 0:
            return m_idx, p_idx
        else:
            min_a = float('inf')
            for p_i in p_in_MIP:
                potential_p = outer_bound[p_i]
                v_mp = np.array(potential_p) - np.array(M)
                v_len = alg.l2_norm(v_mp)
                if v_len < min_a:
                    p_idx = p_i
                    min_a = v_len
            return m_idx, p_idx


def find_ray_edge_intersect(m, p1, p2):
    if p1[1] > m[1] and p2[1] > m[1]:
        i_p = None
        t = None
    elif p1[1] < m[1] and p2[1] < m[1]:
        i_p = None
        t = None
    elif p1[1] == m[1] and p2[1] != m[1]:
        i_p = p1
        t = p1[0] - m[0]
    elif p2[1] == m[1] and p1[1] != m[1]:
        i_p = p2
        t = p2[0] - m[0]
    elif p1[1] == m[1] and p2[1] == m[1]:
        d1 = p1[0] - m[0]
        d2 = p2[0] - m[0]
        if abs(d1) > abs(d2):
            i_p = p2
            t = d2
        else:
            i_p = p1
            t = d1
    else:
        v = np.array(p2) - np.array(p1)
        i_p = (np.array(p1) + np.multiply(v, abs(m[1] - p1[1]) / abs(p2[1] - p1[1]))).tolist()
        t = i_p[0] - m[0]
    return i_p, t


def remove_edge_from_edgeset(edge_set, tar, is_dir=0):
    if tar is None:
        return None
    elif not is_dir:
        for e in edge_set:
            if chk_edge_same(e, tar):
                edge_set.remove(e)
                return e
        return None
    else:
        for e in edge_set:
            if chk_dir_edge_same(e, tar):
                edge_set.remove(e)
                return e
        return None


def chk_edge_same(e1, e2):
    if chk_p_same(e1[0], e2[0]) and chk_p_same(e1[1], e2[1]):
        return True
    elif chk_p_same(e1[0], e2[1]) and chk_p_same(e1[1], e2[0]):
        return True
    else:
        return False


def chk_dir_edge_same(e1, e2):
    if chk_p_same(e1[0], e2[0]) and chk_p_same(e1[1], e2[1]):
        return True
    else:
        return False


def chk_p_same(p1, p2):
    return abs(p1[0] - p2[0]) < EPS and abs(p1[1] - p2[1]) < EPS


def chk_is_triangle(tri_contour):
    p1 = np.array(tri_contour[0])
    p2 = np.array(tri_contour[1])
    p3 = np.array(tri_contour[2])
    v1 = p1 - p2
    v2 = p3 - p2

    if alg.cross(v1, v2) == 0:
        return False
    else:
        return True


