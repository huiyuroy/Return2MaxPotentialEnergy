from core.space import *


class Boundary:
    def __init__(self):
        self.is_out_bound = False
        self.points = []
        self.points_num = 0
        self.center = np.array([0, 0])
        self.barycenter = []
        self.cir_rect = []
        self.orth_cir_rect = []

    def set_contour(self, out_boundary, points):
        self.is_out_bound = out_boundary
        self.points = points
        self.points_num = len(points)
        self.__calc_surround_rect()

    def clean_repeat(self):
        need_clean = []
        for i in range(len(self.points) - 1):
            for j in range(i + 1, len(self.points)):
                if geo.chk_p_same(self.points[i], self.points[j]):
                    need_clean.append(j)
        for idx in need_clean:
            self.points.pop(idx)
        self.points_num = len(self.points)

    def __calc_surround_rect(self):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        for px, py in self.points:
            if px <= min_x:
                min_x = px
            elif px >= max_x:
                max_x = px
            if py <= min_y:
                min_y = py
            elif py >= max_y:
                max_y = py
        self.orth_cir_rect = [min_x, min_y, max_x, max_y]




