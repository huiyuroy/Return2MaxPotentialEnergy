
from enum import Enum

from core.space import *


class TrajectoryType(Enum):
    AbsoluteRoad = 1
    ApproxRoad = 2
    AbsoluteRand = 3


class Trajectory:

    def __init__(self):
        self.id = 0
        self.type = TrajectoryType.AbsoluteRand.value
        self.tar_num = 0
        self.tar_data = None
        self.targets = None
        self.start_idx = 0
        self.end_idx = -1  # 若大于总长度，自动选择最后一个目标

    def range_targets(self, start=0, end=-1):
        self.start_idx = start
        self.end_idx = end
        if self.end_idx == -1 or self.end_idx > self.tar_num:
            self.end_idx = self.tar_num - 1

        self.targets = self.tar_data[self.start_idx:self.end_idx + 1]
