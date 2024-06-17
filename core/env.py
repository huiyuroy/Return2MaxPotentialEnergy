import os
import random
import operator
import numpy as np
import pickle
from enum import Enum
import pygame.gfxdraw

from core import *

from core.space.scene import Scene, DiscreteScene
from core.space.boundary import Boundary
from core.space.grid import Tiling
from core.space.trajectory import Trajectory

from util.file import load_json
from util.log import RDWTrajectoryWalkerLogger
from vis.ui.base import RDWWindow


class BaseEnv:
    """
    Base rdw environment. It contains several agents to experience rdw in common virtual and physical spaces. The
    virtual trajectories are necessary in the simulation.
    """

    def __init__(self):
        self.agents = []
        self.v_scene = None
        self.p_scene = None
        self.v_trajectories = None

    def add_agent(self, agent, agent_name='default'):
        """

        Args:
            agent: a well-constructed agent object.
            agent_name: custom name for the agent, mainly used for log.

        Returns:

        """
        self.agents.append(agent)
        agent.name = agent_name

    def init_agents_state(self, p_loc=None, p_fwd=None, v_loc=None, v_fwd=None):
        for ag in self.agents:
            ag.inputer.set_phy_init_state(p_loc, p_fwd, rand=p_loc is None or p_fwd is None)
            ag.inputer.set_vir_init_state(v_loc, v_fwd)

    def set_scenes(self, v_scene, p_scene):
        self.v_scene = v_scene
        self.p_scene = p_scene
        for ag in self.agents:
            ag.set_scenes(v_scene, p_scene)
            ag.gainer.obtain_agent_scenes()
            ag.rdwer.obtain_agent_scenes()
            ag.resetter.obtain_agent_scenes()
            ag.inputer.obtain_agent_scenes()

    def set_trajectories(self, trajs):
        self.v_trajectories = trajs

    def prepare(self):
        for ag in self.agents:
            ag.prepare()

    def reset(self):
        for ag in self.agents:
            ag.reset()

    def step(self):
        """
        Calling all agent update processes.

        Returns:

        """
        all_done = True
        for ag in self.agents:
            ag.step_early()
            ag.step()
            ag.step_late()
            next_s, done, truncated, info = ag.state_refresh()
            all_done = all_done and done
        return all_done


class RdwEnv(BaseEnv):
    """
    The rdw env containing ui and log. Recommend to modify when designing specific environment.

    """

    def __init__(self):
        super().__init__()
        cur_path = os.path.abspath(os.path.dirname(__file__))
        root_path = cur_path[:cur_path.find('Return2MaxPotentialEnergy') + len('Return2MaxPotentialEnergy')]
        self.ui_path = root_path + "\\vis\\ui_spec\\play_ui.json"

        self.env_ui: RDWWindow = None
        self.env_logs = []

    def load_logger(self):
        for ag in self.agents:
            ag_log = RDWTrajectoryWalkerLogger(ag)
            ag_log.set_agent(ag)
            ag_log.set_epi_data_buffer(100)
            self.env_logs.append(ag_log)

    def load_ui(self):
        self.env_ui = RDWWindow(self, load_json(self.ui_path))

    def set_current_trajectory(self, traj):
        """
        set the trajectory to all agents

        Args:
            traj:

        Returns:

        """
        for ag in self.agents:
            ag.inputer.set_trajectory(traj)

    def prepare(self):
        super().prepare()
        for ag_log in self.env_logs:
            ag_log.prepare_log()
        self.env_ui.prepare_surfs()
        self.env_ui.prepare_agents()

    def reset(self):
        super().reset()
        for ag_log in self.env_logs:
            ag_log.reset_epi_data()

        self.env_ui.reset_surfs()

    def step(self):
        all_done = super().step()
        self.env_ui.update_surfs()  # 3st cost
        return all_done

    def record(self):
        for ag_log in self.env_logs:
            ag_log.record_step_data()

    def output_epi_info(self):
        all_data = []
        for ag_log in self.env_logs:
            data = ag_log.log_epi_data()
            all_data.append(data)
        return all_data
