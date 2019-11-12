import numpy as np
import math
from gym import spaces
from ..miniworld import ContinuousMiniworldEnv, MiniWorldEnv, Room
from ..entity import Box


class FourRooms(MiniWorldEnv):
    """
    Classic four rooms environment.
    The agent must reach the red box to get a reward.
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=250,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=1, max_z=7,
            wall_tex='acustom_wall',
            floor_tex='concrete_tiles',
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=1, max_z=7,
            wall_tex='bcustom_wall',
            floor_tex='concrete_tiles',

        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=-7, max_z=-1,
            wall_tex='ccustom_wall',
            floor_tex='concrete_tiles',

        )
        # Bottom-left room
        room3 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1,
            wall_tex='dcustom_wall',
            floor_tex='concrete_tiles',
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        self.box = self.place_entity(Box(color='red', invisible=True))

        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info


class ContinuousFourRooms(ContinuousMiniworldEnv):
    """
    Classic four rooms environment.
    The agent must reach the red box to get a reward.
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=250,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Box(shape=(1,), low=-1, high=1)

    def _gen_world(self, prob_constraint, min_goal_dist, max_goal_dist):
        # Top-left room
        room0 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=1, max_z=7,
            wall_tex='acustom_wall',
            floor_tex='concrete_tiles',
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=1, max_z=7,
            wall_tex='bcustom_wall',
            floor_tex='concrete_tiles',

        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=-7, max_z=-1,
            wall_tex='ccustom_wall',
            floor_tex='concrete_tiles',

        )
        # Bottom-left room
        room3 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1,
            wall_tex='dcustom_wall',
            floor_tex='concrete_tiles',
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        self.place_agent()

        # With probability prob_constraint, set box within goal difficulty constraints, i.e., nearby the agent
        if np.random.random() < prob_constraint:
            self.box = self.place_entity_nearby(Box(color='red', invisible=True), self.agent.pos, min_goal_dist, max_goal_dist, share_room=True)
        else:
            self.box = self.place_entity(Box(color='red', invisible=True))

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info
