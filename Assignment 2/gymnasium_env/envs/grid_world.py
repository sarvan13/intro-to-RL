from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=12, start_pos=np.array([0,0]), end_pos=np.array([11,0])):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.step_count = 0

        self.close_wall_flag = False
        self.open_hole_flag = False
        self.moving_wall_flag = False
        self.moving_wall_dir = 1

        self.first_change = 500_000
        self.second_change = 1_500_000
        self.third_change = 3_000_000


        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        self.blocks = np.zeros(shape=(self.size,self.size))
        self.bottom_wall_row = self.size - 3
        self.bottom_wall_right = self.size - 2
        self.bottom_wall_left = 4
        for i in range(len(self.blocks)):
            if i < self.bottom_wall_left or i > self.bottom_wall_right:
                continue
            else:
                self.blocks[i][self.bottom_wall_row] = 1

        self.middle_wall_top = 0
        self.middle_wall_bot = self.bottom_wall_row - 3
        self.middle_wall_col = self.bottom_wall_left
        for i in range(len(self.blocks[self.middle_wall_col])):
            if i < self.middle_wall_top or i > self.middle_wall_bot:
                continue
            else:
                self.blocks[self.middle_wall_col][i] = 1

        self.right_wall_col = self.bottom_wall_right - 2
        self.right_wall_top = 2
        self.right_wall_bot = self.bottom_wall_row - 2
        for i in range(len(self.blocks[self.right_wall_col])):
            if i < self.right_wall_top or i > self.right_wall_bot:
                continue
            else:
                self.blocks[self.right_wall_col][i] = 1

        self.moving_wall_bot = self.bottom_wall_row
        self.moving_wall_length = 4 # The length is actual 1 + moving_wall_length
        self.moving_wall_col = 2
        


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return tuple(self._agent_location)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset agent and target locations
        self._agent_location = self.start_pos
        self._target_location = self.end_pos

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        next_location= np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if self.blocks[next_location[0], next_location[1]] == 0:
            self._agent_location = next_location

        
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        self.step_count += 1

        if self.step_count == self.first_change:
            self.close_wall_flag = True
            for i in range(len(self.blocks[self.middle_wall_col])):
                if i < self.bottom_wall_row and i > self.middle_wall_bot:
                    self.blocks[self.middle_wall_col][i] = 1
        if self.step_count == self.second_change:
            self.open_hole_flag = True
            self.blocks[self.middle_wall_col][2] = 0
        if self.step_count == self.third_change:
            self.moving_wall_flag = True

        if self.moving_wall_flag:
            self.moving_wall_bot = self.moving_wall_bot + self.moving_wall_dir
            if self.moving_wall_bot == self.size - 1:
                self.moving_wall_dir = -1
            elif self.moving_wall_bot == self.moving_wall_length:
                self.moving_wall_dir = 1
            # Add wall to blocks
            for i in range(len(self.blocks[self.moving_wall_col])):
                if i < (self.moving_wall_bot - self.moving_wall_length) or i > self.moving_wall_bot:
                    self.blocks[self.moving_wall_col][i] = 0
                else:
                    self.blocks[self.moving_wall_col][i] = 1

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw the blocks
        for i, col in enumerate(self.blocks):
            for j, val in enumerate(col):
                if val == 1:
                    location = np.array([i,j])

                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            pix_square_size * location,
                            (pix_square_size, pix_square_size),
                        ),
                    )


        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
