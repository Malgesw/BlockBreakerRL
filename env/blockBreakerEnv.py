import math
import sys
from typing import Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback


class BlockBreakerEnv(gym.Env):
    """
    Custom Gym environment emulating the game BlockBreaker
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(BlockBreakerEnv, self).__init__()
        self.render_mode = render_mode

        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = 600
        self.BACKGROUND_COLOR = (0, 0, 0)
        self.MAX_COMPLETIONS = 3

        self.score = 0
        self.lives = 3
        self.completions = 0
        self.game_over = False
        self.speed_x = 3
        self.speed_y = 3
        self.sign_x = 1
        self.sign_y = -1
        self.bounced = False

        self.blockGrid = self.BlockGrid(
            4,
            7,
            self.SCREEN_WIDTH,
            self.SCREEN_HEIGHT // 2,
            padding=5,
            offset=70,
        )
        self.platform = pygame.Rect(240, 550, 120, 30)
        self.ball = self.Ball(
            self.platform.centerx, self.platform.centery - 30, 15, (255, 255, 255)
        )

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        # platform pos x, ball pos x, ball pos y, ball speed x, ball speed y, blocks validity matrix
        self.observation_space = spaces.Dict(
            {
                "platform": spaces.Box(
                    low=0, high=self.SCREEN_WIDTH, shape=(1,), dtype=np.float32
                ),
                "ball": spaces.Box(
                    low=np.array([0, 0, -5, -5]),
                    high=np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT, 5, 5]),
                    shape=(4,),
                    dtype=np.float32,
                ),
                "blocks": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.blockGrid.rows, self.blockGrid.columns),
                    dtype=np.uint8,
                ),
                "score": spaces.Box(
                    low=0,
                    high=self.MAX_COMPLETIONS
                    * self.blockGrid.rows
                    * self.blockGrid.columns,
                    shape=(1,),
                    dtype=np.uint32,
                ),
                "lives": spaces.Box(low=0, high=3, shape=(1,), dtype=np.uint8),
                "completions": spaces.Box(
                    low=0, high=self.MAX_COMPLETIONS, shape=(1,), dtype=np.uint8
                ),
            }
        )

        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("BlockBreakerRL")
        self.font = pygame.font.Font("ARCADECLASSIC.TTF", 30)

        self.text_surface = self.update_score_display()
        self.text_rect = self.text_surface.get_rect(
            center=(3 * self.SCREEN_WIDTH // 4, self.blockGrid.offset // 2)
        )
        self.text_surface_l = self.update_lives_display()
        self.text_rect_l = self.text_surface_l.get_rect(
            center=(self.SCREEN_WIDTH // 4, self.blockGrid.offset // 2)
        )

        self.reset_positions()

    def update_score_display(self):
        return self.font.render(f"{self.score:04d}", True, (255, 255, 255))

    def update_lives_display(self):
        return self.font.render(f"{self.lives:02d}", True, (255, 255, 255))

    def ball_rect_collision(self, rect):
        """
        Return True if the circle (center (cx, cy), radius r)
        collides with the rectangle (a pygame.Rect).
        """
        # Find the closest point on the rectangle to the circle's center.
        closest_x = max(rect.left, min(self.ball.center_x, rect.right))
        closest_y = max(rect.top, min(self.ball.center_y, rect.bottom))
        dx = self.ball.center_x - closest_x
        dy = self.ball.center_y - closest_y
        return dx * dx + dy * dy <= self.ball.r * self.ball.r

    def check_block_collision(self):
        """
        Check for a collision between the ball and any block in the column
        corresponding to the ball's x-position.
        If a collision is detected, adjust the ball's position and velocity,
        mark the block as destroyed (set valid to False), and increment score.
        Returns True if a collision was handled.
        """

        # Only check if the ball is within the vertical area of the block grid
        if (
            self.ball.center_y - self.ball.r
            > self.blockGrid.height + self.blockGrid.offset
        ):
            return False

        # Determine which column the ball is over
        j = int(
            self.ball.center_x // (self.blockGrid.block_width + self.blockGrid.padding)
        )
        j = max(0, min(j, self.blockGrid.columns - 1))

        # Decide the order in which to check rows
        row_range = (
            range(self.blockGrid.rows - 1, -1, -1)
            if self.sign_y < 0
            else range(self.blockGrid.rows)
        )

        for i in row_range:
            if self.blockGrid.valid[i][j]:
                block = self.blockGrid.blocks[i][j]
                if self.ball_rect_collision(block):
                    # Determine whether the collision is more horizontal or vertical
                    dx = self.ball.center_x - block.centerx
                    dy = self.ball.center_y - block.centery
                    if abs(dx) > abs(dy):
                        # Horizontal collision:
                        if dx > 0:
                            # Ball is to the right -> hit the block's right side
                            self.ball.center_x = block.right + self.ball.r
                        else:
                            # Ball is to the left -> hit the block's left side
                            self.ball.center_x = block.left - self.ball.r
                        self.sign_x = -self.sign_x
                    else:
                        # Vertical collision
                        if dy > 0:
                            self.ball.center_y = block.bottom + self.ball.r
                        else:
                            self.ball.center_y = block.top - self.ball.r
                        self.sign_y = -self.sign_y
                    self.blockGrid.valid[i][j] = False
                    self.score += 1
                    return True  # Only handle one block per frame
        return False

    def check_collisions(self):
        self.bounced = False
        # ----- Wall collisions -----
        if self.ball.center_x + self.ball.r >= self.SCREEN_WIDTH:
            self.ball.center_x = self.SCREEN_WIDTH - self.ball.r
            self.sign_x = -self.sign_x
        if self.ball.center_x - self.ball.r <= 0:
            self.ball.center_x = self.ball.r
            self.sign_x = -self.sign_x
        if self.ball.center_y - self.ball.r <= self.blockGrid.offset:
            self.ball.center_y = self.ball.r + self.blockGrid.offset
            self.sign_y = -self.sign_y

        # ----- Block collisions -----
        if self.check_block_collision():
            # A block was hit; do not check further in this frame.
            pass

        # ----- Bottom wall (game over condition) -----
        if self.ball.center_y + self.ball.r >= self.SCREEN_HEIGHT:
            self.lives -= 1
            if self.lives == 0:
                self.game_over = True
            self.reset_positions()

        # ----- Platform (paddle) collision -----
        if (
            self.platform.left < self.ball.center_x < self.platform.right
            and self.ball.center_y + self.ball.r >= self.platform.top
        ):
            self.ball.center_y = self.platform.top - self.ball.r
            center_distance = (
                self.ball.center_x - self.platform.centerx
            ) / self.platform.width
            self.speed_x = abs(center_distance) * 8
            # Keep overall speed constant
            self.speed_y = math.sqrt(18 - self.speed_x * self.speed_x)
            self.sign_x = 1 if center_distance >= 0 else -1
            self.sign_y = -self.sign_y
            self.bounced = True

        # ----- Platform side collisions (if needed) -----
        if self.platform.top < self.ball.center_y < self.platform.bottom:
            if (
                self.ball.center_x + self.ball.r >= self.platform.left
                and self.ball.center_x <= self.platform.left
            ):
                self.ball.center_x = self.platform.left - self.ball.r
                self.sign_x = -self.sign_x
            elif (
                self.ball.center_x - self.ball.r <= self.platform.right
                and self.ball.center_x >= self.platform.right
            ):
                self.ball.center_x = self.platform.right + self.ball.r
                self.sign_x = -self.sign_x

    def reset_positions(self):
        self.platform.topleft = (240, 550)
        self.ball.center_x = np.random.uniform(self.platform.left, self.platform.right)
        self.ball.center_y = self.platform.centery - 30
        self.sign_x = 1
        self.sign_y = -1
        self.speed_x = 3
        self.speed_y = 3

    def reset(self, seed=None, options=None):
        """
        Reset the environment
        """
        super().reset(seed=seed)

        self.previous_time = pygame.time.get_ticks()
        self.game_over = False
        self.blockGrid.reset()
        self.reset_positions()
        self.score = 0
        self.lives = 3
        self.completions = 0

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Method returning the observations
        """
        return {
            "platform": np.array([self.platform.left], dtype=np.float32),
            "ball": np.array(
                [
                    self.ball.center_x,
                    self.ball.center_y,
                    self.sign_x * self.speed_x,
                    self.sign_y * self.speed_y,
                ],
                dtype=np.float32,
            ),
            "blocks": np.array(self.blockGrid.get_valid(), dtype=np.uint8),
            "score": np.array([self.score], dtype=np.uint32),
            "lives": np.array([self.lives], dtype=np.uint8),
            "completions": np.array([self.completions], dtype=np.uint8),
        }

    def step(self, action):
        """
        Perform an action (e.g.: a movement of the paddle) and update the game state
        """

        new_time = pygame.time.get_ticks()
        dt = (new_time - self.previous_time) / 1000
        self.previous_time = new_time

        reward = 0
        done = False
        truncated = False
        info = {}

        platform_speed = 5 * float(action[0])
        self.platform.left += platform_speed * 60 * dt
        self.platform.left = max(
            0, min(self.platform.left, self.SCREEN_WIDTH - self.platform.width)
        )

        current_lives = self.lives
        current_score = self.score
        self.check_collisions()

        self.ball.center_x += (
            self.sign_x * self.speed_x * 60 * dt * (self.completions * 0.5 + 1)
        )
        self.ball.center_y += (
            self.sign_y * self.speed_y * 60 * dt * (self.completions * 0.5 + 1)
        )

        if (
            self.score
            == self.MAX_COMPLETIONS * self.blockGrid.rows * self.blockGrid.columns
            or self.game_over
        ):
            # episode (full game) ended
            done = True

        elif (
            self.score
            == (self.completions + 1) * self.blockGrid.rows * self.blockGrid.columns
        ):
            # current level is cleared
            self.completions += 1
            self.blockGrid.reset()
            self.reset_positions()

        if current_lives > self.lives:
            # the ball fell and a life is lost
            reward -= 10
        # reward -= 1.5 * (3 - self.lives)
        if self.score > current_score:
            reward += 7
        if self.score == current_score:
            reward -= 1
        if abs(platform_speed) > 1:
            reward += 0.5
        if self.bounced:
            reward += 3
        # dist = abs(self.ball.center_x - self.platform.centerx)
        # reward += max(0, 10 - dist / 10)

        obs = self._get_obs()  # get current observations after action is taken

        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        """
        Render the game window
        """
        self.screen.fill(self.BACKGROUND_COLOR)
        self.text_surface = self.update_score_display()
        self.text_surface_l = self.update_lives_display()
        self.screen.blit(self.text_surface_l, self.text_rect_l)
        self.screen.blit(self.text_surface, self.text_rect)
        self.blockGrid.draw(self.screen)
        self.ball.draw(self.screen)
        pygame.draw.rect(self.screen, (255, 255, 255), self.platform)
        pygame.display.flip()

    def close(self):
        """
        Close the game window
        """
        pygame.quit()
        sys.exit()

    class BlockGrid:
        def __init__(self, rows, columns, width, height, padding=0, offset=0):
            self.block_width = width / columns - padding
            self.block_height = height / rows - padding
            self.blocks = []
            self.rows = rows
            self.columns = columns
            self.padding = padding
            self.offset = offset
            self.width = width
            self.height = height
            self.valid = [[True for i in range(self.columns)] for j in range(self.rows)]

            for i in range(rows):
                row_blocks = []
                for j in range(columns):
                    rect = pygame.Rect(
                        j * (self.block_width + padding),
                        i * (self.block_height + padding) + offset,
                        self.block_width,
                        self.block_height,
                    )
                    row_blocks.append(rect)
                self.blocks.append(row_blocks)

        def get_row_color(self, row) -> Tuple:
            r = (row * 255 // self.rows) % 256
            g = ((row + 2) * 255 // self.rows) % 256
            b = ((row + 4) * 255 // self.rows) % 256
            return (r, g, b)

        def get_valid(self):
            return np.copy(self.valid)

        def draw(self, screen: pygame.Surface):
            for i in range(self.rows):
                for j in range(self.columns):
                    if self.valid[i][j]:
                        pygame.draw.rect(
                            screen, self.get_row_color(i), self.blocks[i][j]
                        )

        def reset(self):
            for i in range(self.rows):
                for j in range(self.columns):
                    self.valid[i][j] = True

    class Ball:
        def __init__(self, center_x, center_y, radius, color):
            self.center_x = center_x
            self.center_y = center_y
            self.r = radius
            self.color = color

        def draw(self, screen: pygame.Surface):
            pygame.draw.circle(
                screen, self.color, center=(self.center_x, self.center_y), radius=self.r
            )


class RendererCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RendererCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                print(
                    f"Cumulative reward: {episode_reward}, episode length: {episode_length}"
                )
        return True


# Register the environment
gym.envs.registration.register(
    id="BlockBreaker-v0",
    entry_point=BlockBreakerEnv,
    max_episode_steps=70000,
)
