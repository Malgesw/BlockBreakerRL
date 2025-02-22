import math
import sys

import pygame

from button import Button
from map import Ball, BlockGrid

# Constants
BACKGROUND_COLOR = (0, 0, 0)
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
MAX_COMPLETIONS = 3


def update_score_display(font, score):
    return font.render(f"{score:04d}", True, (255, 255, 255))


def update_lives_display(font, lives):
    return font.render(f"{lives:02d}", True, (255, 255, 255))


def circle_rect_collision(cx, cy, r, rect):
    """
    Return True if the circle (center (cx, cy), radius r)
    collides with the rectangle (a pygame.Rect).
    """
    # Find the closest point on the rectangle to the circle's center.
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top, min(cy, rect.bottom))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= r * r


def check_block_collision(ball, blockGrid, sign_x, sign_y, score):
    """
    Check for a collision between the ball and any block in the column
    corresponding to the ball's x-position.
    If a collision is detected, adjust the ball's position and velocity,
    mark the block as destroyed (set valid to False), and increment score.
    Returns True if a collision was handled.
    """

    # Only check if the ball is within the vertical area of the block grid
    if ball.center_y - ball.r > blockGrid.height + blockGrid.offset:
        return False, sign_x, sign_y, score

    # Determine which column the ball is over
    j = int(ball.center_x // (blockGrid.block_width + blockGrid.padding))
    j = max(0, min(j, blockGrid.columns - 1))

    # Decide the order in which to check rows:
    row_range = (
        range(blockGrid.rows - 1, -1, -1) if sign_y < 0 else range(blockGrid.rows)
    )

    for i in row_range:
        if blockGrid.valid[i][j]:
            block = blockGrid.blocks[i][j]
            if circle_rect_collision(ball.center_x, ball.center_y, ball.r, block):
                # Determine whether the collision is more horizontal or vertical
                dx = ball.center_x - block.centerx
                dy = ball.center_y - block.centery
                if abs(dx) > abs(dy):
                    # Horizontal collision:
                    if dx > 0:
                        # Ball is to the right -> hit the block's right side
                        ball.center_x = block.right + ball.r
                    else:
                        # Ball is to the left -> hit the block's left side
                        ball.center_x = block.left - ball.r
                    sign_x = -sign_x
                else:
                    # Vertical collision
                    if dy > 0:
                        ball.center_y = block.bottom + ball.r
                    else:
                        ball.center_y = block.top - ball.r
                    sign_y = -sign_y
                blockGrid.valid[i][j] = False
                score += 1
                return True, sign_x, sign_y, score  # Only handle one block per frame
    return False, sign_x, sign_y, score


def check_collisions(
    ball,
    platform,
    blockGrid,
    sign_x,
    sign_y,
    score,
    lives,
    started_moving,
    mouse_held,
    game_over,
    speed_x,
    speed_y,
    ignore_pressure,
):
    # ----- Wall collisions -----
    if ball.center_x + ball.r >= SCREEN_WIDTH:
        ball.center_x = SCREEN_WIDTH - ball.r
        sign_x = -sign_x
    if ball.center_x - ball.r <= 0:
        ball.center_x = ball.r
        sign_x = -sign_x
    if ball.center_y - ball.r <= blockGrid.offset:
        ball.center_y = ball.r + blockGrid.offset
        sign_y = -sign_y

    # ----- Block collisions -----
    collided, sign_x, sign_y, score = check_block_collision(
        ball, blockGrid, sign_x, sign_y, score
    )
    if collided:
        # A block was hit; do not check further in this frame.
        pass

    # ----- Bottom wall (game over condition) -----
    if ball.center_y + ball.r >= SCREEN_HEIGHT:
        lives -= 1
        if lives == 0:
            game_over = True
        mouse_held = False
        started_moving = False
        sign_x, sign_y, ignore_pressure = reset_positions(
            platform, ball, sign_x, sign_y, ignore_pressure
        )

    # ----- Platform (paddle) collision -----
    if (
        platform.left < ball.center_x < platform.right
        and ball.center_y + ball.r >= platform.top
    ):
        ball.center_y = platform.top - ball.r
        center_distance = (ball.center_x - platform.centerx) / platform.width
        speed_x = abs(center_distance) * 8
        # Keep overall speed constant
        speed_y = math.sqrt(18 - speed_x * speed_x)
        sign_x = 1 if center_distance >= 0 else -1
        sign_y = -sign_y

    # ----- Platform side collisions (if needed) -----
    if platform.top < ball.center_y < platform.bottom:
        if ball.center_x + ball.r >= platform.left and ball.center_x <= platform.left:
            ball.center_x = platform.left - ball.r
            sign_x = -sign_x
        elif (
            ball.center_x - ball.r <= platform.right and ball.center_x >= platform.right
        ):
            ball.center_x = platform.right + ball.r
            sign_x = -sign_x

    return (
        sign_x,
        sign_y,
        score,
        lives,
        started_moving,
        mouse_held,
        game_over,
        speed_x,
        speed_y,
        ignore_pressure,
    )


# Reset ball and platform positions
def reset_positions(platform: pygame.Rect, ball: Ball, sign_x, sign_y, ignore_pressure):
    platform.topleft = (240, 550)
    ball.center_x = platform.centerx
    ball.center_y = platform.centery - 30
    sign_x = 1
    sign_y = -1
    ignore_pressure = True

    return sign_x, sign_y, ignore_pressure


# Function to check if mouse is over platform
def is_hover(mouse_x, mouse_y, target: pygame.Rect):
    return (
        0 < mouse_x - target.left < target.width
        and 0 < mouse_y - target.top < target.height
    )


def main():
    # Initialize Pygame
    pygame.init()
    pygame.font.init()

    # Set up screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BlockBreaker")
    font = pygame.font.Font("./ARCADECLASSIC.TTF", 25)

    # Game Objects
    blockGrid = BlockGrid(7, 10, SCREEN_WIDTH, SCREEN_HEIGHT // 2, padding=5, offset=70)
    platform = pygame.Rect(240, 550, 120, 30)
    ball = Ball(platform.centerx, platform.centery - 30, 15, (255, 255, 255))
    restartButton_center_x = SCREEN_WIDTH // 2
    restartButton_center_y = SCREEN_HEIGHT // 2 + 25
    restartButton = Button(
        restartButton_center_x - 60,
        restartButton_center_y + 30,
        120,
        60,
        (0, 0, 0),
        "Restart",
        (255, 255, 255),
        (255, 255, 255),
        2,
        font,
    )

    # Game variables
    distance = 0
    score = 0
    lives = 3
    mouse_held = False
    started_moving = False
    sign_x = 1
    sign_y = -1
    game_over = False
    speed_x = 3
    speed_y = 3
    completions = 0
    completed = False
    ignore_pressure = True
    previous_time = pygame.time.get_ticks()

    text_surface = update_score_display(font, score)
    text_rect = text_surface.get_rect(
        center=(3 * SCREEN_WIDTH // 4, blockGrid.offset // 2)
    )
    text_surface_l = update_lives_display(font, lives)
    text_rect_l = text_surface_l.get_rect(
        center=(SCREEN_WIDTH // 4, blockGrid.offset // 2)
    )

    # Game Loop
    running = True
    while running:
        screen.fill(BACKGROUND_COLOR)

        current_time = pygame.time.get_ticks()
        dt = (current_time - previous_time) / 1000
        previous_time = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if (
            not game_over
            and score != MAX_COMPLETIONS * blockGrid.rows * blockGrid.columns
            and not completed
        ):
            # Mouse input
            mouse_pressed = pygame.mouse.get_pressed()
            if ignore_pressure:
                if not mouse_pressed[0]:
                    ignore_pressure = False
                continue
            mouse_x, mouse_y = pygame.mouse.get_pos()

            if (
                is_hover(mouse_x, mouse_y, platform)
                and mouse_pressed[0]
                and not mouse_held
            ):
                mouse_held = True
                distance = mouse_x - platform.left

            if not mouse_pressed[0]:
                mouse_held = False

            # Move platform with mouse
            if mouse_held:
                new_x = mouse_x - distance
                new_x = max(0, min(new_x, SCREEN_WIDTH - platform.width))
                old_x = platform.left
                platform.topleft = (new_x, platform.top)

                if not started_moving and new_x != old_x:
                    sign_x = -1 if new_x - old_x >= 0 else 1
                    started_moving = True

            # Check collisions
            if started_moving:
                (
                    sign_x,
                    sign_y,
                    score,
                    lives,
                    started_moving,
                    mouse_held,
                    game_over,
                    speed_x,
                    speed_y,
                    ignore_pressure,
                ) = check_collisions(
                    ball,
                    platform,
                    blockGrid,
                    sign_x,
                    sign_y,
                    score,
                    lives,
                    started_moving,
                    mouse_held,
                    game_over,
                    speed_x,
                    speed_y,
                    ignore_pressure,
                )

            # Move ball if the game has started
            if started_moving:
                ball.center_x += sign_x * speed_x * dt * 60 * (completions * 0.5 + 1)
                ball.center_y += sign_y * speed_y * dt * 60 * (completions * 0.5 + 1)

            # Check if the current level is cleared
            if score == (completions + 1) * blockGrid.rows * blockGrid.columns:
                completed = True

            # Update UI
            text_surface = update_score_display(font, score)
            text_surface_l = update_lives_display(font, lives)
            screen.blit(text_surface_l, text_rect_l)
            screen.blit(text_surface, text_rect)
            blockGrid.draw(screen)
            ball.draw(screen)
            pygame.draw.rect(screen, (255, 255, 255), platform)

            pygame.display.flip()
            # pygame.time.Clock().tick(120)  # 60fps

        else:
            if game_over:
                t_surface = font.render("Game Over!", True, (255, 255, 255))
            else:
                t_surface = font.render("You Win!", True, (255, 255, 255))

            t_rect = t_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(text_surface, text_rect)
            screen.blit(text_surface_l, text_rect_l)

            if completions == MAX_COMPLETIONS or game_over:
                screen.blit(t_surface, t_rect)
                restartButton.draw(screen)
            pygame.display.flip()

            if completions != MAX_COMPLETIONS and completed:
                completions += 1
                completed = False
            elif restartButton.button_pressed():
                game_over = False
                score = 0
                lives = 3
                completions = 0

            started_moving = False
            blockGrid.reset()
            sign_x, sign_y, ignore_pressure = reset_positions(
                platform, ball, sign_x, sign_y, ignore_pressure
            )

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
