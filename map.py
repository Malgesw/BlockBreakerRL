from typing import Tuple

import pygame


class BlockGrid:
    def __init__(self, rows, columns, width, height, padding=0, offset=0):
        self.block_width = width / columns - padding
        self.block_height = height / rows - padding
        self.blocks = []
        self.rows = rows
        self.columns = columns
        self.padding = padding
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

    def draw(self, screen: pygame.Surface):
        for i in range(self.rows):
            for j in range(self.columns):
                if self.valid[i][j]:
                    pygame.draw.rect(screen, self.get_row_color(i), self.blocks[i][j])

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
