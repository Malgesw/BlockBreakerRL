import pygame


class Button:
    def __init__(
        self,
        left,
        top,
        width,
        height,
        main_color,
        text,
        text_color,
        border_color,
        border_thickness,
        font: pygame.font.Font,
    ):
        self.outer_rect = pygame.Rect(left, top, width, height)
        self.inner_rect = pygame.Rect(
            left + border_thickness,
            top + border_thickness,
            width - 2 * border_thickness,
            height - 2 * border_thickness,
        )
        self.main_color = main_color
        self.text_color = text_color
        self.border_color = border_color
        self.font = font
        self.text = text

    def button_pressed(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()
        return (
            self.outer_rect.left <= mouse_x <= self.outer_rect.right
            and self.outer_rect.top <= mouse_y <= self.outer_rect.bottom
            and mouse_pressed[0]
        )

    def draw(self, screen: pygame.Surface):
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.inner_rect.center)
        pygame.draw.rect(screen, self.border_color, self.outer_rect)
        pygame.draw.rect(screen, self.main_color, self.inner_rect)
        screen.blit(text_surface, text_rect)
