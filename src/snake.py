import pygame
import random
import threading
import src.voice_control as voice_control

pygame.init()

voice_thread = None

snake_speed = 5
window_x = 1090
window_y = 720

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)

game_window = pygame.display.set_mode((window_x, window_y))
pygame.display.set_caption('VoiceCommand Snakes')
fps = pygame.time.Clock()

def draw_button(rect, text, mouse, active_color, inactive_color):
    color = active_color if rect.collidepoint(mouse) else inactive_color
    pygame.draw.rect(game_window, color, rect)
    font = pygame.font.SysFont('times new roman', 30)
    text_surface = font.render(text, True, black)
    text_rect = text_surface.get_rect(center=rect.center)
    game_window.blit(text_surface,text_rect)

def main_menu():
    while True:
        game_window.fill(black)
        mouse = pygame.mouse.get_pos()

        # Tittel
        title_font = pygame.font.SysFont('times new roman', 60)
        title_surface = title_font.render("Voice Snake", True, green)
        title_rect = title_surface.get_rect(center=(window_x // 2, window_y // 4))
        game_window.blit(title_surface, title_rect)

        # Knapper
        button_width, button_height = 200, 50
        start_rect = pygame.Rect(window_x // 2 - 100, window_y // 2 - 30, button_width, button_height)
        quit_rect = pygame.Rect(window_x // 2 - 100, window_y // 2 + 40, button_width, button_height)

        draw_button(start_rect, "Start Game", mouse, green, white)
        draw_button(quit_rect, "Quit", mouse, red, white)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if start_rect.collidepoint(mouse):
                    return
                elif quit_rect.collidepoint(mouse):
                    pygame.quit()
                    quit()

def show_score(score, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)

def game_loop():
    global voice_thread

    # Start voice control thread
    voice_control.voice_active = True
    voice_thread = threading.Thread(target=voice_control.voice_control)
    voice_thread.daemon = True
    voice_thread.start()

    direction = 'RIGHT'
    voice_control.change_to = direction
    score = 0
    snake_position = [100, 50]
    snake_body = [ [100, 50], [90, 50], [80, 50], [70, 50] ]
    fruit_position = [random.randrange(1, (window_x // 10)) * 10,
                      random.randrange(1, (window_y // 10)) * 10]
    fruit_spawn = True

    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
    
        # Validation direction
        if voice_control.change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if voice_control.change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if voice_control.change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if voice_control.change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'
        
        
        if direction == 'UP':
            snake_position[1] -= 10
        if direction == 'DOWN':
            snake_position[1] += 10
        if direction == 'LEFT':
            snake_position[0] -= 10
        if direction == 'RIGHT':
            snake_position[0] += 10
        
        snake_body.insert(0, list(snake_position))
        if snake_position == fruit_position:
            score += 10
            fruit_spawn = False
        else:
            snake_body.pop()

        if not fruit_spawn:
            fruit_position = [random.randrange(1, (window_x//10)) * 10,
                            random.randrange(1, (window_y//10)) * 10]
        fruit_spawn = True

        game_window.fill(black)
        for pos in snake_body:
            pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(game_window, white, pygame.Rect(fruit_position[0], fruit_position[1], 10, 10))

        if snake_position[0] < 0 or snake_position[0] > window_x-10 or snake_position[1] < 0 or snake_position[1] > window_y-10:
            return game_over(score)

        for block in snake_body[1:]:
            if snake_position == block:
                return game_over(score)

        show_score(score, white, 'times new roman', 20)
        pygame.display.update()
        fps.tick(snake_speed)


def game_over(score):
    voice_control.voice_active = False
    if voice_thread is not None:
        voice_thread.join()

    while True:
        game_window.fill(black)
        mouse = pygame.mouse.get_pos()

        # Game Over-tittel
        title_font = pygame.font.SysFont('times new roman', 60)
        title_surface = title_font.render("Game Over", True, red)
        title_rect = title_surface.get_rect(center=(window_x // 2, window_y // 4))
        game_window.blit(title_surface, title_rect)

        # Score
        score_font = pygame.font.SysFont('times new roman', 35)
        score_surface = score_font.render(f"Your score is: {score}", True, white)
        score_rect = score_surface.get_rect(center=(window_x // 2, window_y // 3 + 40))
        game_window.blit(score_surface, score_rect)

        # Knapper
        button_width, button_height = 200, 50
        retry_rect = pygame.Rect(window_x // 2 - 100, window_y // 2, button_width, button_height)
        quit_rect = pygame.Rect(window_x // 2 - 100, window_y // 2 + 70, button_width, button_height)

        draw_button(retry_rect, "Try again", mouse, green, white)
        draw_button(quit_rect, "Quit", mouse, red, white)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if retry_rect.collidepoint(mouse):
                    return True
                elif quit_rect.collidepoint(mouse):
                    return False