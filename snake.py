import pygame 
import time
import random
import threading
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from tensorflow.keras import models

# === Voice Control Setup ===
model = models.load_model("voice_model.keras", compile=False)
labels = ['down', 'left', 'right', 'up']
voice_active = False

def record_audio():
    fs = 16000
    seconds = 1
    print("Listening...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def predict_command():
    try: 
        audio = record_audio()
        spec = get_spectrogram(audio)
        spec = tf.expand_dims(spec, 0)
        prediction = model.predict(spec)
        predicted_index = tf.argmax(prediction[0]).numpy()

        command = labels[predicted_index]
        print(f"Godkjent kommando: {command}")
        return command
    except Exception as e:
        print(f"Feil ved stemmegjenkjenning: {e}")
        return None

def voice_control():
    global change_to, voice_active
    while voice_active:
        command = predict_command()
        if command in ['up', 'down', 'left', 'right']:
            change_to = command.upper()
        # time.sleep(0.5)


# === Game Setup ===

snake_speed = 5
window_x = 1090
window_y = 720

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

pygame.init()
pygame.display.set_caption('VoiceCommand Snakes')
game_window = pygame.display.set_mode((window_x, window_y))
fps = pygame.time.Clock()

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

        pygame.draw.rect(game_window, green if start_rect.collidepoint(mouse) else white, start_rect)
        pygame.draw.rect(game_window, red if quit_rect.collidepoint(mouse) else white, quit_rect)

        font = pygame.font.SysFont('times new roman', 30)
        game_window.blit(font.render("Start Game", True, black), (start_rect.x + 35, start_rect.y + 10))
        game_window.blit(font.render("Quit", True, black), (quit_rect.x + 70, quit_rect.y + 10))

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

def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)

def game_loop():
    global direction, change_to, score, snake_position, snake_body, fruit_position, fruit_spawn, voice_active

    # Start voice control thread
    voice_active = True
    voice_thread = threading.Thread(target=voice_control)
    voice_thread.daemon = True
    voice_thread.start()

    direction = 'RIGHT'
    change_to = direction
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
                game_over(score)
    
        # Validation direction
        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
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
            game_over(score)

        for block in snake_body[1:]:
            if snake_position == block:
                game_over(score)

        show_score(1, white, 'times new roman', 20)
        pygame.display.update()
        fps.tick(snake_speed)

def game_over(score):
    global voice_active
    voice_active = False

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

        pygame.draw.rect(game_window, green if retry_rect.collidepoint(mouse) else white, retry_rect)
        pygame.draw.rect(game_window, red if quit_rect.collidepoint(mouse) else white, quit_rect)

        font = pygame.font.SysFont('times new roman', 30)
        game_window.blit(font.render("Try again", True, black), (retry_rect.x + 35, retry_rect.y + 10))
        game_window.blit(font.render("Quit", True, black), (quit_rect.x + 60, quit_rect.y + 10))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if retry_rect.collidepoint(mouse):
                    game_loop()
                    return
                elif quit_rect.collidepoint(mouse):
                    pygame.quit()
                    quit()

main_menu()
game_loop()