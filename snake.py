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
model = models.load_model("voice_model.keras")
labels = ['down', 'left', 'right', 'up']

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
    audio = record_audio()
    spec = get_spectrogram(audio)
    spec = tf.expand_dims(spec, 0)
    prediction = model(spec)
    predicted_index = tf.argmax(prediction[0]).numpy()
    return labels[predicted_index]

def voice_control():
    global change_to
    while True:
        command = predict_command()
        if command in ['up', 'down', 'left', 'right']:
            change_to = command.upper()
        time.sleep(0.5)


# === Game Setup ===

snake_speed = 10
window_x = 720
window_y = 480

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

pygame.init()
pygame.display.set_caption('VoiceCommand Snakes')
game_window = pygame.display.set_mode((window_x, window_y))
fps = pygame.time.Clock()

snake_position = [100, 50]
snake_body = [  [100, 50], [90, 50], [80, 50], [70, 50] ]

fruit_position = [random.randrange(1, (window_x//10)) * 10,
                  random.randrange(1, (window_y//10))* 10]
fruit_spawn = True

direction = 'RIGHT'
change_to = direction

score = 0

def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)

# Start voice control thread
voice_thread = threading.Thread(target=voice_control)
voice_thread.daemon = True
voice_thread.start()

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
    
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
        pygame.quit()
        quit()

    for block in snake_body[1:]:
        if snake_position == block:
            pygame.quit()
            quit()

    show_score(1, white, 'times new roman', 20)
    pygame.display.update()
    fps.tick(snake_speed)