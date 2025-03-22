from src.snake import main_menu, game_loop

if __name__ == "__main__":
    main_menu()
    keep_playing = True
    while keep_playing:
        keep_playing = game_loop()