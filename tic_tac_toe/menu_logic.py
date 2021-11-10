import pygame
from game import *
from play import *



def check_change_mode(pos, current_state):

    x, y = pos

    if 210 <= x <= 310 and 55<=y<=105:
        if current_state:
            return True

    if 520 <= x <= 620 and 55<=y<=105:
        if not current_state:
            return True

    return False
class Game():
    def __init__(self):
        pygame.init()
        self.running, self.playing = True, False
        self.UP_KEY, self.DOWN_KEY, self.START_KEY, self.BACK_KEY = False, False, False, False
        self.DISPLAY_W, self.DISPLAY_H = 800, 800
        self.display = pygame.Surface((self.DISPLAY_W,self.DISPLAY_H))
        self.window = pygame.display.set_mode([800, 800])
        self.font_name = '8-BIT WONDER.TTF'
        #self.font_name = pygame.font.get_default_font()
        self.BLACK, self.WHITE = (0, 0, 0), (255, 255, 255)
        self.YELLOW = (240, 226, 123)
        self.main_menu = MainMenu(self)
        self.credits = CreditsMenu(self)
        self.curr_menu = MainMenu(self)

    def game_loop(self):
        if self.playing:
            self.window = pygame.display.set_mode([800, 800])
            obs = env.reset()
            PLAYER_TURN = True
            AI_TURN = False
            DONE = False
            IMPOSSIBLE = True
            font = pygame.font.Font(self.font_name,25) 
            normal = font.render('Normal', True, self.YELLOW)
            impossible = font.render('Impossible', True, self.YELLOW)
            self.window.blit(normal, (200, 65))
            self.window.blit(impossible, (500, 65))
            
            while self.playing:
                
                if DONE:
                    obs = env.reset()
                    DONE = False
                    AI_TURN = True
                    PLAYER_TURN = False

                clock.tick(5)
                env.render()
                if AI_TURN and not DONE:
                    if IMPOSSIBLE:
                        action = ai_action(obs, "IMPOSSIBLE")
                        print("BEST MOVE BOT")
                    else:
                        action, _ = ai_action(obs, "NORMAL")
    
                    if env.isValid(action):
                        obs, reward, DONE, info = env.step(action,1)
                        print(reward)
                        AI_TURN = False
                        PLAYER_TURN = True
                    env.render()


                if PLAYER_TURN and not DONE:
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONDOWN :
                                pos = pygame.mouse.get_pos()
                                if check_change_mode(pos, IMPOSSIBLE):
                                    IMPOSSIBLE = not IMPOSSIBLE
                                player_move = player_action(pos)
                                obs, reward, DONE, info = env.step(player_move,2)
                                AI_TURN = True
                                PLAYER_TURN = False
                                env.render()

                



    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running, self.playing = False, False
                self.curr_menu.run_display = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.START_KEY = True
                if event.key == pygame.K_BACKSPACE:
                    self.BACK_KEY = True
                if event.key == pygame.K_DOWN:
                    self.DOWN_KEY = True
                if event.key == pygame.K_UP:
                    self.UP_KEY = True

    def reset_keys(self):
        self.UP_KEY, self.DOWN_KEY, self.START_KEY, self.BACK_KEY = False, False, False, False

    def draw_text(self, text, size, x, y):
        font = pygame.font.Font(self.font_name,size)
        text_surface = font.render(text, True, self.YELLOW)
        text_rect = text_surface.get_rect()
        text_rect.center = (x,y)
        self.display.blit(text_surface,text_rect)




