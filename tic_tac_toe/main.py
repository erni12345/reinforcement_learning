"""

Ficher pour faire tourner le jeu
"""

from game import *


g = AlphaToe()

while g.running:
    """
    Demarer jeu
    """
    g.curr_menu.display_menu()
    g.game_loop()