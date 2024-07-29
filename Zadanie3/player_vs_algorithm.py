from alpha_beta import alpha_beta_search
from tic_tac_toe import TicTacToe
import numpy as np
import random




def main():

    game = TicTacToe()
    ans = input("Do you want to play first? (y/n): ").lower()
    if ans == 'y':
        player = 1
    else:
        player = -1
    maximizing_player = False if player == 1 else True
    while not game.get_done():
        if player == 1:
            game.print_board()
            try:
                move = tuple(map(int, input("Enter move (row, col): ").split(',')))
                game.step(move)
            except AssertionError:
                continue
            player = -1
        else:
            _, move, evals = alpha_beta_search(game, depth=10, maximizing_player=maximizing_player, alpha=-np.inf, beta=np.inf)
            print(evals)
            game.step(move)
            player = 1



if __name__ == "__main__":
    main()