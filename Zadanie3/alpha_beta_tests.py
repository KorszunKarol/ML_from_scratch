from tic_tac_toe import TicTacToe
from alpha_beta import alpha_beta_search
import numpy as np
import random



def test_alpha_beta():
    game = TicTacToe()
    game.board = np.array([[-1, 0, -1], [-1, 1, 1], [0, 0, 0]])
    game.turn = 1
    _, move, evals = alpha_beta_search(game, depth=10, maximizing_player=True)
    print(move)
    print(evals)
    print(game.print_board())


def main():
    test_alpha_beta()



if __name__ == "__main__":
    main()