import numpy as np
from copy import deepcopy


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.turn = 1
        self.winner = 0
        self.done = False
        self.moves = 0
        self.last_move = None

    def reset(self):
        self.board = np.zeros((3, 3))
        self.turn = 1
        self.winner = 0
        self.done = False
        self.moves = 0

    def step(self, action: tuple):
        assert self.board[action[0], action[1]] == 0
        self.last_move = action
        self.board[action[0], action[1]] = self.turn
        self.moves += 1
        if self.check_winner() == 0 and self.moves == 9:
            self.done = True
        elif self.check_winner() != 0:
            self.winner = self.turn
            self.done = True
        self.turn = -self.turn

    def check_winner(self):
        board = self.board
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2]:
                if board[i][0] != None:
                    return board[i][0]

            if board[0][i] == board[1][i] == board[2][i]:
                if board[0][i] != None:
                    return board[0][i]

        if board[0][0] == board[1][1] == board[2][2]:
            if board[0][0] != None:
                return board[0][0]
        if board[0][2] == board[1][1] == board[2][0]:
            if board[0][2] != None:
                return board[0][2]

        return 0

    def print_board(self):
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 1:
                    print("X", end=" ")
                elif self.board[i, j] == -1:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

    def copy(self):
        return deepcopy(self)

    def undo_last_move(self):
        action = self.last_move
        self.board[action[0], action[1]] = 0
        self.turn = -self.turn
        self.moves -= 1

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def get_state(self):
        return self.board

    def get_winner(self):
        return self.winner

    def get_turn(self):
        return self.turn

    def get_done(self):
        return self.done

    def get_moves(self):
        return self.moves


def main():
    game = TicTacToe()
    game.print_board()
    game.step((2, 2))
    game.print_board()
    game.step((1, 1))
    game.print_board()
    game.step((0, 0))
    game.print_board()
    game.step((1, 0))
    game.print_board()
    game.step((0, 2))
    game.print_board()
    game.step((1, 2))
    game.print_board()
    print(game.get_winner())
    print(game.get_done())
    print(game.get_moves())


if __name__ == "__main__":
    main()
