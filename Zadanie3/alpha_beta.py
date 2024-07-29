import numpy as np
from tic_tac_toe import TicTacToe
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import time

NODES_PRUNED = 0


def evaluate_state(state: TicTacToe) -> float:
    """
    Evaluates a game state from the perspective of the maximizing player.

    Args:
        state (TicTacToe): The game state to evaluate.

    Returns:
        float: The evaluation score of the game state.
    """
    if state.get_winner() == 1:
        return 1
    elif state.get_winner() == -1:
        return -1
    else:
        return 0.2 * calculate_potential(state)


def calculate_potential(state: TicTacToe) -> float:
    """
    Calculates the potential of a game state.

    Args:
        state (TicTacToe): The game state to evaluate.

    Returns:
        float: The potential score of the game state.
    """
    score = 0
    for row in state.board:
        if sum(row) == 2:
            score += 1
        elif sum(row) == -2:
            score -= 1
    for col in state.board.T:
        if sum(col) == 2:
            score += 1
        elif sum(col) == -2:
            score -= 1
    diag1 = state.board.diagonal()
    diag2 = np.fliplr(state.board).diagonal()
    for diag in [diag1, diag2]:
        if sum(diag) == 2:
            score += 1
        elif sum(diag) == -2:
            score -= 1
    return score




def alpha_beta_search(
    state: TicTacToe,
    depth: int,
    alpha: float = -np.inf,
    beta: float = np.inf,
    maximizing_player: bool = True,
) -> tuple[float, dict]:
    """
    Performs an alpha-beta search algorithm to find the best move for a given state in a game.

    Args:
        state (TicTacToe): The current state of the game.
        depth (int): The maximum depth to search in the game tree.
        alpha (float, optional): The alpha value for alpha-beta pruning. Defaults to -np.inf.
        beta (float, optional): The beta value for alpha-beta pruning. Defaults to np.inf.
        maximizing_player (bool, optional): Indicates whether the current player is maximizing or minimizing.
            Defaults to True.

    Returns:
        tuple[float, dict]: A tuple containing the evaluation value of the best move and a dictionary of
            move evaluations for debugging purposes.
    """
    global NODES_PRUNED

    if depth == 0 or state.get_done():
        return state.check_winner(), None, {}
    moves_evaluations = {}

    if maximizing_player:
        max_eval = -np.inf
        best_move = None
        for move in state.get_valid_moves():
            state_copy = deepcopy(state)
            state_copy.step(move)
            eval, _, _ = alpha_beta_search(state_copy, depth - 1, alpha, beta, False)

            if eval > max_eval:
                max_eval = eval
                best_move = move
            moves_evaluations[tuple(move)] = eval
            if beta < max_eval:
                NODES_PRUNED += 1
                break
            alpha = max(alpha, max_eval)
        return max_eval, best_move, moves_evaluations
    else:
        min_eval = np.inf
        best_move = None
        for move in state.get_valid_moves():
            state_copy = deepcopy(state)
            state_copy.step(move)
            eval, _, _ = alpha_beta_search(state_copy, depth - 1, alpha, beta, True)
            moves_evaluations[tuple(move)] = eval
            if eval < min_eval:
                min_eval = eval
                best_move = move
            if min_eval < alpha:
                NODES_PRUNED += 1
                break
            beta = min(beta, min_eval)
        return min_eval, best_move, moves_evaluations


def main():
    global NODES_PRUNED
    game = TicTacToe()
    win_percentages = []

    while not game.get_done():
        if game.get_turn() == 1:
            tic = time.time()
            _, move, evals = alpha_beta_search(game, depth=10, maximizing_player=True)
            tac = time.time()
            print(f"Time taken: {tac - tic} seconds")
            print(f"Nodes pruned: {NODES_PRUNED}")
            NODES_PRUNED = 0

            max_value = max(evals.values())
            max_keys = [k for k, v in evals.items() if v == max_value]
            move = random.choice(max_keys)
            print(evals)
            print(f"Move: {move}")
            game.step(move)
        else:
            tic = time.time()
            _, moves, evals = alpha_beta_search(game, depth=10, maximizing_player=False)
            tac = time.time()
            print(f"Time taken: {tac - tic} seconds")
            print(f"Nodes pruned: {NODES_PRUNED}")
            NODES_PRUNED = 0
            print(evals)
            print(f"Move: {move}")
            min_value = min(evals.values())
            min_keys = [k for k, v in evals.items() if v == min_value]
            move = random.choice(min_keys)
            game.step(move)
        game.print_board()


if __name__ == "__main__":
    main()
