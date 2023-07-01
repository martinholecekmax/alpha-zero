import numpy as np

from ticktacktoe import TicTacToe
from monte_carlo_tree_search import MonteCarloTreeSearch

print("numpy version: ", np.__version__)

ticktacktoe = TicTacToe()
player = 1

args = {
    "C": 1.41,  # sqrt(2)
    "num_searches": 100,
}

mcts = MonteCarloTreeSearch(ticktacktoe, args)

state = ticktacktoe.get_initial_state()

while True:
    print("Player: ", player)
    print(state)

    if player == 1:
        valid_moves = ticktacktoe.get_valid_moves(state)
        print("Valid moves: ", [i for i in range(ticktacktoe.action_size) if valid_moves[i] == 1])
        action = int(input("Enter action: "))

        if valid_moves[action] == 0:
            print("Invalid move!")
            continue
    else:
        neutral_state = ticktacktoe.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)

    state = ticktacktoe.get_next_state(state, action, player)
    value, is_terminal = ticktacktoe.get_value_and_terminated(state, action)

    if is_terminal:
        print("Player: ", player)
        print(state)
        if value == 1:
            print("Player ", player, " wins!")
        else:
            print("Draw!")

        break

    player = ticktacktoe.get_opponent(player)
