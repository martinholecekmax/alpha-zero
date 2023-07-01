from ticktacktoe import TicTacToe
from connect_four import ConnectFour
import torch
from monte_carlo_tree_search import MonteCarloTreeSearch
import numpy as np
import matplotlib.pyplot as plt

from resnet import ResNet

torch.manual_seed(0)

# game = TicTacToe()
game = ConnectFour()

player = 1

args = {
    "C": 2,
    "num_searches": 100,
    "dirichlet_epsilon": 0.0,
    "dirichlet_alpha": 0.3,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device=device)
# model.load_state_dict(torch.load("weights/model_0_ConnectFour.pt", map_location=device))
model.eval()

mcts = MonteCarloTreeSearch(game, args, model)

state = game.get_initial_state()

while True:
    print("Player: ", player)
    print(state)

    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("Valid moves: ", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input("Enter action: "))

        if valid_moves[action] == 0:
            print("Invalid move!")
            continue
    else:
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)

    state = game.get_next_state(state, action, player)
    value, is_terminal = game.get_value_and_terminated(state, action)

    if is_terminal:
        print("Player: ", player)
        print(state)
        if value == 1:
            print("Player ", player, " wins!")
        else:
            print("Draw!")

        break

    player = game.get_opponent(player)
