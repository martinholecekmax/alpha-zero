# AlphaZero from scratch

This is a project to implement AlphaZero from scratch. The goal is to have a simple, readable implementation that can be used to learn about AlphaZero and Reinforcement Learning in general.

## Folder structure

### Simple implementation

The `simple` folder contains a simple implementation of AlphaZero without using a neural network only monte carlo tree search.

### Neural network implementation

The `nn` folder contains an implementation of AlphaZero using a neural network and monte carlo tree search. The neural network in our case ResNet model is implemented using PyTorch.

### Stored Weights of the model

The `weights` folder contains the weights and optimizer of the model after training. The weights are stored in a `.pt` file.

## Games

You can choose from 2 games to play: TicTacToe and ConnectFour. The `games` folder contains the code for the games.

## Training

The `alphazero_train.py` file contains the code to train the model.

You can train the model sequentially or in parallel by choosing either `AlphaZero` or `AlphaZeroParallel` as the model in the `alphazero_train.py` file. You can also choose the game by changing the `game` variable in the `alphazero_train.py` file to either `TicTacToe` or `ConnectFour`.

## Playing against the model

The `alphazero_play.py` file contains the code to play against the model. You can load the weights of the model and play against it.

You can switch between games by changing the `game` variable in the `alphazero_play.py` file to either `TicTacToe` or `ConnectFour`.
