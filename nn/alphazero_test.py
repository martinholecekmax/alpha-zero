from games.tictactoe import TicTacToe
import torch
import matplotlib.pyplot as plt

from resnet import ResNet

torch.manual_seed(0)

tictactoe = TicTacToe()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = tictactoe.get_initial_state()
state = tictactoe.get_next_state(state, 2, -1)
state = tictactoe.get_next_state(state, 4, -1)
state = tictactoe.get_next_state(state, 6, 1)
state = tictactoe.get_next_state(state, 8, 1)

encoded_state = tictactoe.get_encoded_state(state)

tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

model = ResNet(tictactoe, 4, 64, device=device)

model.load_state_dict(torch.load("weights/model_2.pt", map_location=device))

model.eval()

policy, value = model(tensor_state)

value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print("\nValue: \n", value)
print("\nState: \n", state)
print("\nTensor state: \n", tensor_state)

plt.bar(range(tictactoe.action_size), policy)
plt.show()
