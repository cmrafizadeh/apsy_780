import torch

class SmallMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # self.fc = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.fc1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, input):
        hidden_rep = self.fc1(input)
        hidden_rep_relu = self.relu(hidden_rep)
        output = self.fc2(hidden_rep_relu)
        return output
        # return self.fc2(self.relu(self.fc1(input)))

model = SmallMLP(in_dim=10, hidden_dim=50, out_dim=3)
input = torch.randn(1, 10)
logits = model(input)

print("Input: ", input)
print("Logits: ", logits)

output_softmaxed = torch.softmax(logits, dim=1)
print("Probabilities: ", output_softmaxed)

print("Parameters of Network: ", model.parameters())