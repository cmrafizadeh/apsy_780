import torch

class MyOneLayerdNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # super(MyOneLayerdNet, self).__init__()
        self.fc = torch.nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, input):
        output = self.fc(input)
        return output

model = MyOneLayerdNet(in_dim=10, out_dim=3)
input = torch.randn(1, 10)

output = model(input)

print("Input: ", input)
print("Output: ", output)

print("Weights: ", model.fc.weight)
print("Biases: ", model.fc.bias)