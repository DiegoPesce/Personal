from torch import nn, optim

class GrandeNN(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,5),
            nn.Sigmoid(),
            nn.Linear(5,3),
            nn.Sigmoid(),
            nn.Linear(3,output_size),
            nn.Tanh()
        )
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.L1Loss()

    def forward(self, x):
        x = self.model(x)
        return x
    
    def train_step(self, input, labels):
        self.optimizer.zero_grad()
        output = self.forward(input)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

