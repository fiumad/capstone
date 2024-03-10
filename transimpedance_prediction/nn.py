import torch
from torch import nn
from torch.utils.data import DataLoader
import data as SimulationData

# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_device(device)
# print(f"Using {device} device")


class TransimpedanceModel(nn.Module):
    def __init__(self):
        super(TransimpedanceModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    dataset = SimulationData.TransimpedanceData()
    # Spot check the data
    print("Data Spot Check:")
    dataset.print_data(50)
    dataloader = DataLoader(dataset, batch_size=16)  # , shuffle=True)
    num_epochs = 1000

    model = TransimpedanceModel()
    loss_fn = nn.MSELoss()
    learning_rate = 0.001556
    new_low = 4000
    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for batch in dataloader:
            inputs, targets = batch["inputs"], batch["targets"]
            predictions = model(inputs)
            loss = loss_fn(predictions, targets.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate += 0.5e-5
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        if loss.item() < 4000:
            learning_rate = learning_rate / 2
        if loss.item() < 1000:
            learning_rate = learning_rate / 2
    # torch.save(model.state_dict(), "model.pth")

    input_data = (5e8, 1.84e-6)
    input_tensor = torch.tensor(input_data, dtype=torch.float)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Input: {input_data}, Output: {output.item()}")
