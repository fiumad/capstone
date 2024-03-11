################################################################'''
'''
Author: Dan Fiumara
Usage Instructions:
    1. Ensure that Pytorch is installed via the following tutorial:
        - https://pytorch.org/get-started/locally/
    2. Create a Python Virtual Environment via this command:
        -   python3 -m venv [your-env-name]
    3. Activate the Virtual Environment via this command:
        -   source [your-env-name]/bin/activate
    4. Install the required packages via the following commands:
        - pip install matplotlib
        - pip install pandas
    5. At the commandline, run the following command:
        - python3 nn.py

Expected Output:
    The proram will output a series of loss values as the neural network 
    learns to proedict the output of a transimpedance amplifier. These loss
    values should consistently decrease over time. This indicates that the 
    network is successfully learning. The loss is in the form of a mean squared
    error, which is a measure of the difference between the predicted and actual 
    values in the current batch of data. This means that the square root of the loss
    after each epoch is the average difference between the predicted and actual values.
    The loss over time will be plotted in a graph after the training loop ends.
    The first few hudred epochs will be ignored in the graph to better visualize our loss trends.
    
    After training is complete, the resulting model will be used to make a prediction on a new input. 
    The expected output is 15000. The input is (5e8, 1.84e-6).
'''
################################################################
import torch
from torch import nn
from torch.utils.data import DataLoader
import data as SimulationData
import matplotlib.pyplot as plt

# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_device(device)
# print(f"Using {device} device")


def plot_losses(loss_list):
    plt.figure(figsize=(12, 6))
    plt.plot(loss_list, label="Training Loss", marker=".")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


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
    num_epochs = 2000
    loss_list = []

    model = TransimpedanceModel()
    loss_fn = nn.MSELoss()
    learning_rate = 0.001556

    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for batch in dataloader:
            inputs, targets = batch["inputs"], batch["targets"]
            predictions = model(inputs)
            loss = loss_fn(predictions, targets.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss.item() > 1000:
                learning_rate += 0.5e-5
        loss_list.append(loss.item())

        if loss.item() < 4000:
            learning_rate = learning_rate / 2
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    # torch.save(model.state_dict(), "model.pth")

    input_data = (5e8, 1.84e-6)
    input_tensor = torch.tensor(input_data, dtype=torch.float)

    # Plot the loss
    plot_losses(loss_list[220:])

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Input: {input_data}, Output: {output.item()}, Expected: 15000")
