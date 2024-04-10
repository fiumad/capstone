################################################################
"""
Author: Dan Fiumara
Usage Instructions:
    1. Ensure that your development ievironment is set up according to README.md
    2. At the commandline, run the following command:
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
"""
################################################################
import torch
from torch import nn
from torch.utils.data import DataLoader
import data as SimulationData
import TIModel as t
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Using {device} device")


def plot_losses(loss_list):
    plt.figure(figsize=(12, 6))
    plt.plot(loss_list, label="Training Loss", marker=".")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    # plt.show()
    plt.savefig("loss.png")


if __name__ == "__main__":
    dataset = SimulationData.TransimpedanceData()
    # Spot check the data
    print("Data Spot Check:")
    dataset.print_data(50)
    dataloader = DataLoader(dataset, batch_size=32)  # , shuffle=True)
    num_epochs = 4000
    loss_list = []

    model = t.TransimpedanceModel()
    loss_fn = nn.MSELoss()
    learning_rate = 1.38e-7
    first = True

    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for batch in dataloader:
            inputs, targets = batch["inputs"], batch["targets"]
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, LR: {learning_rate}")
        if loss.item() < 0.058 and first:
            learning_rate = 1.38e-8
            first = False
        if not first and loss.item() > loss_list[-2]:
            learning_rate /= 10

        if (
            not first
            and loss.item() < 0.03
            and loss_list[-1] == loss_list[-2] == loss_list[-3]
        ):
            break

    # Plot the loss
    plot_losses(loss_list[10:])

    torch.save(model, "model.pth")
