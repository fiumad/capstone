import torch
import data
import matplotlib.pyplot as plt
import numpy as np


def plotHist(modelPath):
    all_weights = []
    state_dict = torch.load(modelPath).state_dict()
    for param_tensor in state_dict:
        if "weight" in param_tensor:
            weights = state_dict[param_tensor].cpu().numpy().flatten()
            all_weights.extend(weights)

    plt.figure(figsize=(10, 6))
    plt.hist(all_weights, bins=100, color="blue")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Weights")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    modelPath = "./Checkpoints/4/model.pth"
    plotHist(modelPath)
