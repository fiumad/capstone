import torch
import data
import matplotlib.pyplot as plt
import numpy as np


def plotHist(modelPath):
    all_weights = []
    state_dict = torch.load(modelPath, map_location=torch.device("cpu")).state_dict()
    for param_tensor in state_dict:
        if "weight" in param_tensor:
            weights = state_dict[param_tensor].cpu().numpy().flatten()
            all_weights.extend(weights)

    plt.figure(figsize=(16, 12))
    plt.hist(all_weights, bins=100, color="blue")
    plt.xlabel("Weight Value", fontsize=34)
    plt.ylabel("Frequency", fontsize=36)
    plt.title("Histogram of Weights", fontsize=36)
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    plt.grid(True)
    plt.savefig("WeightHistogram.png")


if __name__ == "__main__":
    modelPath = "./Checkpoints/4/model.pth"
    plotHist(modelPath)
