import torch
import nn
from torch import nn
import data
import matplotlib.pyplot as plt


def makePrediction(frequency, width):
    model = torch.load("model.pth")

    model.eval()

    with torch.no_grad():
        input_data = (frequency, width)
        input_tensor = torch.tensor(input_data, dtype=torch.float)
        output = model(input_tensor)
        print(f"Input: {input_data}, Output: {output.item()}")


def benchmark(width):
    TIData = data.TransimpedanceData()
    model = torch.load("model.pth")
    model.eval()
    predictions = []
    input_list = []
    output_list = []
    for point in TIData.data:
        if point[1] == width:
            with torch.no_grad():
                input_data = (
                    point[0],
                    point[1],
                )

                input_list.append(point[0])
                output_list.append(point[2])

                input_tensor = torch.tensor(input_data, dtype=torch.float)
                output = model(input_tensor)
                predictions.append(output.item())

    plt.figure(figsize=(10, 6))
    plt.plot(
        input_list,
        output_list,
        label="Expected",
        color="red",
    )
    plt.plot(
        input_list,
        predictions,
        label="Predicted",
        color="blue",
    )
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Transimpedance (V/A)")
    plt.title("Actual vs Predicted Transimpedance")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.show()

    sqError = 0
    for i in range(len(predictions)):
        sqError += (output_list[i] - predictions[i]) ** 2

    print(f"Mean Squared Error: {sqError/len(predictions)}")


if __name__ == "__main__":
    benchmark(2e-6)
