import torch
import nn
from torch import nn
import data
import matplotlib.pyplot as plt
import math


def makePrediction(gain, bandwidth, power):
    model = torch.load("./Checkpoints/5/model.pth", map_location=torch.device("cpu"))

    model.eval()

    with torch.no_grad():
        input_data = (gain, bandwidth, power)
        input_tensor = torch.tensor(input_data, dtype=torch.float)
        output = model(input_tensor)
        print(
            f"Input: {input_data}, Input Width: {output[0].item() / 1000000}, Load Resistance: {output[1].item() * 10000}, Tail Width: {output[2].item() / 1000000}"
        )  # was output.item()

def score(targetGain, targetBW, targetPower, actualGain, actualBW, actualPower):
    print(f"Target Gain: {targetGain}, Target Bandwidth: {targetBW}, Target Power: {targetPower}")
    print(f"Actual Gain: {actualGain}, Actual Bandwidth: {actualBW}, Actual Power: {actualPower}")
    gainError = ((actualGain - targetGain) / targetGain) * 100
    bwError = ( (math.log(actualBW/10**9, 10) - math.log(targetBW/10**9, 10)) / (math.log(targetBW/10**i9, 10)) ) * 100
    powerError = ((targetPower - actualPower) / targetPower) * 100
    print(f"Gain Error: {gainError}, Bandwidth Error: {bwError}, Power Error: {powerError}")

    gainScore = 101 - (1 / (2.7182818 ** (0.091*gainError))) + (gainError/10)
    bwScore = ((-1 * (bwError**2)) / 100) + 100
    powerScore = 101 - (1 / (2.7182818 ** (0.153*powerError))) + (powerError/5)
    
    print(f"Gain Score: {gainScore}, Bandwidth Score: {bwScore}, Power Score: {powerScore}")

def benchmark(width):
    CData = data.CircuitData()
    model = torch.load("model.pth")
    model.eval()
    predictions = []
    input_list = []
    output_list = []
    for point in CData.data:
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
    # plt.show()
    plt.savefig("Benchmark.png")

    sqError = 0
    for i in range(len(predictions)):
        sqError += (output_list[i] - predictions[i]) ** 2

    print(f"Mean Squared Error: {sqError/len(predictions)}")


if __name__ == "__main__":
    # benchmark(2e-6)
    while True:
        gain = float(input("Enter Target Gain (V/A): ")) / 10000
        bandwidth = float(input("Enter Target Bandwidth (Hz): ")) / 10000000000
        power = float(input("Enter Target Tail Current (A): ")) * 100
        makePrediction(gain, bandwidth, power)
        actualGain = float(input("Enter Actual Gain (V/A): "))
        actualBW = float(input("Enter Actual Bandwidth (Hz): "))
        actualPower = float(input("Enter Actual Tail Current (A): "))
        score(gain * 10000, bandwidth*10000000000, power/100, actualGain, actualBW, actualPower)
