import torch
import nn
from torch import nn
import TIModel as t

def makePrediction(frequency, width):
    model = t.TransimpedanceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.035)

    checkpoint = torch.load(model.pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()

    with torch.no_grad():
        input_data = (frequency, width)
        input_tensor = torch.tensor(input_data, dtype=torch.float)
        output = model(input_tensor)
        print(f"Input: {input_data}, Output: {output.item()}")

if __name__ == "__main__":
    makePrediction(5e8, 1.84e-6)
    makePrediction(100, 1.84e-6)
    makePrediction(100, 2.2e-6)
    makePrediction(1000, 2.2e-6)
