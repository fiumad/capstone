import torch
import nn
from torch import nn

def makePrediction(frequency, width):
    model = torch.load("model.pth")
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

model = TransimpedanceModel(nn.Module)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
