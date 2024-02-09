import torch
x = torch.rand(5,3)
print(x)
print("Cuda is available?",torch.cuda.is_available())
