import torch

# Load the model weights from .pth file
model_weights = torch.load('weights/resnet50_dh128_lvis.pth')

# # Get the model architecture
# model = model_weights['model']

# # Calculate the total number of parameters in the model
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total number of parameters in the model: {total_params}')

total_params = sum(p.numel() for p in model_weights.values())
print(f'Total number of parameters in the model: {total_params/1000000} M')