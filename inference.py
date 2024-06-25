
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional
from SwinUNet import SwinUNet
from dataset import test_dataloader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNet(H=224,W=224,seq_len=30,C=32,out_seq=1,num_blocks=3,patch_size=4)
loss_fn = nn.MSELoss()
last_saved_model_path = 'Model_weights\model_epoch_100.pth'
model.load_state_dict(torch.load(last_saved_model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def get_sample_by_index(dataloader, index):
    for i, sample in enumerate(dataloader):
        if i == index:
            return sample
    raise IndexError(f"Index {index} out of range")


index = 0

sample = get_sample_by_index(test_dataloader, index)
x_sample, y_sample,input_file_path, target_file_path = sample


x_sample = x_sample.to(DEVICE)
y_sample = y_sample.to(DEVICE)
print(x_sample.shape)
# Forward pass through the model
with torch.no_grad():
    y_pred = model(x_sample)


loss = loss_fn(y_pred, y_sample)


print(f"Index of sample: {index}")
print(f"Loss: {loss.item()}")

for i in range(len(x_sample)):
        # Extract predicted and ground truth for the current sample
        y_pred_sample = y_pred[i, 0].cpu().numpy()  # Convert to numpy array and move to CPU
        y_sample_vis = y_sample[i, 0].cpu().numpy()  # Convert to numpy array and move to CPU

        # Plot the predicted and ground truth images for the current sample
        plt.figure(figsize=(8, 5))
        plt.title(f'For {target_file_path[i]} \n')
        plt.subplot(1, 2, 1)
        plt.title(f'Predicted\n')  # Include file path as title
        plt.imshow(y_pred_sample)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f'Ground Truth\n')  # Include file path as title
        plt.imshow(y_sample_vis)
        plt.axis('off')

        plt.show()