import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon = 1e-6):
        super(SoftDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='mean')
        batch_size, num_classes, *spatial_dims = predictions.shape
        predictions_flat = predictions.view(batch_size, num_classes, -1)
        #targets_flat = F.one_hot(targets.view(batch_size, -1), num_classes=num_classes).permute(0, 2, 1).float()

        targets_flat = targets.view(batch_size, num_classes, -1)
        intersection = torch.sum(predictions_flat * targets_flat, dim = 2)
        union = torch.sum(predictions_flat ** 2, dim = 2) + torch.sum(targets_flat ** 2, dim = 2)
        dice_loss = 1 - (2 * intersection / (union + self.epsilon)).mean()

        return dice_loss + ce_loss





"""
batch_size = 2
num_classes = 1
spatial_dims = (96, 96, 96)
predictions = torch.randn(batch_size, num_classes, *spatial_dims)
targets = torch.randint(0, num_classes, (batch_size, *spatial_dims))

loss_fn = SoftDiceLoss()
loss = loss_fn(predictions, targets)
print(f"Dice loss: {loss.item()}")
"""


