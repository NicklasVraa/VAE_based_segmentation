# Importing libraries. --------------------------------------------------
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import os
from os.path import join


# Global constants. ------------------------------------------------------
cmap_seg = ListedColormap(['none', 'red']) # For drawing tumors in red.


# Checkpointing. ---------------------------------------------------------
def list_checkpoints(dir):
    epochs = []
    for name in os.listdir(dir):
        if os.path.splitext(name)[-1] == '.pth':
            epochs += [int(name.strip('ckpt_.pth'))]
    return epochs

def save_checkpoint(dir, epoch, model, optimizer=None):
    checkpoint = {}; checkpoint['epoch'] = epoch

    if isinstance(model, torch.nn.DataParallel):
        checkpoint['model'] = model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    else:
        checkpoint['optimizer'] = None

    torch.save(checkpoint, os.path.join(dir, 'ckpt_%02d.pth'% epoch))

def load_checkpoint(dir, epoch=0):
    if epoch == 0: epoch = max(list_checkpoints(dir))
    checkpoint_path = os.path.join(dir, 'ckpt_%02d.pth'% epoch)
    return torch.load(checkpoint_path, map_location='cpu')

def load_model(dir, model, epoch=0):
    ckpt = load_checkpoint(dir, epoch)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt['model'])
    return model

def load_optimizer(dir, optimizer, epoch=0):
    ckpt = load_checkpoint(dir, epoch)
    optimizer.load_state_dict(ckpt['optimizer'])
    return optimizer


# Guaging performance. ---------------------------------------------------
def IoU(label, recon, thresh):
    inter = ((label >= thresh) & (recon >= thresh)) * 1.0
    union = ((label >= thresh) | (recon >= thresh)) * 1.0
    return inter.sum() / union.sum() / label.shape[0]


# Visualizing performance. -----------------------------------------------
def superimpose(image, label):
    fig, axs = plt.subplots(1, 2, figsize=(8,5))
    axs[0].imshow(torch.squeeze(image), cmap='gray')
    axs[1].imshow(torch.squeeze(image), cmap='gray')
    axs[1].imshow(torch.squeeze(label), cmap=cmap_seg)
    fig.canvas.draw()

def draw(x, x_hat):
    fig, axs = plt.subplots(1, 2, figsize=(8,5))
    img_0 = x[0].detach().numpy()
    img_1 = x_hat[0].detach().numpy()
    axs[0].imshow(img_0, vmin=0, vmax=1, cmap='gray')
    axs[1].imshow(img_1, vmin=0, vmax=1, cmap='gray')
    fig.canvas.draw()


# Custom dataset class. --------------------------------------------------
class CT_Dataset(Dataset):
    def __init__(self, path, organ, resolution):
        self.images = torch.load(join(path, organ + '_image_slices_' + str(resolution) + '.pt'))

        self.labels = torch.load(join(path, organ + '_label_slices_' + str(resolution) + '.pt'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Adding channel dimension.
        return image[None, :], label[None, :]

    def show_datapoint(self, index):
        image, label = self.__getitem__(index)
        superimpose(image, label)
