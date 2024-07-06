import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import re

# trans = transforms.Compose([
#         # transforms.ToPILImage(),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
#         transforms.Lambda(lambda x: x.float()),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])

# def get_png_files_in_directory(directory_path):
#     path = []
#     for f in os.listdir(directory_path):
#         if f.lower().endswith('.png'):
#
#     return [f for f in os.listdir(directory_path) if f.lower().endswith('.png')]

def extract_number(filename):
    return int(re.search(r"\d+", filename).group())

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

class DrozyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.jpg')]  ##记得改后缀
        self.image_files.sort(key= lambda x: int(re.search(r"\d+", x).group()))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])

        return image_path

# Set your data directory (where the images are stored)


def argument_in():
    parser = argparse.ArgumentParser(description='dataloader Script')
    parser.add_argument("--data_dir",type=str, default='./testpart/DROZY/video_to_jpg', help='Specify the datasets value')
    args = parser.parse_args()
    data_dir_value = args.data_dir
    return data_dir_value

def get_loader():
    data_dir = argument_in()
    # Create an instance of your custom dataset
    image_dataset = DrozyDataset(data_dir, transform=None)
    # Create a DataLoader for sequential loading
    batch_size = 1
    image_path = DataLoader(dataset=image_dataset, batch_size=batch_size, shuffle=False)
    # print(f'{image_path}')
    return image_path
