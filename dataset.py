import os
import numpy as np
import torch 
from torchvision import transforms
from torchvision.io import read_image



class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, of_root, split_txt_file):
        self.root_dir = root_dir
        self.of_root = of_root
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
        ])
        self.files = []
        self.labels = []
        with open(split_txt_file, 'r') as f:
            for line in f:
                line = line.split()
                self.files.append(line[0])
                verb = int(line[2])
                verb -= 1
                noun = int(line[3])
                action = int(line[1])
                action -= 1
                if noun > 16 and noun < 44:
                    noun -= 1
                elif noun > 44:
                    noun -= 2
                noun -= 1
                # label from [0, n)
                self.labels.append((noun, verb, action))

    def __len__(self):
        assert len(self.files) == len(self.labels), "Error: Length of files and labels do not match in dataset"
        return len(self.labels)

    def __getitem__(self, index):
        folder = self.files[index]
        label = self.labels[index]
        folder_path = os.path.join(self.root_dir, folder)
        l = os.listdir(folder_path)
        assert len(l) == 16, "Error: Number of frames in a folder is not 16"
        images = [self.transform(read_image(os.path.join(folder_path, img))) for img in l] # image[0].shape = (3, 480, 640)
        frames = torch.stack(images) # frames.shape = (T, 3, 480, 640)

        of_file = os.path.join(self.of_root, folder+".npz")
        of = np.load(of_file)['data']
        of = torch.from_numpy(of)

        return frames, of, label, folder