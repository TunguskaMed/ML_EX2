from torch.utils.data import Dataset
#import stud.transformer_utils as transformer_utils
from torchvision.io import read_image
from termcolor import colored
from typing import List
import time
import os
from torchvision import transforms

class CarActionDataset(Dataset):
    """Car action dataset class
    """
    def __init__(self, samples: List[tuple]):
        """Constructor for car action dataset

        Args:
            samples (List[tuple]): List of tuples (image_path, action_label)
            
        """
        
        self.samples = samples 
        
        #print(self.samples)
        
        
    

    def __len__(self):
        """Return samples length

        Returns:
            int: length of samples list (number of samples)
        """
        return len(self.samples)

    def __getitem__(self, index: int):
        
        """Get item for dataloader input

        Args:
            index (int): index-th sample to access

        Returns:
            tuple: (image, labels) open image_path and return the tuple (image,label) related to the index-th element 
        """
        
        #convert index-th sample senses in indices
        #print(colored(self.samples[0],"red"))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
                         
                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image = transform(read_image(self.samples[index][0])).detach()
        
        
        #print(colored(image.shape,"yellow"))
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        
        
        return image, self.samples[index][1]
    
        