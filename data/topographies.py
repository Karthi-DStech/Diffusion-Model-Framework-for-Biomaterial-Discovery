import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image

from data.datasets import BaseDataset
from utils import images_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from torchvision import transforms

class BiologicalObservation(BaseDataset):
    """
    Biological Observation dataset class
    
    Parameters
    ----------
    opt : argparse.Namespace
        The options used to initialize the dataset
    
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the BiologicalObservation class
        """
        super().__init__(opt)
        self._name = "BiologicalObservation"
        
        # Define a default transform
        self._transform = images_utils.get_transform(img_size=self._opt.img_size, mean= self._opt.mean, std= self._opt.std, grayscale=True)
        
        self._print_dataset_info()

    def _create_dataset(self) -> None:
        """
        This function creates the dataset. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Creates the dataset by loading the images from the specified folder.
        """
        
        self.images_path = self._opt.images_folder
        self._img_type = self._opt.img_type
        
        # Find all images with the specified extension
        image_files = glob.glob(os.path.join(self.images_path, f"*.{self._img_type}"))
        
        if not image_files:
            raise ValueError(f"No images found in {self.images_path} with type {self._img_type}")
        
        self._images_names = [os.path.basename(image) for image in image_files]

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        This function returns the image at the specified index.
        
        Parameters
        ----------
        index : int
            The index of the image to return
            
        Returns
        -------
        image : torch.Tensor
            The image at the specified index
        """
        
        if index >= len(self._images_names):
            raise IndexError(f"Index {index} is out of range for images list of length {len(self._images_names)}")
        
        image_name = self._images_names[index]
        img_path = os.path.join(self.images_path, image_name)
        
        image = Image.open(img_path)
        image = self._transform(image)
        return image

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        """
        return len(self._images_names)
