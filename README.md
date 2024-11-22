# Denoising-Diffusion-Probablistic-Models-for-Biomaterial-Discovery

**Diffusion Model Training for Image Generation using Biomaterial Topography Dataset**

This project contains the code to train a diffusion model for Biomaterial Discovery. It includes a PyTorch implementation of the U-Net model, several building blocks used in the model architecture, and scripts for training and logging.

**The Generated images can be found in the `Generated Image/` folder. Currently, this repo holds the DDPM-generated images of 32X32 and 64X64 pixel resolution topographies.**


### Generated One-by-One Biomaterial Designs after 40_000 Epochs

![image](https://github.com/Karthi-DStech/Denoising-Diffusion-Probablistic-Models-for-Biomaterial-Discovery/assets/126179797/7515456e-09cf-48ac-9c30-882b2dfc95a8)




### Project Structure

#### 1. `data/`
Contains scripts related to dataset handling and preprocessing:

- **`datasets.py`**: Defines dataset loaders and acts as a base class for the other datasets.
- **`mnist.py`**: Contains utilities specific to the MNIST dataset.
- **`topographies.py`**: Handles biomaterial topographic datasets.

### 2. `model/`
Implements core architectures and modules for generative models:
- **`ddpm.py`**: Defines the implementation of the Denoising Diffusion Probabilistic Model (DDPM).
- **`unet.py`**: Implements the U-Net architecture used in diffusion models.
- **`attention_block.py`**: Implements attention mechanisms for enhancing feature extraction.
- **`resnet_block.py`**: Contains ResNet blocks for residual learning.
- **`nin_block.py`**: Implements Network-in-Network blocks for feature extraction.
- **`downsampling_block.py`**: Defines the downsampling layers.
- **`upsampling_block.py`**: Defines the upsampling layers.
- **`timestep_embedding.py`**: Encodes temporal information into model embeddings.
- **`networks.py`**: Base class for all the networks created. 
- **`models.py`**: Base class for the models and acts as an entry point for accessing various model instances. 

### 3. `option/`
Handles command-line arguments and configurable options for training and experiments:
- **`base_options.py`**: Defines shared options for general configurations.
- **`train_options.py`**: Extends `base_options.py` with training-specific parameters.

### 4. `utils/`
Provides helper functions and utilities for the project.

### 5. Other Files
- **`train.py`**: Main script for training the models.
- **`call_methods.py`**: Contains method calls dynamically for initiating specific processes or experiments.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.

### Requirements

To run the code, you need the following:

- Python 3.8 or above
- PyTorch 1.7 or above
- torchvision
- tqdm
- matplotlib
- Tensorboard 2.7.0

Install the necessary packages using pip:


### Dataset
