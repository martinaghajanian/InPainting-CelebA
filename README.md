# InPainting-CelebA

This project focuses on inpainting smiling faces from the CelebA dataset using Generative Adversarial Networks (GANs). The model is trained to reconstruct masked regions, specifically targeting the smile area, and generate realistic outputs.

## Clone the Repository
```bash
git clone https://github.com/yourusername/InPainting-CelebA.git
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Download the Dataset
Go to [CelebA Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and download the entire dataset in root directory of the project.

## Instructions
After downloading the dataset, follow the steps in **sections 1 and 2** of `InPainting.ipynb` to prepare the dataset and set up the training pipeline.  
To see the results of the already trained model, run the code cells in `Demo.ipynb` after data preprocessing.

## Training
To train the model on your own:
1. Go through the entire `InPainting.ipynb` notebook.
2. Follow the steps for training and validation to generate your own inpainting results.
