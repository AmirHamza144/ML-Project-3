Overview
This project extends the Group 17 Machine Learning pipeline to build and evaluate a Convolutional Neural Network (CNN) using PyTorch, designed to classify images as containing the official University of Arkansas Razorback logo or not.

The model supports automated detection of potential unlicensed logo usage on Etsy, helping the university identify vendors for further investigation.
The notebook automates preprocessing, model building, training, tuning, and evaluation using a reproducible pipeline built on torchvision and PyTorch.
Environment Setup
Install required dependencies:
pip install torch torchvision numpy matplotlib pillow
GPU support (CUDA) is optional but recommended.
Project Outline
1. Dataset Development
Collected a custom image dataset from Etsy and the official University of Arkansas branding page.
Each image labeled as:
razorback → contains the official logo
non_razorback → does not contain the official logo
Minimum dataset size: 50+ images (as required).
Dataset split:
Split	Percentage
Train	70%
Valid	20%
Test	10%
Final preprocessing transformation:
transforms.Resize((500, 500))
transforms.ToTensor()
Implemented a safe image loader to automatically skip corrupted .png files and prevent crashing.
2. CNN Model Training & Tuning
Loaded dataset using torchvision.datasets.ImageFolder.
Designed a custom CNN architecture with:
Convolutional layers
ReLU activation
MaxPooling
Fully connected layers
Dropout to reduce overfitting
Trained using:
Loss: CrossEntropyLoss
Optimizer: Adam
Batch size: tuned experimentally
Epochs: adjusted based on validation loss
Experimented with several architectures by modifying:
• Number of filters
• Kernel sizes
• Depth of convolution stack
• Dropout rate
Selected the best model based on validation accuracy and learning curves.
Saved final model:
torch.save(model, "Group_17_CNN_FullModel.ph")
3. Evaluation Metrics
Evaluated the model using:
Training vs Validation accuracy
Loss curves over epochs
Per-class performance
Visual inspection of predictions
Misclassification analysis
The CNN successfully learned to classify Razorback vs non-Razorback images with strong performance despite the small dataset size.
4. Final Output and Deliverables
This project produces:
File	Description
Group_17_CNN_FullModel.ph	Final saved PyTorch CNN model
Jupyter Notebook	Full training + evaluation workflow
Presentation Slides	Summary of dataset creation, model design, and results
GitHub Repository	Clean, well-commented code and documentation
Output File Description
File Name	Type	Purpose
Group_17_CNN_FullModel.ph	Serialized PyTorch Model	Used to classify new images as logo or non-logo
To reload the model:
import torch
model = torch.load("Group_17_CNN_FullModel.ph", map_location="cpu")
model.eval()
Reproduction Steps
Clone/download the repository.
Prepare dataset using this structure:
data/
   train/
      razorback/
      non_razorback/
   valid/
      razorback/
      non_razorback/
   test/
      razorback/
      non_razorback/
Open the notebook:
project3_group17_cnn.ipynb
Run all cells sequentially.
The notebook will automatically:
Load and preprocess images
Train multiple CNN architectures
Select the best model
Save Group_17_CNN_FullModel.ph
Reflection
This project demonstrates:
The importance of a clean, high-quality dataset for vision tasks
The power of convolutional layers in extracting spatial logo features
Advantages of dropout, Adam optimizer, and validation curves for stable training
Benefits of safe loaders to handle corrupted input files
Practical use of PyTorch for scalable and reproducible CNN experiments
Future directions:
Add data augmentation to improve generalization
Expand dataset size significantly
Apply pretrained models (e.g., ResNet-18) for transfer learning
Develop a production-ready inference script or web API
Group Members
Amir Hamza Akash
Brynn van Guilder
