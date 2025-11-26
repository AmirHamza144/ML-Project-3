# **Group 17: Project 3 – Razorback Logo Classification (CNN)**

**Updated:** 2025-11-26  

## **Overview**

This project extends the Group 17 Machine Learning pipeline to implement a **Convolutional Neural Network (CNN)** model on a **custom Razorback image dataset**, predicting whether an image contains the **official University of Arkansas Razorback logo** or not.

The notebook automates dataset preprocessing, model tuning, and evaluation using `torch` and `torchvision` utilities such as **ImageFolder**, **transforms**, and **custom CNN architectures**, ensuring consistent transformations and reproducible results across the training, validation, and testing phases.

---

## **Environment Setup**

Before running, ensure we have a Python environment with the following dependencies installed:

```bash
pip install torch torchvision numpy matplotlib pillow
```

---

## **Project Outline**

### **1. Dataset Development and Preprocessing**

- Collected a custom dataset of Razorback and non-Razorback images from Etsy and the official Razorback branding website.  
- Organized data into the required directory structure:

```
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
```

- Used a safe image loader to bypass any corrupted `.png` files that cause `UnidentifiedImageError`.  
- Applied preprocessing transformations:
  - `transforms.Resize((500, 500))`
  - `transforms.ToTensor()`

---

### **2. Model Training and Tuning**

- Loaded the dataset using `ImageFolder` with consistent preprocessing.  
- Designed a custom CNN architecture containing:
  - Convolutional layers  
  - ReLU activations  
  - MaxPooling layers  
  - Fully connected layers  
  - Dropout for regularization  

- Trained the model using:
  - **Loss function:** `CrossEntropyLoss`
  - **Optimizer:** Adam

- Tuned model behavior by experimenting with:
  - Number of convolution filters  
  - Kernel sizes  
  - Hidden layer dimensions  
  - Dropout rates  

- Compared multiple candidate architectures and selected the **best performing model** using validation accuracy and stability of loss curves.

---

### **3. Evaluation Metrics**

- Computed:
  - Training accuracy  
  - Validation accuracy  
  - Loss curves across epochs  

- Compared predicted vs. actual labels on validation images.  
- Performed error analysis based on misclassified samples.

---

### **4. Final Model Saving**

The final selected model was saved using the naming format required by the assignment:

```
Group_17_CNN_FullModel.ph
```

To reload the model:

```python
import torch
model = torch.load("Group_17_CNN_FullModel.ph", map_location="cpu")
model.eval()
```

---

## **Output File Description**

| File Name | Description |
|-----------|-------------|
| **Group_17_CNN_FullModel.ph** | Serialized PyTorch model for Razorback logo classification |

---

## **Reproduction Steps**

1. Place the dataset into the directory structure shown above.  
2. Open the notebook:

```
Group17_Razorback_CNN_Project.ipynb
```

3. Run all cells sequentially.  
4. The notebook will automatically:
   - Load and preprocess the dataset  
   - Train multiple CNN variants  
   - Evaluate accuracy and loss metrics  
   - Save the final model as:  
     ```
     Group_17_CNN_FullModel.ph
     ```

---

## **Reflection**

This project highlights:
- The importance of consistent image preprocessing.  
- How CNN layers capture spatial features critical for Razorback logo identification.  
- The effects of convolution depth, filter size, and dropout on model generalization.  
- The usefulness of validation curves in preventing overfitting.  
- The value of a safe loader for handling corrupted images during ingestion.

Future improvements include:
- Expanding the dataset size to improve model robustness.  
- Adding data augmentation (rotations, color jitter, flipping).  
- Using transfer learning approaches such as ResNet or MobileNet.  
- Building a more streamlined inference pipeline for vendor-screening tasks.

---
