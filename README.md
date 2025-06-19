# 📦 Inventory Monitoring at Distribution Centers – End-to-End AWS SageMaker Project

This project presents a full deep learning pipeline to automate object counting in bins using computer vision. Built on **Amazon SageMaker**, it demonstrates best practices in training, hyperparameter tuning, model deployment, and inference—all using a CNN model (EfficientNetB3) fine-tuned on the **Amazon Bin Image Dataset**.

---

## 🚀 Project Overview

The goal is to predict the **number of objects** in a bin from an image, solving it as a **multi-class classification** problem. This can improve inventory tracking efficiency in automated warehouses.

---

## 📁 Dataset

**Amazon Bin Image Dataset**
- 500,000+ RGB bin images with metadata
- Subset of ~10,000 used for training
- Input: Image of a bin
- Label: Integer count of items (1 to 5)
- Organized into folders named by object count

🔗 [Dataset Source](https://registry.opendata.aws/amazon-bin-imagery/)

---

## 🧠 Model Architecture

- ✅ **EfficientNetB3** pre-trained on ImageNet
- ✅ Final layers replaced for 5-class classification
- ✅ Dropout + ReLU + Linear layers
- ✅ Trained using PyTorch and fine-tuned on GPU (`ml.g4dn.xlarge`)
- ✅ Supports training via `train.py`, tuning via `tuner.py`, and inference via `inference.py`

---

## 🔄 Pipeline Components

### 🔧 1. Training (`train.py`)
- Loads dataset using `ImageFolder`
- Splits into Train/Val/Test
- Applies data augmentation + normalization
- Trains model and saves `model.pth`
- Uses `SMDebug` hook for logging and visualization

### 🔍 2. Inference (`inference.py`)
- Implements SageMaker-compatible `model_fn`, `input_fn`, `predict_fn`, `output_fn`
- Supports JPEG input or URL via JSON
- Outputs predicted class index, label, and probability vector

### 🎯 3. Hyperparameter Tuning (`tuner.py`)
- Allows grid/random search using `learning_rate`, `batch_size`, etc.
- Selects best model based on validation accuracy
- Saves best model weights

---

## 🧪 Evaluation Metrics

- **Accuracy** on validation/test set
- **Loss curves** and logs via CloudWatch/SMDebug
- Inference tested on image files and JSON URLs
- Optional: confusion matrix and class-wise breakdown

---

## 🔧 Setup Instructions

```bash
# install dependencies
pip install torch torchvision sagemaker pandas matplotlib pillow tqdm requests
```

### Run Training
```bash
python train.py --data_dir data --save_path model --epochs 20
```

### Run Inference (Locally)
```bash
python inference.py
# Or deploy as a SageMaker endpoint
```

### Run Hyperparameter Tuning
```bash
python tuner.py --data_dir data --save_path model --epochs 50
```

---

## 🔐 Security and Deployment

- IAM roles scoped to SageMaker + S3
- Endpoint securely invoked via Lambda or API Gateway
- Model serialized to S3-compatible directory

---

## 🌟 Optional Enhancements

| Feature                | Status      |
|------------------------|-------------|
| Model Deployment       | ✅ Completed via `inference.py` |
| Hyperparameter Tuning  | ✅ Done via `tuner.py` |
| Cost Optimization      | ⬜ Optional Spot Instance config |
| Multi-Instance Training| ⬜ Optional future step |

---

## 📬 Contact

**Author**: Ishfaq Malik
**Email**: ishfakmalik@hotmail.com  
**LinkedIn**: [linkedin.com/in/yourprofile](https://www.linkedin.com/in/ishfaq-malik/)

---

## 📜 License

MIT License – see `LICENSE` for details.