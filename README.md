# 🏭 MLOps Pipeline for Tabular Classification with ONNX & FastAPI

This project demonstrates a complete end-to-end **MLOps pipeline** for a tabular classification problem using the Titanic dataset. It covers training, evaluation, ONNX export, FastAPI deployment, Dockerization, and CI/CD automation.

---

## 🎯 Goals

- Build a full ML pipeline from scratch (data → training → inference)
- Train a tabular classification model using PyTorch 
- Export the trained model to **ONNX format**
- Serve predictions using **FastAPI**
- Containerize the system using **Docker**
- Automate deployment workflow using **GitHub Actions CI/CD**

---

## 📊 Dataset

We use the Titanic dataset from Kaggle:

https://www.kaggle.com/datasets/yasserh/titanic-dataset

---

## 🧠 Pipeline Overview

The system follows this workflow:

1. Data ingestion & preprocessing
2. Feature engineering
3. Model training (PyTorch MLP)
4. Model evaluation
5. Export trained model to ONNX
6. Load ONNX model using ONNX Runtime
7. Serve inference via FastAPI
8. Dockerize the application
9. CI/CD automation using GitHub Actions

---

## ⚙️ Tech Stack

- Python 3.11
- PyTorch
- ONNX
- ONNX Runtime
- FastAPI
- Uvicorn
- NumPy
- Docker
- GitHub Actions


