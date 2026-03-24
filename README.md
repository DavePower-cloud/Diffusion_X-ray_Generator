# 🩻 Diffusion X-Ray Generator  

Diffusion-based generative model for synthesising chest X-ray images using a 3D U-Net-style architecture and sinusoidal noise embeddings.  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  
![License](https://img.shields.io/badge/License-MIT-green)  
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)  
![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-lightgrey)  
![Model](https://img.shields.io/badge/Model-Diffusion%20Model-purple)  
![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41598--025--96336--5-blue)  
---

## 📌 Overview  

This repository implements a denoising diffusion probabilistic model (DDPM) adapted for **medical imaging**, specifically chest X-ray generation.  

The project demonstrates how diffusion models originally developed for natural images can be extended to a healthcare domain.  

The pipeline includes:  

- dataset loading and preprocessing  
- diffusion noise scheduling  
- sinusoidal timestep embeddings  
- U-Net denoising network  
- exponential moving average (EMA) stabilisation  
- image generation and visualisation  

---

## 🧱 Repository Structure  


src/
  config.py
  data.py
  diffusion.py
  model.py
  train.py
  generate.py
  utils.py

notebooks/
  demo_generation.ipynb

---

## 🚀 Quick Start

Install dependencies:
- pip install -r requirements.txt

Train the model:
- python src/train.py

Generate X-ray images from a trained model:
- python src/generate.py

---

## 📊 Features
Diffusion-based generative modelling (DDPM-style)
Sinusoidal timestep embeddings
U-Net denoising architecture
EMA stabilisation of weights
Config-driven training pipeline
Reproducible training and generation scripts

---


## ⚠️ Dataset  

The dataset is not included in this repository, but is publicly available on Kaggle under an MIT license:  

👉 https://www.kaggle.com/datasets/divyam6969/chest-xray-pneumonia-dataset

Expected structure:  

data/  
  chest_xray/  
    train/  
      bacterial_pneumonia/  
      viral_pneumonia/
      fungal_pneumonia

After downloading, place the dataset in the `data/` directory as shown above.  

---

## 📖 Acknowledgements

Parts of this implementation are adapted from diffusion model examples presented in Foster Generative AI by David Foster, originally applied to natural image datasets (e.g., flowers).

This repository extends and modifies those approaches for medical imaging applications, including changes to dataset handling, model configuration, and training workflow.

---

## 📚 Reference

Foster, D. Generative Deep Learning. 2nd Ed. O’Reilly Media.

---

## 🧠 Key Contribution

Demonstrates application of diffusion models to medical image generation
Translates generative AI methods from natural images to healthcare
Provides a clean, modular implementation for experimentation and extension

---

## 👤 Author

David Power
University College Cork 
