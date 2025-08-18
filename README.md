---
title: House Plant Species Identifier API
sdk: docker
app_file: app.py
---

# 🌿 House Plant Species Identifier API

This repository contains the containerized Flask backend for the **House Plant Species Identifier**, a full-stack, decoupled machine learning application.

### Project Overview

This API serves a fine-tuned PyTorch (EfficientNet-B0) model capable of identifying 47 different species of house plants. It receives an image file via a `POST` request to the `/predict` endpoint and returns the predicted species and a confidence score in JSON format.

This backend is designed to be a standalone microservice, containerized with Docker, and is currently deployed on Hugging Face Spaces.

### 🔗 Project Links

| Link                                   | URL                                                                                                         |
| :------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| 🚀 **Live Demo**                       | **[house-plant-frontend.vercel.app](https://house-plant-frontend-3vsr32tzq-md-ehsanul-haque-kanans-projects.vercel.app/)** |
| 🎨 **Frontend Repository (React)**     | [github.com/MdEhsanulHaqueKanan/house-plant-frontend](https://github.com/MdEhsanulHaqueKanan/house-plant-frontend) |
| ⚙️ **Backend API Repository (This Repo)** | [github.com/MdEhsanulHaqueKanan/house-plant-api](https://github.com/MdEhsanulHaqueKanan/house-plant-api)       |
| 📦 **Original Monolithic Project**     | [github.com/MdEhsanulHaqueKanan/house-plant-species-identifier-machine-learning-flask-app](https://github.com/MdEhsanulHaqueKanan/house-plant-species-identifier-machine-learning-flask-app) |


### Technology Stack

*   **Backend:** Python, Flask
*   **Machine Learning:** PyTorch, Torchvision, Pillow
*   **Containerization:** Docker
*   **Deployment:** Hugging Face Spaces