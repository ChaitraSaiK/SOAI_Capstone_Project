# Medical AI Assistant

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A medical AI assistant with custom pre-trained models, fine-tuned Phi-2, and multimodal image analysis capabilities.

## Features

- **2.7B parameter model** trained from scratch on multiple GPUs
- **Fine-tuned Phi-2** with medical domain data
- **Medical image analysis** using MedGemma-4B-IT
- **MCP agent architecture** for intelligent routing
- **Web-based UI** for easy interaction

## Quick Start

git clone https://github.com/ChaitraSaiK/SOAI_Capstone_Project/
cd SOAI_Capstone_Project/medical_ui
pip install -r requirements.txt
python launch_ui.py

Visit http://localhost:8000 to use the medical assistant.

## Environment Variables
GEMINI_API_KEY=your_api_key
PHI2_MODEL_PATH=path_to_model
MEDGEMMA_MODEL_NAME=google/medgemma-4b-it

## Usage

- **Text Chat**: Ask medical questions through the web interface
- **Image Analysis**: Upload medical images for AI analysis
- **Session Management**: Maintain conversation context

## Tech Stack

- FastAPI backend with Python 3.8+
- Custom 2.7B parameter model + Fine-tuned Phi-2
- MedGemma-4B-IT for multimodal analysis
- Modern web UI with responsive design

## Repository

[https://github.com/ChaitraSaiK/SOAI_Capstone_Project/](https://github.com/ChaitraSaiK/SOAI_Capstone_Project/)
