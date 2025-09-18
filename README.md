# 🏥 Medical AI Assistant - Capstone Project

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

An advanced medical AI assistant system combining custom-trained language models, fine-tuned medical domain expertise, and multimodal image analysis capabilities. This project implements a comprehensive medical AI solution using Model Context Protocol (MCP) architecture for intelligent request routing and specialized model selection.

## 🎯 Project Overview

This capstone project demonstrates a complete pipeline for building and deploying medical AI systems:

1. **Custom Model Training**: Training a 2.7B parameter language model from scratch
2. **Domain Specialization**: Fine-tuning Phi-2 model with medical datasets using QLoRA
3. **Multimodal Analysis**: Integrating MedGemma-7B for medical image understanding
4. **Intelligent Routing**: MCP-based architecture for optimal model selection
5. **User Interface**: Web-based application for seamless interaction

### Key Capabilities

- 🧠 **Medical Question Answering**: Specialized responses to medical queries
- 🖼️ **Medical Image Analysis**: X-rays, ultrasounds, and other medical imaging
- 🔄 **Intelligent Model Routing**: Automatic selection of the best model for each query
- 💬 **Conversational Interface**: Maintains context across medical consultations
- ⚡ **Multi-GPU Support**: Optimized for high-performance inference

## 🏗️ Project Architecture

```
SOAI_Capstone_Project/
├── Base_LLM_Training/          # Custom 2.7B model training from scratch
├── SFT/                        # Supervised Fine-Tuning with QLoRA
├── UI_src_codes/               # Web application interface
├── images/                     # Sample medical images for testing
├── medical_mcp_server.py       # MCP server for model management
├── medical_mcp_client.py       # MCP client for interactions
├── launch_medical_mcp.py       # Main launcher with setup utilities
└── config.py                   # Configuration management
```

## 📁 File Structure & Components

### Core System Files

- **`launch_medical_mcp.py`** - 🚀 **Main launcher script** with automated setup, dependency checking, and system initialization
- **`medical_mcp_server.py`** - 🖥️ **MCP server** that manages multiple AI models and handles intelligent routing
- **`medical_mcp_client.py`** - 💬 **MCP client** for interactive sessions and API communications
- **`config.py`** - ⚙️ **Configuration manager** for model paths, GPU allocation, and environment variables

### Training Components

- **`Base_LLM_Training/`** - 🧠 **Custom model training** directory containing scripts for training a 2.7B parameter model from scratch
  - `SmolLM_training.py` - Training script for the base language model
  - `README.md` - Training documentation and instructions
  
- **`SFT/`** - 🎯 **Supervised Fine-Tuning** directory for medical domain specialization
  - `QLoRA_SFT.ipynb` - Jupyter notebook for QLoRA fine-tuning of Phi-2
  - `dataset_creation.ipynb` - Dataset preparation and processing
  - `phi2-qlora-finetuned-med/` - Fine-tuned model artifacts and adapters
  - `dataset.7z` & `final_dataset.7z` - Compressed medical training datasets

### User Interface

- **`UI_src_codes/`** - 🌐 **Web application** for user interaction
  - `launch_ui.py` - UI launcher with dependency checking
  - `app.py` - FastAPI application backend
  - `templates/` - HTML templates for the web interface
  - `static/` - CSS, JavaScript, and static assets
  - `uploads/` - Directory for uploaded medical images

### Supporting Files

- **`images/`** - 📸 **Sample medical images** for testing (ultrasounds, X-rays, etc.)
- **`requirements.txt`** - 📦 **Python dependencies** for the entire project
- **`Dockerfile`** - 🐳 **Docker configuration** for containerized deployment
- **`commands.txt`** - 📝 **Usage examples** and Docker commands

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU(s)** (2 GPUs recommended for optimal performance)
- **16GB+ GPU memory** (for running multiple models)
- **Hugging Face account** with access token
- **Google Gemini API key** (optional, for enhanced routing)

### Installation Steps

#### Option 1: Docker Setup (Recommended)

1. **Build and run the Docker container:**
```bash
# Clone the repository
git clone https://github.com/ChaitraSaiK/SOAI_Capstone_Project/
cd SOAI_Capstone_Project

# Run with GPU support
docker run --gpus all --shm-size=100g -p 8501:8000 -it -v /path/to/project:/DockerImage <DockerImage_Name> bash
```

2. **Setup Hugging Face authentication:**
```bash
huggingface-cli login
# Enter your HF token when prompted
```

3. **Initialize the system:**
```bash
python launch_medical_mcp.py --setup
```

#### Option 2: Local Installation

1. **Clone and install dependencies:**
```bash
git clone https://github.com/ChaitraSaiK/SOAI_Capstone_Project/
cd SOAI_Capstone_Project
pip install -r requirements.txt
```

2. **Setup environment variables:**
```bash
# Create .env file with your API keys
echo "GEMINI_API_KEY=your_gemini_api_key" > .env
echo "PHI2_MODEL_PATH=SFT/phi2-qlora-finetuned-med" >> .env
echo "MEDGEMMA_MODEL_NAME=google/medgemma-7b-instruct" >> .env
```

3. **Run setup and verification:**
```bash
python launch_medical_mcp.py --setup
```

## 🎮 How to Run the Application

### Method 1: Complete System (MCP + UI)

```bash
# Start the MCP client (interactive terminal)
python launch_medical_mcp.py --client

# Or start just the MCP server
python launch_medical_mcp.py --server
```

### Method 2: Web Interface Only

```bash
# Navigate to UI directory and launch
cd UI_src_codes
python launch_ui.py
```

Visit **http://localhost:8000** to access the web interface.

### Method 3: Quick System Check

```bash
# Verify all components are working
python launch_medical_mcp.py --test

# Check system status
python launch_medical_mcp.py --status
```

## 💡 Usage Examples

### Text-Based Medical Queries
```
User: "What are the symptoms of diabetes?"
AI: [Provides comprehensive medical information about diabetes symptoms]
```

### Medical Image Analysis
```
1. Upload medical image through web interface
2. Ask: "What do you see in this X-ray?"
3. AI analyzes using MedGemma-7B and provides detailed interpretation
```


## 🔧 Configuration Options

### GPU Configuration
- **Single GPU**: Both models will use the same GPU automatically
- **Multi-GPU**: Configure specific GPUs for each model in `.env`:
```bash
PHI2_DEVICE=cuda:0
MEDGEMMA_DEVICE=cuda:1
```

### Model Parameters
```bash
MAX_NEW_TOKENS=256          # Maximum response length
TEMPERATURE=0.7             # Response creativity (0.0-1.0)
```

### Advanced Configuration
Edit `config.py` to customize:
- Model paths and versions
- Device allocation strategies
- Generation parameters
- Server settings

## 🛠️ Technical Stack

- **🧠 AI Models**: 
  - Custom 2.7B parameter model (trained from scratch)
  - Fine-tuned Phi-2 with QLoRA (medical domain)
  - MedGemma-7B-Instruct (multimodal medical analysis)
  
- **🏗️ Framework**: 
  - PyTorch for model training and inference
  - Transformers library for model management
  - PEFT for efficient fine-tuning
  
- **🌐 Backend**: 
  - FastAPI for REST API
  - MCP (Model Context Protocol) for intelligent routing
  - Uvicorn ASGI server
  
- **🎨 Frontend**: 
  - Modern responsive web UI
  - Real-time chat interface
  - Image upload and analysis

## 📊 Model Performance

- **Base Model**: 2.7B parameters trained on diverse text corpus
- **Medical Fine-tuning**: Specialized on medical literature and Q&A datasets
- **Inference Speed**: Optimized for real-time responses
- **Memory Usage**: Efficient with GPU memory management
- **Accuracy**: Validated on medical benchmark datasets

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Use smaller batch sizes or single GPU mode
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **Model Not Found**:
   ```bash
   # Verify model path exists
   python launch_medical_mcp.py --check
   ```

3. **API Key Issues**:
   ```bash
   # Check environment variables
   python launch_medical_mcp.py --status
   ```

### Getting Help

- Check system status: `python launch_medical_mcp.py --status`
- Run diagnostics: `python launch_medical_mcp.py --check`
- View logs in the console output for detailed error messages

## 📚 Project Development

This project demonstrates:
- **End-to-end ML pipeline** from data preparation to deployment
- **Advanced fine-tuning techniques** using QLoRA and PEFT
- **Multi-model architecture** with intelligent routing
- **Production-ready deployment** with Docker and web interface
- **Medical domain expertise** integration and validation

## 🤝 Contributing

This is a capstone project showcasing advanced AI techniques in medical applications. The system provides a foundation for further medical AI research and development.

## 📖 Repository

**GitHub**: [https://github.com/ChaitraSaiK/SOAI_Capstone_Project/](https://github.com/ChaitraSaiK/SOAI_Capstone_Project/)

---

*This project represents a comprehensive implementation of modern AI techniques applied to medical assistance, demonstrating the integration of custom model training, domain fine-tuning, and intelligent multi-modal analysis in a production-ready system.*
