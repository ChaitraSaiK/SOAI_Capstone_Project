#!/usr/bin/env python3
"""
Configuration file for Medical MCP Client/Server
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MedicalMCPConfig:
    """Configuration for Medical MCP Client/Server"""
    
    # Model paths and names
    phi2_model_path: str = "SFT/phi2-qlora-finetuned-med"
    medgemma_model_name: str = "google/medgemma-7b-instruct"
    
    # GPU configuration
    phi2_device: str = "cuda:0"  # GPU for Phi-2 model
    medgemma_device: str = "cuda:1"  # GPU for MedGemma model
    
    # API keys
    gemini_api_key: Optional[str] = None
    
    # Model parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    
    # Server configuration
    server_name: str = "medical-assistant"
    server_version: str = "1.0.0"
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        if self.gemini_api_key is None:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    
    @classmethod
    def load_from_env(cls) -> 'MedicalMCPConfig':
        """Load configuration from environment variables"""
        return cls(
            phi2_model_path=os.getenv("PHI2_MODEL_PATH", "SFT/phi2-qlora-finetuned-med"),
            medgemma_model_name=os.getenv("MEDGEMMA_MODEL_NAME", "google/medgemma-7b-instruct"),
            phi2_device=os.getenv("PHI2_DEVICE", "cuda:0"),
            medgemma_device=os.getenv("MEDGEMMA_DEVICE", "cuda:1"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "256")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
        )
    
    def validate(self) -> bool:
        """Validate the configuration"""
        import torch
        
        # Check if specified devices are available
        if "cuda" in self.phi2_device:
            gpu_id = int(self.phi2_device.split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                print(f"Warning: GPU {gpu_id} not available for Phi-2. Falling back to cuda:0")
                self.phi2_device = "cuda:0"
        
        if "cuda" in self.medgemma_device:
            gpu_id = int(self.medgemma_device.split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                print(f"Warning: GPU {gpu_id} not available for MedGemma. Falling back to cuda:0")
                self.medgemma_device = "cuda:0"
        
        # Check if model path exists
        if not os.path.exists(self.phi2_model_path):
            print(f"Warning: Phi-2 model path not found: {self.phi2_model_path}")
            return False
        
        return True
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 Medical MCP Configuration:")
        print(f"  Phi-2 Model Path: {self.phi2_model_path}")
        print(f"  MedGemma Model: {self.medgemma_model_name}")
        print(f"  Phi-2 Device: {self.phi2_device}")
        print(f"  MedGemma Device: {self.medgemma_device}")
        print(f"  Gemini API Key: {'Set' if self.gemini_api_key else 'Not set'}")
        print(f"  Max New Tokens: {self.max_new_tokens}")
        print(f"  Temperature: {self.temperature}")


# Default configuration instance
default_config = MedicalMCPConfig() 
