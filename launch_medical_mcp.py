#!/usr/bin/env python3
"""
Medical MCP Launcher
Easy setup and launch script for the Medical MCP Client/Server
"""

import os
import sys
import asyncio
import subprocess
import argparse
from pathlib import Path

def load_env():
    """Load environment variables"""
    try:
        from dotenv import load_dotenv
        # Get the directory containing this script
        script_dir = Path(__file__).parent.absolute()
        env_path = script_dir / '.env'
        if load_dotenv(env_path):
            print("✅ Environment variables loaded")
            return True
    except ImportError:
        print("⚠️  python-dotenv not available")
    return False

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Load environment variables first
    load_env()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
            if gpu_count < 2:
                print("⚠️  Warning: Only 1 GPU available. Both models will use the same GPU.")
        else:
            print("❌ CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check model path
    model_path = Path("/app/SFT/phi2-qlora-finetuned-med")
    abs_model_path = model_path.resolve()
    if model_path.exists():
        print(f"✅ Fine-tuned model found: {abs_model_path}")
    else:
        print(f"❌ Fine-tuned model not found: {abs_model_path}")
        print(f"   Checked path: {model_path}")
        print(f"   Current directory: {Path.cwd()}")
        return False
    
    # Check PEFT adapter
    peft_path = model_path / "peft_model"
    if peft_path.exists():
        print(f"✅ PEFT adapter found: {peft_path.resolve()}")
    else:
        print(f"❌ PEFT adapter not found: {peft_path.resolve()}")
        return False
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ Gemini API key configured")
    else:
        print("⚠️  Warning: Gemini API key not set. Routing will use fallback logic.")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Setup environment variables"""
    print("🔧 Setting up environment...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        
        # Get API key from user
        api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
        
        env_content = f"""# Medical MCP Configuration
# GEMINI_API_KEY={api_key}
PHI2_MODEL_PATH=/app/SFT/phi2-qlora-finetuned-med
MEDGEMMA_MODEL_NAME=google/medgemma-7b-instruct
PHI2_DEVICE=cuda:0
MEDGEMMA_DEVICE=cuda:0
MAX_NEW_TOKENS=256
TEMPERATURE=0.7
"""
        
        env_file.write_text(env_content)
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")
    
    return True

def run_tests():
    """Run system tests"""
    print("🧪 Running tests...")
    
    try:
        from medical_mcp_client import test_client
        asyncio.run(test_client())
        return True
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False

def start_interactive_client():
    """Start the interactive MCP client"""
    print("🚀 Starting Medical MCP Client...")
    
    try:
        from medical_mcp_client import interactive_session
        asyncio.run(interactive_session())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting client: {e}")

def start_server_only():
    """Start only the MCP server"""
    print("🚀 Starting Medical MCP Server...")
    
    try:
        import medical_mcp_server
        # The server will start automatically when imported
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def print_status():
    """Print system status"""
    print("📊 Medical MCP System Status")
    print("=" * 40)
    
    from config import MedicalMCPConfig
    config = MedicalMCPConfig.load_from_env()
    config.print_config()
    
    print("\n🔧 System Validation:")
    is_valid = config.validate()
    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has issues")
    
    print("\n📋 Available Commands:")
    print("  python launch_medical_mcp.py --setup    # Initial setup")
    print("  python launch_medical_mcp.py --client   # Start interactive client")
    print("  python launch_medical_mcp.py --server   # Start server only")
    print("  python launch_medical_mcp.py --test     # Run tests")
    print("  python launch_medical_mcp.py --status   # Show this status")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Medical MCP Launcher")
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument("--setup", action="store_true", 
                      help="Run initial setup (install deps, create .env)")
    group.add_argument("--client", action="store_true", 
                      help="Start interactive MCP client")
    group.add_argument("--server", action="store_true", 
                      help="Start MCP server only")
    group.add_argument("--test", action="store_true", 
                      help="Run system tests")
    group.add_argument("--status", action="store_true", 
                      help="Show system status")
    group.add_argument("--check", action="store_true", 
                      help="Check requirements only")
    
    args = parser.parse_args()
    
    print("🏥 Medical MCP Launcher")
    print("=" * 25)
    
    if args.setup:
        print("🔧 Running initial setup...")
        if not install_dependencies():
            sys.exit(1)
        if not setup_environment():
            sys.exit(1)
        if not check_requirements():
            print("⚠️  Setup completed but some requirements are not met")
            sys.exit(1)
        print("✅ Setup completed successfully!")
        
    elif args.check:
        if check_requirements():
            print("✅ All requirements met!")
            sys.exit(0)
        else:
            print("❌ Some requirements are not met")
            sys.exit(1)
            
    elif args.test:
        if not check_requirements():
            print("❌ Requirements not met. Run --setup first.")
            sys.exit(1)
        run_tests()
        
    elif args.server:
        if not check_requirements():
            print("❌ Requirements not met. Run --setup first.")
            sys.exit(1)
        start_server_only()
        
    elif args.client:
        if not check_requirements():
            print("❌ Requirements not met. Run --setup first.")
            sys.exit(1)
        start_interactive_client()
        
    elif args.status:
        print_status()
        
    else:
        # Default: check requirements and start client if ready
        if not check_requirements():
            print("\n❌ Requirements not met.")
            print("💡 Run: python launch_medical_mcp.py --setup")
            sys.exit(1)
        
        print("\n🚀 Starting Medical MCP Client...")
        start_interactive_client()

if __name__ == "__main__":
    main() 
