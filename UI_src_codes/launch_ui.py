#!/usr/bin/env python3
"""
Launch script for Medical Chatbot UI
This script starts the FastAPI web server for the medical assistant interface.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'jinja2',
        'aiofiles',
        'PIL',
        'mcp'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package if package != 'PIL' else 'PIL')
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required dependencies are installed")
    return True


def check_medical_server():
    """Check if medical MCP server exists"""
    server_path = Path(__file__).parent.parent / "medical_mcp_server.py"
    if not server_path.exists():
        print(f"❌ Medical MCP server not found at: {server_path}")
        print("Please ensure the medical_mcp_server.py file exists in the parent directory")
        return False
    
    print("✅ Medical MCP server found")
    return True


def setup_environment():
    """Setup environment variables if needed"""
    # Check for .env file in parent directory
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print(f"✅ Found .env file at: {env_path}")
    else:
        print("⚠️  No .env file found. Make sure environment variables are set:")
        print("   - GEMINI_API_KEY (for routing)")
        print("   - PHI2_MODEL_PATH (optional)")
        print("   - MEDGEMMA_MODEL_NAME (optional)")


def main():
    """Main function to launch the UI application"""
    print("🏥 Medical Chatbot UI Launcher")
    print("=" * 50)
    
    # Change to the UI directory
    ui_dir = Path(__file__).parent
    os.chdir(ui_dir)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check medical server
    if not check_medical_server():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    print("\n🚀 Starting Medical Chatbot UI...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/api/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print(f"📁 Working directory: {os.getcwd()}")
    print("=" * 50)
    
    try:
        # Launch the FastAPI application
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
