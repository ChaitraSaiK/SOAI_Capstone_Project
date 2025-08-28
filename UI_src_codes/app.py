#!/usr/bin/env python3
"""
Medical Chatbot UI Application
FastAPI backend for the medical assistant chatbot interface.
"""

import asyncio
import base64
import io
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from medical_mcp_client import MedicalMCPClient


# Pydantic models for request/response
class ChatMessage(BaseModel):
    message: str
    timestamp: str
    sender: str  # 'user' or 'assistant'
    message_type: str = 'text'  # 'text' or 'image'
    image_url: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    message_type: str = 'text'


class ImageChatRequest(BaseModel):
    description: str = ""
    session_id: Optional[str] = None


# FastAPI app initialization
app = FastAPI(
    title="Medical Assistant Chatbot",
    description="Professional medical AI assistant for healthcare professionals",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Global variables
medical_client: Optional[MedicalMCPClient] = None
active_sessions: Dict[str, MedicalMCPClient] = {}
chat_histories: Dict[str, List[ChatMessage]] = {}
upload_directory = Path(__file__).parent / "uploads"

# Ensure upload directory exists
upload_directory.mkdir(exist_ok=True)


async def get_or_create_client(session_id: str) -> MedicalMCPClient:
    """Get or create a medical client for a session"""
    if session_id not in active_sessions:
        # Create new client for this session
        # Use absolute path to the parent directory's medical server
        parent_dir = Path(__file__).parent.parent
        server_script_path = str(parent_dir / "medical_mcp_server.py")
        
        print(f"🔌 Creating client for session {session_id}")
        print(f"📁 Server script path: {server_script_path}")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 Parent directory: {parent_dir}")
        print(f"📁 Server exists: {Path(server_script_path).exists()}")
        
        # Change working directory to parent before creating client
        original_cwd = os.getcwd()
        os.chdir(parent_dir)
        print(f"📁 Changed working directory to: {os.getcwd()}")
        
        try:
            client = MedicalMCPClient(server_script_path="./medical_mcp_server.py")
            print(f"🔗 Attempting to connect...")
            await client.connect()
            active_sessions[session_id] = client
            chat_histories[session_id] = []
            print(f"✅ Created new session: {session_id}")
        except Exception as e:
            print(f"❌ Failed to create client for session {session_id}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to connect to medical service: {str(e)}")
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            print(f"📁 Restored working directory to: {os.getcwd()}")
    
    return active_sessions[session_id]


def add_to_chat_history(session_id: str, message: str, sender: str, message_type: str = 'text', image_url: Optional[str] = None):
    """Add message to chat history"""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    chat_message = ChatMessage(
        message=message,
        timestamp=datetime.now().isoformat(),
        sender=sender,
        message_type=message_type,
        image_url=image_url
    )
    chat_histories[session_id].append(chat_message)


def validate_image(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/bmp"]
    return file.content_type in allowed_types and file.size <= 10 * 1024 * 1024  # 10MB limit


async def save_uploaded_image(file: UploadFile, session_id: str) -> str:
    """Save uploaded image and return file path"""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{session_id}_{timestamp}.{file_extension}"
    file_path = upload_directory / filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return str(file_path)


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_string = base64.b64encode(image_data).decode('utf-8')
            return base64_string
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("🚀 Starting Medical Chatbot UI Application")
    print(f"📁 Upload directory: {upload_directory}")
    print("✅ Application ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🔄 Shutting down application...")
    
    # Close all active sessions
    for session_id, client in active_sessions.items():
        try:
            await client.disconnect()
            print(f"Closed session: {session_id}")
        except Exception as e:
            print(f"Error closing session {session_id}: {e}")
    
    active_sessions.clear()
    chat_histories.clear()
    print("✅ Cleanup completed!")


# Routes

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chatbot interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_text(request: ChatRequest):
    """Handle text-based chat messages"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get or create client
        client = await get_or_create_client(session_id)
        
        # Add user message to history
        add_to_chat_history(session_id, request.message, "user")
        
        # Get response from medical assistant using intelligent routing
        try:
            print(f"💭 Analyzing question: {request.message}")
            
            # Use the same routing logic as the main application
            # First, route the query to see if it should use image analysis or medical QA
            route_response = await client.route_query(request.message, bool(client.image_history))
            print(f"🔀 Route response: {route_response}")
            
            # Check if there's a relevant image for this question
            relevant_image = client.find_image_for_query(request.message)
            
            if relevant_image:
                # If we found a relevant image, use image question analysis
                print(f"🖼️ Found relevant image for question: {relevant_image['path']}")
                response = await client.ask_image_question(None, request.message)
                print(f"🔍 Image-based response received")
            else:
                # If no relevant image found, use medical QA
                print("🩺 Using medical knowledge base")
                response = await client.ask_medical_question(request.message)
                print(f"💡 Medical QA response received")
            
            print(f"🤖 Raw response from medical server: {response[:200]}...")
            
            # Don't reject responses that contain "Error" as they might be legitimate medical content
            if not response or response.strip() == "":
                raise Exception(f"Empty response from medical server")
        except Exception as e:
            print(f"Error getting medical response: {e}")
            import traceback
            traceback.print_exc()
            response = f"I apologize, but I'm experiencing technical difficulties: {str(e)}. Please consult a healthcare professional for medical advice."
        
        # Add assistant response to history
        add_to_chat_history(session_id, response, "assistant")
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            message_type="text"
        )
        
    except Exception as e:
        print(f"Error in chat_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/image")
async def chat_image(
    file: UploadFile = File(...),
    description: str = Form(""),
    session_id: Optional[str] = Form(None)
):
    """Handle image-based chat messages"""
    try:
        # Validate image
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Generate session ID if not provided
        session_id = session_id or str(uuid.uuid4())
        
        # Get or create client
        client = await get_or_create_client(session_id)
        
        # Save uploaded image
        image_path = await save_uploaded_image(file, session_id)
        image_url = f"/static/uploads/{Path(image_path).name}"
        
        # Add user message to history (with image)
        user_message = f"Image uploaded: {file.filename}"
        if description:
            user_message += f" - {description}"
        add_to_chat_history(session_id, user_message, "user", "image", image_url)
        
        # Analyze image with medical assistant using the same routing as main app
        try:
            print(f"🖼️ Analyzing uploaded image: {file.filename}")
            
            # Use analyze_medical_image to get the analysis and add to client's image history
            if description:
                response = await client.ask_image_question(image_path, description)
                print(f"🔍 Image question response received")
            else:
                response = await client.analyze_medical_image(image_path, "")
                print(f"🔍 Image analysis response received")
            
            print(f"🖼️ Raw image analysis response: {response[:200]}...")
            
            # Don't reject responses that contain "Error" as they might be legitimate medical content
            if not response or response.strip() == "":
                raise Exception(f"Empty response from medical server")
        except Exception as e:
            print(f"Error analyzing medical image: {e}")
            import traceback
            traceback.print_exc()
            response = f"I apologize, but I'm unable to analyze this image: {str(e)}. Please consult a healthcare professional for medical advice."
        
        # Add assistant response to history
        add_to_chat_history(session_id, response, "assistant")
        
        return JSONResponse({
            "response": response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message_type": "text",
            "image_url": image_url
        })
        
    except Exception as e:
        print(f"Error in chat_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/image/clipboard")
async def chat_image_clipboard(
    image_data: str = Form(...),
    description: str = Form(""),
    session_id: Optional[str] = Form(None)
):
    """Handle clipboard image paste"""
    try:
        # Generate session ID if not provided
        session_id = session_id or str(uuid.uuid4())
        
        # Get or create client
        client = await get_or_create_client(session_id)
        
        # Process base64 image data
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Create PIL image to validate
            img = Image.open(io.BytesIO(image_bytes))
            
            # Save clipboard image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{session_id}_clipboard_{timestamp}.png"
            image_path = upload_directory / filename
            
            # Save image
            img.save(image_path, 'PNG')
            image_url = f"/static/uploads/{filename}"
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Add user message to history (with image)
        user_message = "Image pasted from clipboard"
        if description:
            user_message += f" - {description}"
        add_to_chat_history(session_id, user_message, "user", "image", image_url)
        
        # Analyze image with medical assistant using intelligent routing
        try:
            print(f"📋 Analyzing clipboard image")
            
            # Use the same routing as main app for clipboard images
            if description:
                response = await client.ask_image_question(str(image_path), description)
                print(f"🔍 Clipboard image question response received")
            else:
                response = await client.analyze_medical_image(str(image_path), "")
                print(f"🔍 Clipboard image analysis response received")
            
            print(f"📋 Raw clipboard image analysis response: {response[:200]}...")
            
            # Don't reject responses that contain "Error" as they might be legitimate medical content
            if not response or response.strip() == "":
                raise Exception(f"Empty response from medical server")
        except Exception as e:
            print(f"Error analyzing clipboard image: {e}")
            import traceback
            traceback.print_exc()
            response = f"I apologize, but I'm unable to analyze this image: {str(e)}. Please consult a healthcare professional for medical advice."
        
        # Add assistant response to history
        add_to_chat_history(session_id, response, "assistant")
        
        return JSONResponse({
            "response": response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message_type": "text",
            "image_url": image_url
        })
        
    except Exception as e:
        print(f"Error in chat_image_clipboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        history = chat_histories.get(session_id, [])
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        print(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/session/{session_id}")
async def end_chat_session(session_id: str):
    """End a chat session and clean up resources"""
    try:
        # Close MCP client connection
        if session_id in active_sessions:
            client = active_sessions[session_id]
            await client.disconnect()
            del active_sessions[session_id]
        
        # Clear chat history
        if session_id in chat_histories:
            del chat_histories[session_id]
        
        return {"message": f"Session {session_id} ended successfully"}
        
    except Exception as e:
        print(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_active_sessions():
    """List all active sessions (for debugging)"""
    return {
        "active_sessions": list(active_sessions.keys()),
        "total_sessions": len(active_sessions)
    }


@app.post("/api/debug/test-connection")
async def test_medical_connection():
    """Test connection to medical MCP server"""
    try:
        # Create a test client
        test_client = MedicalMCPClient(
            server_script_path=str(Path(__file__).parent.parent / "medical_mcp_server.py")
        )
        
        # Try to connect
        await test_client.connect()
        
        # Try a simple medical question
        test_response = await test_client.ask_medical_question("What is a headache?")
        
        # Disconnect
        await test_client.disconnect()
        
        return {
            "status": "success",
            "message": "Successfully connected to medical server",
            "test_response": test_response[:200] + "..." if len(test_response) > 200 else test_response
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return {
            "status": "error",
            "message": f"Failed to connect to medical server: {str(e)}",
            "error_details": error_trace
        }


# Serve uploaded images
@app.get("/static/uploads/{filename}")
async def serve_uploaded_image(filename: str):
    """Serve uploaded images"""
    file_path = upload_directory / filename
    if file_path.exists():
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    import uvicorn
    
    print("🏥 Medical Chatbot UI")
    print("🔗 Starting FastAPI server...")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
