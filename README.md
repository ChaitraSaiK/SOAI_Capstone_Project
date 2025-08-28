# Medical Assistant Chatbot UI

A professional web-based user interface for the Medical AI Assistant, built with FastAPI and modern web technologies.

## Features

- **Interactive Chat Interface**: Modern chatbot-style conversation interface
- **Text Queries**: Ask medical questions and get AI-powered responses
- **Image Analysis**: Upload medical images for AI analysis
- **Session Management**: Maintain conversation context across interactions
- **Professional Design**: Clean, medical-themed UI with responsive design
- **Real-time Communication**: Fast, asynchronous communication with the AI backend

## Architecture

```
medical_ui/
├── app.py                 # FastAPI backend application
├── launch_ui.py          # Launch script
├── requirements.txt      # UI-specific dependencies
├── static/               # Static assets
│   ├── css/
│   │   └── style.css    # Main stylesheet
│   └── js/
│       └── app.js       # Frontend JavaScript
├── templates/            # HTML templates
│   └── index.html       # Main chat interface
└── uploads/             # Uploaded images storage
```

## Installation

1. **Navigate to the UI directory**:
   ```bash
   cd medical_ui
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the main medical server is available**:
   - The UI connects to `../medical_mcp_server.py`
   - Make sure all main project dependencies are installed

## Usage

### Option 1: Using the launch script (Recommended)
```bash
python launch_ui.py
```

### Option 2: Direct FastAPI launch
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Python direct execution
```bash
python app.py
```

## Accessing the Application

Once started, the application will be available at:
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Chat Endpoints
- `POST /api/chat` - Send text messages
- `POST /api/chat/image` - Upload and analyze images
- `GET /api/chat/history/{session_id}` - Get conversation history
- `DELETE /api/chat/session/{session_id}` - End a session

### Utility Endpoints
- `GET /health` - Health check
- `GET /api/sessions` - List active sessions (debug)

## Features Guide

### Text Chat
1. Type your medical question in the input field
2. Press Enter or click the send button
3. Wait for the AI response

### Image Analysis
1. Click the attachment button (📎)
2. Upload an image by:
   - Clicking the upload area and selecting a file
   - Dragging and dropping an image
3. Optionally add a description or specific question
4. Click "Analyze Image"

### Session Management
- Each conversation maintains context within a session
- Use "New Session" to start fresh
- Use "Clear Chat" to clear the current conversation

## Configuration

The UI application automatically connects to the medical MCP server. Ensure the following:

1. **Environment Variables** (optional, inherits from main project):
   - `GEMINI_API_KEY` - For query routing
   - `PHI2_MODEL_PATH` - Path to fine-tuned model
   - `MEDGEMMA_MODEL_NAME` - MedGemma model identifier

2. **Model Files**: Ensure the medical models are properly set up in the parent directory

## Troubleshooting

### Common Issues

1. **Server won't start**:
   - Check if port 8000 is available
   - Ensure all dependencies are installed
   - Verify the medical MCP server exists

2. **AI responses not working**:
   - Check the medical MCP server is functioning
   - Verify environment variables are set
   - Check server logs for errors

3. **Image upload issues**:
   - Ensure `uploads/` directory exists and is writable
   - Check file size (max 10MB)
   - Verify image format (JPG, PNG, GIF supported)

### Debug Mode

For debugging, you can access:
- API docs at `/api/docs`
- Active sessions at `/api/sessions`
- Server logs in the terminal

## Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Styling**: Custom CSS with modern design principles
- **Icons**: Font Awesome
- **Fonts**: Inter (Google Fonts)

## Security Notes

- File uploads are validated for type and size
- Session IDs are randomly generated
- CORS is enabled for development (adjust for production)
- No authentication implemented (add as needed)

## Development

To modify the UI:

1. **Frontend changes**: Edit files in `static/` and `templates/`
2. **Backend changes**: Edit `app.py`
3. **Styling**: Modify `static/css/style.css`
4. **Functionality**: Update `static/js/app.js`

The application runs with auto-reload enabled during development.

## License

This UI application is part of the Medical AI Assistant project.
