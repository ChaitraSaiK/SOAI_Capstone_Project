#!/usr/bin/env python3
"""
Medical MCP Server
Implements medical question answering and medical image analysis capabilities.
"""

import asyncio
import base64
import io
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    
    # Get the directory containing the server script
    server_dir = Path(__file__).parent.absolute()
    print(f"Server directory: {server_dir}", flush=True)
    
    # Look for .env in the server directory
    env_path = server_dir / '.env'
    print(f"Looking for .env at: {env_path}", flush=True)
    
    # Try to load .env file
    env_loaded = load_dotenv(env_path)
    if env_loaded:
        print("✅ Environment variables loaded from .env file", flush=True)
    else:
        print("❌ Failed to load .env file", flush=True)
        print(f"Current working directory: {os.getcwd()}", flush=True)
    
except ImportError:
    print("python-dotenv not available, using system environment variables", flush=True)

import torch

# Disable torch compilation completely - multiple methods
os.environ["PYTORCH_DISABLE_COMPILATION"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.reset()
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.cache_size_limit = 1
    print("Torch dynamo aggressively disabled", flush=True)
except ImportError:
    pass
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel
import google.generativeai as genai

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent


# Conversation history tracking
@dataclass
class Message:
    """Represents a single message in the conversation"""
    content: str
    response: str
    timestamp: datetime = field(default_factory=datetime.now)
    has_image: bool = False
    image_description: str = ""

@dataclass
class ImageContext:
    """Tracks information about an image"""
    path: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    key_terms: List[str] = field(default_factory=list)  # Important terms from the image description

@dataclass
class ConversationHistory:
    """Tracks conversation history for a session"""
    messages: List[Message] = field(default_factory=list)
    images: List[ImageContext] = field(default_factory=list)
    current_image_idx: int = -1  # Index of the current image being discussed
    
    def add_image(self, path: str, description: str) -> None:
        """Add a new image to the history"""
        # Store the full description for better context matching
        self.images.append(ImageContext(
            path=path,
            description=description,
            key_terms=[]  # We'll use full description matching instead
        ))
        self.current_image_idx = len(self.images) - 1
        
        # Add the initial image analysis as a message too
        self.add_message("", description, has_image=True, image_description=description)
    
    def get_current_image(self) -> Optional[ImageContext]:
        """Get the current image context"""
        if 0 <= self.current_image_idx < len(self.images):
            return self.images[self.current_image_idx]
        return None
    
    def find_relevant_image(self, query: str) -> Optional[ImageContext]:
        """Find the most relevant image based on the query and conversation history"""
        if not self.images:
            return None
            
        query_lower = query.lower()
        
        # First, check for direct matches with image descriptions
        for img in self.images:
            desc_lower = img.description.lower()
            # If the query terms are found in the image description
            if any(term in desc_lower for term in query_lower.split()):
                self.current_image_idx = self.images.index(img)
                return img
        
        # If no direct match found, check recent messages for context
        recent_msgs = self.messages[-3:] if self.messages else []
        for msg in reversed(recent_msgs):
            msg_lower = (msg.content + " " + msg.response).lower()
            # Find which image's description matches the recent conversation
            for img in self.images:
                desc_lower = img.description.lower()
                if any(term in desc_lower for term in msg_lower.split()):
                    self.current_image_idx = self.images.index(img)
                    return img
        
        # If still no match found, keep current image
        return self.get_current_image()
    
    def add_message(self, content: str, response: str, has_image: bool = False, image_description: str = "") -> None:
        """Add a new message to the history"""
        self.messages.append(Message(
            content=content,
            response=response,
            has_image=has_image,
            image_description=image_description
        ))
    
    def get_context(self, last_n: int = 5) -> List[Message]:
        """Get the last N messages for context"""
        return self.messages[-last_n:] if self.messages else []
    
    def clear(self) -> None:
        """Clear the conversation history"""
        self.messages.clear()
        self.current_image_path = None
        self.current_image_description = None

# Global instances
qa_model = None
qa_tokenizer = None
qa_initialized = False

image_analyzer = None
image_tokenizer = None  
image_initialized = False

router_model = None

# Dictionary to store conversation histories by session ID
conversation_histories: Dict[str, ConversationHistory] = {}

# Initialize FastMCP server
mcp = FastMCP("medical-assistant")

# Configuration  
PHI2_MODEL_PATH = os.path.abspath("SFT/phi2-qlora-finetuned-med")
PHI2_DEVICE = "cuda:0"  
MEDGEMMA_DEVICE = "cuda:1"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
print("Gemini key:", GEMINI_API_KEY)

# Set environment variables to avoid compilation issues globally
os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_CUDA_ARCH_LIST"] = ""
os.environ["FORCE_CUDA"] = "0"
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["TRITON_DISABLE_COMPILE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

print(f"Current working directory: {os.getcwd()}", flush=True)
print(f"PHI2_MODEL_PATH: {PHI2_MODEL_PATH}", flush=True)
print(f"PEFT path exists: {os.path.exists(os.path.join(PHI2_MODEL_PATH, 'peft_model'))}", flush=True)

async def initialize_qa_model():
    """Initialize the Phi-2 medical QA model"""
    global qa_model, qa_tokenizer, qa_initialized
    
    if qa_initialized:
        return
        
    print(f"Loading Phi-2 medical model on {PHI2_DEVICE}...", flush=True)
    
    # Load tokenizer from the same path as your working code
    print(f"Loading tokenizer from: {PHI2_MODEL_PATH}", flush=True)
    qa_tokenizer = AutoTokenizer.from_pretrained(
        PHI2_MODEL_PATH,
        trust_remote_code=True
    )
    if qa_tokenizer.pad_token is None:
        qa_tokenizer.pad_token = qa_tokenizer.eos_token
    
    print(f"Tokenizer loaded. Vocab size: {qa_tokenizer.vocab_size}", flush=True)
    print(f"Special tokens: eos={qa_tokenizer.eos_token}, pad={qa_tokenizer.pad_token}", flush=True)
    
    # Load PEFT config to get base model name (match your working approach)
    from peft import PeftConfig
    peft_path = os.path.join(PHI2_MODEL_PATH, "peft_model")
    print(f"Loading PEFT config from: {peft_path}", flush=True)
    peft_config = PeftConfig.from_pretrained(peft_path)
    base_model_name = peft_config.base_model_name_or_path
    
    print(f"Loading base model: {base_model_name}", flush=True)
    # Load base model directly to the specific device instead of using device_map="auto"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print(f"Base model loaded", flush=True)
    
    print(f"Loading PEFT adapter from: {peft_path}", flush=True)
    qa_model = PeftModel.from_pretrained(base_model, peft_path)
    print(f"PEFT model loaded successfully", flush=True)
    
    # Move the entire model to the specific device
    qa_model = qa_model.to(PHI2_DEVICE)
    print(f"Model moved to device: {PHI2_DEVICE}", flush=True)
    
    # Verify all parameters are on the same device
    devices = {param.device for param in qa_model.parameters()}
    print(f"Model parameters are on devices: {devices}", flush=True)
    
    qa_model.eval()
    qa_initialized = True
    print("Phi-2 medical model loaded successfully!", flush=True)

async def initialize_image_analyzer():
    """Initialize the MedGemma vision-language model using HFTOKEN"""
    global image_analyzer, image_tokenizer, image_initialized
    
    if image_initialized:
        return
        
    print(f"Loading MedGemma vision model on {MEDGEMMA_DEVICE}...", flush=True)
    
    # Get HF token from environment like Gemini API key
    HFTOKEN = os.getenv("HFTOKEN", "")
    
    print(f"HFTOKEN status: {'Found' if HFTOKEN else 'Not found'}", flush=True)
    if HFTOKEN:
        print(f"HFTOKEN length: {len(HFTOKEN)}", flush=True)
        print(f"HFTOKEN starts with: {HFTOKEN[:10]}..." if len(HFTOKEN) > 10 else f"HFTOKEN: {HFTOKEN}", flush=True)
    
    try:
        print("Loading MedGemma vision model (like your working code)...", flush=True)
        # Import the correct classes for vision-language model
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        model_id = "google/medgemma-4b-it"  # Use the same model as your working code
        
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared", flush=True)
        
        # Use exactly the same loading approach as your working code
        image_analyzer = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Use auto like your working code
            low_cpu_mem_usage=True
            # Note: Not using token to match your working code exactly
        )
        print("Model loaded successfully", flush=True)
        
        image_tokenizer = AutoProcessor.from_pretrained(model_id)
        print("Processor loaded successfully", flush=True)
        
        print("MedGemma vision model loaded successfully!", flush=True)
        
    except Exception as e:
        print(f"Error loading MedGemma: {e}", flush=True)
        print("Using alternative image analysis setup...", flush=True)
        
        # Fallback to BLIP model which doesn't require authentication
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            print("Loading BLIP model for image captioning...", flush=True)
            image_tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            image_analyzer = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            image_analyzer = image_analyzer.to(MEDGEMMA_DEVICE)
            image_analyzer.eval()
            print("BLIP model loaded successfully!", flush=True)
            
        except Exception as blip_error:
            print(f"BLIP also failed: {blip_error}, using basic fallback...", flush=True)
            # Final fallback - basic image analysis
            image_analyzer = "basic_fallback"
            image_tokenizer = None
    
    image_initialized = True

def initialize_router():
    """Initialize Gemini Flash router"""
    global router_model
    
    # Debug prints to understand environment variable loading
    print(f"Checking GEMINI_API_KEY from environment...", flush=True)
    print(f"GEMINI_API_KEY exists: {bool(GEMINI_API_KEY)}", flush=True)
    print(f"GEMINI_API_KEY length: {len(GEMINI_API_KEY) if GEMINI_API_KEY else 0}", flush=True)
    
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        router_model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Gemini Flash router initialized", flush=True)
    else:
        print("❌ No Gemini API key - using fallback routing", flush=True)
        print("💡 Tip: Make sure .env file is in the correct location and contains GEMINI_API_KEY", flush=True)

def route_request(user_input: str, session_id: str, has_image: bool = False) -> str:
    """Route user requests to appropriate capability using conversation history"""
    # Get or create conversation history for this session
    history = conversation_histories.get(session_id)
    if not history:
        history = ConversationHistory()
        conversation_histories[session_id] = history
    
    # Get recent context
    recent_messages = history.get_context(last_n=3)
    
    # First try to find relevant image based on just the current question
    relevant_image = history.find_relevant_image(user_input)
    
    # If no match found, try using recent conversation context
    if not relevant_image and recent_messages:
        context_text = " ".join(msg.content + " " + msg.response for msg in recent_messages)
        relevant_image = history.find_relevant_image(context_text)
    
    # Determine if this is an image-related question
    has_recent_image = relevant_image is not None
    """Route user requests to appropriate capability"""
    if router_model:
        prompt = (
            f"Analyze the following user request and determine if it requires image analysis "
            f"or general medical knowledge.\n\n"
            f"User request: \"{user_input}\"\n"
            f"Previous image available: {has_recent_image}\n"
            f"Recent conversation context:\n"
            + "\n".join(f"User: {msg.content}\nAssistant: {msg.response}" 
                       for msg in recent_messages) + "\n\n"
            f"Current question: {user_input}\n\n"
            "IMPORTANT ROUTING RULES:\n\n"
            "1. Visual Analysis (image_analysis):\n"
            "   For Images:\n"
            "   - Questions about visual features: 'normal', 'abnormal', 'visible', 'appearance'\n"
            "   - Questions about measurements or dimensions\n"
            "   - Questions about specific anatomical locations or positions\n"
            "   - Questions about image quality or clarity\n"
            "   - Questions using demonstrative pronouns: 'this', 'these', 'that'\n"
            "   - Questions about comparing different parts of the image\n"
            "   - Questions about identifying structures or anomalies\n"
            "   - Questions about measurements, growth, or changes\n"
            "   - Questions containing visual terms: 'see', 'look', 'appears', 'shows'\n"
            "   - Questions about location or orientation\n\n"
            "   For Reports:\n"
            "   - Questions about specific values or readings in the report\n"
            "   - Questions about test results or findings\n"
            "   - Questions about reference ranges or normal values\n"
            "   - Questions about trends or changes in values\n"
            "   - Questions about specific sections or entries\n"
            "   - Questions about interpretations of results\n"
            "   - Questions about abnormal findings or flags\n"
            "   - Questions comparing current vs previous results\n\n"
            "2. General Medical Knowledge (medical_qa):\n"
            "   - Questions about medical conditions without reference to the image\n"
            "   - Questions about treatments, medications, or procedures\n"
            "   - Questions about general health guidelines or recommendations\n"
            "   - Questions about medical terminology or definitions\n"
            "   - Questions about causes, effects, or symptoms without visual reference\n"
            "   - Questions about medical history or statistics\n"
            "   - Questions about prevention or risk factors\n"
            "   - Questions about general developmental stages without reference to the image\n"
            "   - Questions starting with 'what is', 'what are', 'how do', 'why do'\n"
            "   - Questions about medical processes or mechanisms\n\n"
            "CRITICAL DECISION FACTORS:\n"
            "1. Context Awareness: Consider if the question requires visual information to answer\n"
            "2. Temporal Reference: 'this', 'these', 'here' often indicate image reference\n"
            "3. Anatomical Specificity: Mentions of specific locations usually need image\n"
            "4. Visual Assessment Terms: 'normal', 'abnormal', 'visible' usually need image\n"
            "5. Default Behavior: When truly ambiguous, prefer medical_qa\n\n"
            "Examples:\n"
            "- \"Is the heart visible in this view?\" -> image_analysis\n"
            "- \"What are normal fetal measurements at 20 weeks?\" -> medical_qa\n"
            "- \"Are these features normal?\" -> image_analysis\n"
            "- \"What causes birth defects?\" -> medical_qa\n"
            "- \"Can you point out any abnormalities?\" -> image_analysis\n"
            "- \"How does fetal development progress?\" -> medical_qa\n\n"
            "Respond with ONLY ONE of these exact strings: \"medical_qa\" or \"image_analysis\""
        )
        
        try:
            response = router_model.generate_content(prompt)
            result = response.text.strip().lower()
            
            # Force image_analysis if image is provided
            if has_image:
                return "image_analysis"
            
            # Otherwise use model's decision
            if "image" in result or "analysis" in result:
                return "image_analysis"
            return "medical_qa"
            
        except Exception as e:
            print(f"Gemini routing failed: {e}. Using fallback logic.", flush=True)
    
    # Enhanced fallback routing with comprehensive rules
    user_input_lower = user_input.lower()
    
    # 1. If a new image is being provided, it's definitely image analysis
    if has_image and not user_input:
        return "image_analysis"
    
    # 2. Check for definitional/conceptual questions (always medical_qa)
    medical_qa_patterns = [
        # General medical knowledge patterns
        'what is', 'what are', 'define', 'explain', 'tell me about',
        'describe', 'what causes', 'why does', 'how does',
        'symptoms of', 'treatment for', 'how to treat', 'what happens',
        'when should', 'is it normal for', 'how long does',
        # Medical processes
        'treatment', 'procedure', 'medication', 'medicine',
        'therapy', 'surgery', 'prevention', 'risk',
        # General questions
        'how common', 'how often', 'what percentage', 'statistics',
        'research', 'studies show', 'doctors recommend'
    ]
    
    # 3. Check for image analysis patterns
    image_analysis_patterns = [
        # Direct visual references
        'in the image', 'in this scan', 'in the scan', 'in the report',
        'do you see', 'can you see', 'show me', 'point out',
        'visible', 'appears', 'looks like', 'showing',
        # Positions and locations
        'on the left', 'on the right', 'at the top', 'at the bottom',
        'upper', 'lower', 'anterior', 'posterior', 'lateral',
        # Demonstrative references
        'this feature', 'these features', 'this area', 'this region',
        'this part', 'these parts', 'this structure', 'this section',
        # Visual and report assessment
        'normal appearance', 'abnormal', 'measurement', 'reading',
        'size', 'shape', 'position', 'orientation', 'value',
        # General assessment terms
        'structure', 'pattern', 'density', 'contrast', 'range',
        'feature', 'marker', 'finding', 'appearance', 'result',
        # Report-specific terms
        'test result', 'lab value', 'reference range', 'interpretation',
        'trend', 'comparison', 'previous result', 'current value',
        'flag', 'out of range', 'within normal', 'elevated',
        'decreased', 'increased', 'positive', 'negative'
    ]
    
    # Count matches for each category
    medical_qa_score = sum(1 for pattern in medical_qa_patterns if pattern in user_input_lower)
    image_analysis_score = sum(1 for pattern in image_analysis_patterns if pattern in user_input_lower)
    
    # Decision logic
    if medical_qa_score > image_analysis_score:
        return "medical_qa"
    elif image_analysis_score > medical_qa_score:
        return "image_analysis"
    elif image_analysis_score > 0 and has_image:
        # If tied but we have an image and some image-related terms, prefer image analysis
        return "image_analysis"
    else:
        # Default to medical_qa for ambiguous cases
        return "medical_qa"

# Router will be initialized in main() after environment variables are loaded
router_model = None


@mcp.tool()
async def medical_qa(question: str, session_id: str) -> str:
    """Answer medical questions using a fine-tuned Phi-2 model."""
    # Get or create conversation history
    history = conversation_histories.get(session_id)
    if not history:
        history = ConversationHistory()
        conversation_histories[session_id] = history
    print(f"CALLED: medical_qa(question: {question[:50]}...)", flush=True)
    
    if not question:
        return "Please provide a medical question."
    
    try:
        # Initialize model if needed
        await initialize_qa_model()
        
        # Format prompt to match the fine-tuned model's expected format
        prompt = f"<|user|>{question}<|assistant|>"
        
        inputs = qa_tokenizer(prompt, return_tensors="pt").to(PHI2_DEVICE)
        
        with torch.no_grad():
            try:
                print(f"Input prompt: {prompt}", flush=True)
                print(f"Input tensor device: {inputs['input_ids'].device}", flush=True)
                
                # Verify device consistency
                model_devices = {param.device for param in qa_model.parameters()}
                print(f"Model devices: {model_devices}", flush=True)
                
                print(f"Generating response with sampling...", flush=True)
                # Use parameters that match your working implementation
                outputs = qa_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=512,  # Increased max length
                    min_length=50,   # Ensure minimum response length
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=qa_tokenizer.eos_token_id,
                    do_sample=True,
                    repetition_penalty=1.2,  # Prevent repetitive text
                    length_penalty=1.0,      # Encourage complete sentences
                    early_stopping=True      # Stop at natural endpoints
                )
                
                print(f"Generation completed. Output shape: {outputs.shape}", flush=True)
                response = qa_tokenizer.decode(outputs[0], skip_special_tokens=False)
                print(f"Full generated response: {response}", flush=True)
                
                # Extract only the assistant's response (match your working code)
                if "<|assistant|>" in response:
                    answer = response.split("<|assistant|>")[1].split("<|endoftext|>")[0].strip()
                    print(f"Extracted answer: {answer}", flush=True)
                else:
                    # Fallback extraction
                    answer = response.split(prompt)[-1].strip()
                    print(f"Fallback extracted answer: {answer}", flush=True)
                
                # Validate the answer
                if not answer or answer.strip() == "" or answer.count("!") > 10:
                    print(f"Response validation failed. Answer: '{answer}', Length: {len(answer) if answer else 0}, Exclamation count: {answer.count('!') if answer else 0}", flush=True)
                    raise RuntimeError("Invalid response format")
                
                return answer
                
            except Exception as e:
                print(f"Sampling failed with error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                print("Trying greedy decoding...", flush=True)
                try:
                    # Try with greedy decoding as fallback
                    outputs = qa_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=256,
                        do_sample=False,  # Use greedy decoding
                        pad_token_id=qa_tokenizer.eos_token_id
                    )
                    response = qa_tokenizer.decode(outputs[0], skip_special_tokens=False)
                    print(f"Greedy response: {response[:100]}...", flush=True)
                    
                    # Extract only the assistant's response (match your working code)
                    if "<|assistant|>" in response:
                        answer = response.split("<|assistant|>")[1].split("<|endoftext|>")[0].strip()
                    else:
                        answer = response.split(prompt)[-1].strip()
                    
                    if answer and answer.count("!") <= 10:
                        # Store in conversation history
                        history.add_message(question, answer)
                        return answer
                    else:
                        print("Greedy decoding also failed, providing fallback response", flush=True)
                        return "I apologize, but I'm having trouble generating a proper response. Please try rephrasing your question or contact a healthcare professional for medical advice."
                        
                except Exception as greedy_error:
                    print(f"Greedy decoding also failed: {greedy_error}", flush=True)
                    import traceback
                    traceback.print_exc()
                    return "I apologize, but I'm experiencing technical difficulties. Please try again later or consult a healthcare professional for medical advice."
        
    except Exception as e:
        print(f"Complete failure in medical_qa: {e}", flush=True)
        # Provide a basic fallback response for common medical questions
        question_lower = question.lower()
        if "corona" in question_lower or "covid" in question_lower:
            return (
                "If you test positive for COVID-19, here are important precautions:\n\n"
                "1. **Isolate immediately** - Stay home and away from others for at least 5 days\n"
                "2. **Monitor symptoms** - Watch for fever, cough, difficulty breathing\n"
                "3. **Contact healthcare provider** - Call your doctor for guidance\n"
                "4. **Rest and hydrate** - Get plenty of rest and drink fluids\n"
                "5. **Wear mask** - If you must be around others, wear a well-fitted mask\n"
                "6. **Clean surfaces** - Regularly disinfect high-touch surfaces\n"
                "7. **Seek emergency care** - If you have severe symptoms like difficulty breathing\n\n"
                "Please consult with a healthcare professional for personalized medical advice."
            )
        else:
            return "I apologize, but I'm experiencing technical difficulties. Please consult a healthcare professional for medical advice."

@mcp.tool()
async def medical_image_analysis(image_data: str, description: str = "") -> str:
    """Analyze medical images using MedGemma model."""
    print(f"CALLED: medical_image_analysis(description: {description[:30]}...)", flush=True)
    
    if not image_data:
        return "Please provide image data."
    
    try:
        # Initialize model if needed
        await initialize_image_analyzer()
        
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Image loaded: {image.size}, mode: {image.mode}", flush=True)
        
        # Debug: Check what type of analyzer we have
        print(f"Image analyzer type: {type(image_analyzer)}", flush=True)
        print(f"Image tokenizer type: {type(image_tokenizer)}", flush=True)
        
        if image_analyzer == "basic_fallback":
            # Basic fallback analysis
            analysis = (
                f"Basic Image Analysis:\n"
                f"This appears to be a medical image with dimensions {image.size[0]}x{image.size[1]}.\n\n"
                f"General observations that can be made from the image structure:\n"
                f"- Image format: {image.mode}\n"
                f"- Image size: {image.size}\n\n"
                f"Note: For proper medical image analysis, please consult with a qualified "
                f"radiologist or medical professional. This system cannot provide diagnostic information."
            )

        elif hasattr(image_analyzer, 'generate') and hasattr(image_tokenizer, 'apply_chat_template'):  # MedGemma vision model
            print("Using MedGemma vision model for image analysis...", flush=True)
            
            # Use the same approach as your working code
            prompt = f"Describe this medical image in detail. {description}" if description else "Describe this medical image in detail"
            
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Process inputs exactly like your working code
            inputs = image_tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(image_analyzer.device, dtype=torch.bfloat16)  # Use model.device like your working code
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = image_analyzer.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    num_beams=1
                )
                generation = generation[0][input_len:]
            
            analysis = image_tokenizer.decode(generation, skip_special_tokens=True)
            
            # Clean up memory like your working code
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        elif hasattr(image_analyzer, 'generate'):  # BLIP model
            print("Using BLIP model for image analysis...", flush=True)
            
            # BLIP expects different input format
            inputs = image_tokenizer(image, return_tensors="pt").to(MEDGEMMA_DEVICE)
            
            with torch.no_grad():
                outputs = image_analyzer.generate(**inputs, max_length=50, num_beams=5)
            
            caption = image_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            analysis = (
                f"Medical Image Analysis using BLIP model:\n\n"
                f"Image Caption: {caption}\n\n"
                f"Description provided: {description if description else 'None'}\n\n"
                f"Medical Context: This appears to be a medical image. While I can provide a basic "
                f"description, please note that proper medical image interpretation requires expertise "
                f"from qualified medical professionals such as radiologists or specialists.\n\n"
                f"Important: This analysis is for informational purposes only and should not be used "
                f"for diagnostic decisions. Always consult with healthcare professionals for medical advice."
            )
            
        else:
            analysis = "Unable to analyze image with current setup. Please ensure proper model configuration."
        
        return analysis
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

@mcp.tool()
async def medical_image_question(image_data: str, question: str, session_id: str) -> str:
    """Answer specific questions about medical images using MedGemma VLM."""
    # Get or create conversation history
    history = conversation_histories.get(session_id)
    if not history:
        history = ConversationHistory()
        conversation_histories[session_id] = history
    print(f"CALLED: medical_image_question(question: {question[:50]}...)", flush=True)
    
    if not image_data:
        return "Please provide image data."
    
    if not question:
        return "Please provide a question about the image."
    
    try:
        # Initialize model if needed
        await initialize_image_analyzer()
        
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Image loaded for question: {image.size}, mode: {image.mode}", flush=True)
        
        # Debug: Check what type of analyzer we have
        print(f"Image analyzer type: {type(image_analyzer)}", flush=True)
        
        if image_analyzer == "basic_fallback":
            return (
                f"I apologize, but the vision model is not available for answering questions about images.\n\n"
                f"Your question: \"{question}\"\n\n"
                f"Basic image info:\n"
                f"- Dimensions: {image.size[0]}x{image.size[1]}\n"
                f"- Format: {image.mode}\n\n"
                f"Please ensure the MedGemma vision model is properly loaded to answer specific questions about medical images."
            )

        elif hasattr(image_analyzer, 'generate') and hasattr(image_tokenizer, 'apply_chat_template'):  # MedGemma vision model
            print("Using MedGemma vision model for image question...", flush=True)
            
            # Use the user's custom question
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist. Answer the specific question about this medical image."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Process inputs exactly like working code
            inputs = image_tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(image_analyzer.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = image_analyzer.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    num_beams=1
                )
                generation = generation[0][input_len:]
            
            analysis = image_tokenizer.decode(generation, skip_special_tokens=True)
            
            # Clean up memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Find the most relevant image for this question
            relevant_image = history.find_relevant_image(question)
            if relevant_image:
                # Update current image if a more relevant one was found
                history.current_image_idx = history.images.index(relevant_image)
            
            # Store in conversation history with image flag
            history.add_message(question, analysis, has_image=True)
            return analysis
            
        else:
            return f"Vision model not properly configured to answer questions about images. Question: {question}"
        
    except Exception as e:
        print(f"Error in medical_image_question: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return f"Error answering question about image: {str(e)}. Question was: {question}"


@mcp.tool()
def end_session(session_id: str) -> str:
    """End a conversation session and clear its history."""
    if session_id in conversation_histories:
        conversation_histories[session_id].clear()
        del conversation_histories[session_id]
        return "Session ended and history cleared."
    return "Session not found."

@mcp.tool()
def route_query(user_input: str, session_id: str, has_image: bool = False) -> str:
    """Route user queries to appropriate medical capability."""
    print(f"CALLED: route_query(user_input: {user_input[:30]}..., session_id: {session_id}, has_image: {has_image})", flush=True)
    
    if not user_input:
        return "Please provide user input to route."
        
    if not session_id:
        return "Please provide a session ID."
    
    try:
        capability = route_request(user_input, session_id, has_image)
        return f"Recommended capability: {capability}"
    except Exception as e:
        return f"Error routing query: {str(e)}"


if __name__ == "__main__":
    print("STARTING Medical MCP Server", flush=True)
    
    # Initialize router after environment variables are loaded
    initialize_router()
    
    print("Available capabilities:", flush=True)
    print("1. medical_qa - Answer medical questions using fine-tuned Phi-2", flush=True)
    print("2. medical_image_analysis - Analyze medical images using MedGemma", flush=True)
    print("3. route_query - Route queries using Gemini Flash", flush=True)
    
    # Check if running with mcp dev command
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run()  # Run without transport for dev server
    else:
        mcp.run(transport="stdio")  # Run with stdio for direct execution
