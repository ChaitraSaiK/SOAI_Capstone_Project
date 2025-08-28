#!/usr/bin/env python3
"""
Medical MCP Client
Connects to the Medical MCP Server and provides an interface for medical Q&A and image analysis.
"""

import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

from PIL import Image
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MedicalMCPClient:
    """Client for interacting with the Medical MCP Server"""
    
    def __init__(self, server_script_path: str = "./medical_mcp_server.py"):
        self.server_script_path = server_script_path
        self.session: Optional[ClientSession] = None
        self.stdio_context = None
        self.session_context = None
        # Store image history
        self.image_history: List[Dict[str, str]] = []  # List of {'path': str, 'data': str, 'description': str}
        # Generate a unique session ID
        import uuid
        self.session_id = str(uuid.uuid4())
    
    async def connect(self):
        """Connect to the Medical MCP Server"""
        try:
            print("Establishing connection to MCP server...")
            # Start the server process and connect via stdio
            server_params = StdioServerParameters(
                command="python",
                args=[self.server_script_path]
            )
            
            # Use the proper context manager pattern
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()
            
            print("Connection established, creating session...")
            self.session_context = ClientSession(read, write)
            self.session = await self.session_context.__aenter__()
            
            print("Session created, initializing...")
            await self.session.initialize()
            
            print("✅ Connected to Medical MCP Server")
            print("Available tools:")
            
            # List available tools
            tools_result = await self.session.list_tools()
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def ask_medical_question(self, question: str) -> str:
        """Ask a medical question"""
        if not self.session:
            return "Error: Not connected to server"
        
        try:
            result = await self.session.call_tool(
                "medical_qa",
                {
                    "question": question,
                    "session_id": self.session_id
                }
            )
            
            if result.content:
                return result.content[0].text
            else:
                return "No response received"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def analyze_medical_image(self, image_path: str, description: str = "") -> str:
        """Analyze a medical image"""
        if not self.session:
            return "Error: Not connected to server"
        
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Store in image history
            self.image_history.append({
                'path': image_path,
                'data': image_b64,
                'description': ''  # Will be updated with the response
            })
            
            result = await self.session.call_tool(
                "medical_image_analysis",
                {
                    "image_data": image_b64,
                    "description": description,
                    "session_id": self.session_id
                }
            )
            
            if result.content:
                response = result.content[0].text
                # Update the description in image history
                if self.image_history:
                    self.image_history[-1]['description'] = response
                return response
            else:
                return "No response received"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def ask_image_question(self, image_path: Optional[str], question: str) -> str:
        """Ask a specific question about a medical image using VLM"""
        if not self.session:
            return "Error: Not connected to server"
        
        try:
            # If no image path provided, find relevant image from history
            if image_path is None:
                relevant_image = self.find_image_for_query(question)
                if not relevant_image:
                    return "Error: No relevant image found for this question"
                image_b64 = relevant_image['data']
                print(f"Using image: {os.path.basename(relevant_image['path'])}")
            else:
                # Read and encode new image
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                # Store in image history
                self.image_history.append({
                    'path': image_path,
                    'data': image_b64,
                    'description': ''  # Will be updated with the response
                })
            
            result = await self.session.call_tool(
                "medical_image_question",
                {
                    "image_data": image_b64,
                    "question": question,
                    "session_id": self.session_id
                }
            )
            
            if result.content:
                return result.content[0].text
            else:
                return "No response received"
                
        except Exception as e:
            return f"Error: {str(e)}"

    async def route_query(self, user_input: str, has_image: bool = False) -> str:
        """Route a user query to determine the appropriate capability"""
        if not self.session:
            return "Error: Not connected to server"
        
        try:
            result = await self.session.call_tool(
                "route_query",
                {
                    "user_input": user_input,
                    "has_image": has_image,
                    "session_id": self.session_id
                }
            )
            
            if result.content:
                return result.content[0].text
            else:
                return "No response received"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def find_image_for_query(self, query: str) -> Optional[Dict[str, str]]:
        """Find the most relevant image for a given query based on context matching"""
        if not self.image_history:
            return None
            
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # Score each image based on semantic overlap with the query
        best_score = 0
        best_match = None
        
        for img in self.image_history:
            desc_lower = img['description'].lower()
            desc_terms = set(desc_lower.split())
            
            # Calculate overlap between query terms and description
            matching_terms = query_terms.intersection(desc_terms)
            
            # Basic score based on term overlap
            score = len(matching_terms)
            
            # Boost score if key terms from description appear in query
            # This helps identify when a question is specifically about this image
            key_terms = set(term for term in desc_terms 
                          if len(term) > 3 and not term.lower() in 
                          {'this', 'that', 'with', 'from', 'have', 'shows', 'image'})
            if any(term in query_lower for term in key_terms):
                score += 2
                
            if score > best_score:
                best_score = score
                best_match = img
        
        # Only return if we found a meaningful match
        # Require at least 2 matching terms for confidence
        return best_match if best_score >= 2 else None

    async def disconnect(self):
        """Disconnect from the server"""
        try:
            # Clean up session history
            if self.session:
                try:
                    await self.session.call_tool(
                        "end_session",
                        {"session_id": self.session_id}
                    )
                except Exception as e:
                    print(f"Warning: Error clearing session history: {e}")

            if self.session_context:
                await self.session_context.__aexit__(None, None, None)
                self.session_context = None
                
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
                self.stdio_context = None
                
            self.session = None
            print("🔌 Disconnected from Medical MCP Server")
        except Exception as e:
            print(f"Warning: Error during disconnect: {e}")


async def interactive_session():
    """Run an interactive session with the Medical MCP Client"""
    client = MedicalMCPClient()
    
    print("🏥 Medical MCP Assistant")
    print("====================")
    print("Connecting to Medical MCP Server...")
    
    if not await client.connect():
        print("Failed to connect. Exiting.")
        return
    
    print("\n💡 You can:")
    print("  • Ask any medical questions")
    print("  • Share medical images for analysis")
    print("  • Ask follow-up questions about previous images")
    print("  • Type 'quit' to exit")
    print("\nI'll automatically determine whether to use general medical knowledge or image analysis.")
    print()
    
    try:
        while True:
            user_input = input("👤 ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                break
            
            # Check if the input contains a file path that might be an image
            words = user_input.split()
            potential_image_path = None
            
            for word in words:
                if any(word.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']):
                    if os.path.exists(word):
                        potential_image_path = word
                        # Remove the image path from the question
                        user_input = user_input.replace(word, '').strip()
                        break
            
            if potential_image_path:
                # If an image path is provided, ALWAYS use image analysis
                print("🖼️ Analyzing new image...")
                if user_input.strip():  # If there's a question along with the image
                    response = await client.ask_image_question(potential_image_path, user_input)
                else:  # If only image path provided
                    response = await client.analyze_medical_image(potential_image_path, "")
                print(f"🔍 {response}")
            else:
                # Route the query using the server's routing logic
                print("💭 Analyzing your question...")
                route_response = await client.route_query(user_input, bool(client.image_history))
                
                # First try to find a relevant image for the question
                relevant_image = client.find_image_for_query(user_input)
                
                if relevant_image:
                    # If we found a relevant image, use image analysis
                    print(f"🖼️ Analyzing with image: {os.path.basename(relevant_image['path'])}")
                    response = await client.ask_image_question(None, user_input)
                    print(f"🔍 {response}")
                else:
                    # If no relevant image found, use medical QA
                    print("🩺 Consulting medical knowledge...")
                    response = await client.ask_medical_question(user_input)
                    print(f"💡 {response}")
            
            print()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    finally:
        await client.disconnect()


async def test_client():
    """Test the client with sample queries"""
    client = MedicalMCPClient()
    
    print("🧪 Testing Medical MCP Client")
    print("=============================")
    
    if not await client.connect():
        print("Failed to connect. Exiting.")
        return
    
    # Test medical question
    print("\n1. Testing medical question...")
    question = "What are the common symptoms of diabetes?"
    response = await client.ask_medical_question(question)
    print(f"Q: {question}")
    print(f"A: {response}")
    
    # Test routing
    print("\n2. Testing query routing...")
    queries = [
        "I have a headache and fever",
        "What's the weather like today?",
        "How do I treat a broken bone?"
    ]
    
    for query in queries:
        response = await client.route_query(query)
        print(f"Query: {query}")
        print(f"Route: {response}")
    
    await client.disconnect()
    print("\n✅ Tests completed")


def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_client())
    else:
        asyncio.run(interactive_session())


if __name__ == "__main__":
    main() 
