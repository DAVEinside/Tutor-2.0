import base64
import io
import os
import logging
from typing import Dict, Any, Optional
from PIL import Image
import openai
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file if present

logger = logging.getLogger(__name__)

class GPT4OMathService:
    """GPT-4o service for math problem analysis"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info("✅ GPT-4o client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            image_bytes = buffer.getvalue()
            
            # Encode to base64
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from GPT-4o response, handling markdown code blocks"""
        if not self.client:
            return None
            
        try:
            # First try direct JSON parsing
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, use GPT-4o to extract the JSON
            try:
                parse_response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user", 
                            "content": f"""Extract the JSON data from this response and return only valid JSON:

{content}

Return only the JSON object, no markdown formatting or extra text."""
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                
                parsed_content = parse_response.choices[0].message.content.strip()
                
                # Remove any markdown code block markers
                if parsed_content.startswith('```'):
                    lines = parsed_content.split('\n')
                    parsed_content = '\n'.join(lines[1:-1])
                
                return json.loads(parsed_content)
                
            except Exception as e:
                logger.error(f"JSON parsing failed: {e}")
                return None

    def analyze_math_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze math problem image using GPT-4o"""
        if not self.client:
            return {
                "success": False,
                "error": "GPT-4o client not initialized. Please set OPENAI_API_KEY environment variable."
            }
        
        try:
            # Encode image
            base64_image = self._encode_image(image)
            if not base64_image:
                return {"success": False, "error": "Failed to encode image"}
            
            # Create the prompt for math analysis
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are an expert math tutor analyzing a handwritten mathematical solution. 

Please:
1. Extract the mathematical expression/equation from the image
2. Evaluate if the solution is correct
3. Identify any errors
4. Provide step-by-step correction if needed
5. Give encouraging feedback

Respond in JSON format:
{
    "extracted_math": "the mathematical expression you see",
    "is_correct": true/false,
    "errors": ["list of any errors found"],
    "correct_solution": "step-by-step correct solution",
    "explanation": "detailed explanation of the math",
    "feedback": "encouraging feedback for the student",
    "confidence": 0.95
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Call GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Parse JSON using enhanced parser
            parsed_result = self._parse_json_response(content)
            
            if parsed_result:
                parsed_result["success"] = True
                parsed_result["raw_response"] = content
                return parsed_result
            else:
                # Fallback if parsing completely fails
                return {
                    "success": True,
                    "extracted_math": "Could not parse structured response",
                    "raw_response": content,
                    "is_correct": None,
                    "errors": [],
                    "correct_solution": content,
                    "explanation": content,
                    "feedback": "Analysis provided but could not parse format",
                    "confidence": 0.8
                }
                
        except Exception as e:
            logger.error(f"GPT-4o analysis failed: {e}")
            return {
                "success": False,
                "error": f"GPT-4o analysis failed: {str(e)}"
            }
    
    def extract_text_only(self, image: Image.Image) -> Dict[str, Any]:
        """Extract just the mathematical text from image"""
        if not self.client:
            return {
                "success": False,
                "error": "GPT-4o client not initialized"
            }
        
        try:
            base64_image = self._encode_image(image)
            if not base64_image:
                return {"success": False, "error": "Failed to encode image"}
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the mathematical expression/equation from this image. Return only the math text, exactly as written."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=200,
                temperature=0.1
            )
            
            extracted_text = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "confidence": 0.95,
                "source": "gpt4o"
            }
            
        except Exception as e:
            logger.error(f"GPT-4o text extraction failed: {e}")
            return {
                "success": False,
                "error": f"Text extraction failed: {str(e)}"
            }
    
    def ask_followup_question(self, original_problem: str, question: str) -> Dict[str, Any]:
        """Ask a follow-up question about a math problem"""
        if not self.client:
            return {
                "success": False,
                "error": "GPT-4o client not initialized"
            }
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""I have a math problem: {original_problem}

My follow-up question is: {question}

Please provide a clear, educational answer that helps me understand the concept better."""
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=800,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "original_problem": original_problem
            }
            
        except Exception as e:
            logger.error(f"Follow-up question failed: {e}")
            return {
                "success": False,
                "error": f"Follow-up question failed: {str(e)}"
            }

# Global service instance
gpt4o_service = None

def get_gpt4o_service():
    global gpt4o_service
    if gpt4o_service is None:
        gpt4o_service = GPT4OMathService()
    return gpt4o_service