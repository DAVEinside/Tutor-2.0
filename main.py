from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from gpt4o_math_service import get_gpt4o_service
from PIL import Image
import io
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GPT-4o Math Tutor",
    description="Real-time mathematical analysis using GPT-4o",
    version="2.0.0"
)

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "GPT-4o Math Tutor"}

@app.post("/analyze-math")
async def analyze_math_image(file: UploadFile = File(...)):
    """Analyze mathematical content in uploaded image using GPT-4o"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and convert image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get GPT-4o service
        gpt4o_service = get_gpt4o_service()
        
        # Analyze the math image
        result = gpt4o_service.analyze_math_image(image)
        
        if result["success"]:
            return {
                "success": True,
                "filename": file.filename,
                "analysis": result
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Analysis failed"),
                "filename": file.filename
            }
            
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/extract-text")
async def extract_math_text(file: UploadFile = File(...)):
    """Extract mathematical text from image using GPT-4o"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and convert image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get GPT-4o service
        gpt4o_service = get_gpt4o_service()
        
        # Extract text
        result = gpt4o_service.extract_text_only(image)
        
        return result
            
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return {
            "success": False,
            "error": str(e)
        }

class TextAnalysisRequest(BaseModel):
    text: str

class FollowupQuestionRequest(BaseModel):
    original_problem: str
    question: str

@app.post("/analyze-text")
async def analyze_math_text(request: TextAnalysisRequest):
    """Analyze mathematical text directly (for manual input)"""
    try:
        # For now, return a simple analysis
        # In production, you could use GPT-4 (text) for this
        text = request.text.strip()
        
        # Basic evaluation
        is_equation = '=' in text
        has_operators = any(op in text for op in ['+', '-', '*', '/', '^'])
        
        return {
            "success": True,
            "text": text,
            "is_equation": is_equation,
            "has_operators": has_operators,
            "analysis": "Text analysis endpoint - use image analysis for full GPT-4V evaluation"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/ask-followup")
async def ask_followup_question(request: FollowupQuestionRequest):
    """Ask a follow-up question about a math problem"""
    try:
        # Get GPT-4o service
        gpt4o_service = get_gpt4o_service()
        
        # Ask follow-up question
        result = gpt4o_service.ask_followup_question(
            original_problem=request.original_problem,
            question=request.question
        )
        
        return result
            
    except Exception as e:
        logger.error(f"Error processing follow-up question: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)