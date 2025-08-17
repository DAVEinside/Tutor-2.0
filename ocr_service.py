from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import logging
import io

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info("Loading TrOCR model for handwritten text recognition...")
            # Using Microsoft's TrOCR model fine-tuned for handwritten text
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            self.model.to(self.device)
            logger.info(f"TrOCR model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load TrOCR model: {e}")
            raise

    def extract_text(self, image_data: bytes) -> str:
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            logger.info(f"OCR extracted text: {generated_text}")
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return f"OCR Error: {str(e)}"

    def preprocess_for_math(self, text: str) -> str:
        """
        Clean and prepare OCR text for mathematical evaluation
        """
        # Basic cleaning - can be enhanced
        text = text.replace("×", "*")
        text = text.replace("÷", "/")
        text = text.replace("²", "^2")
        text = text.replace("³", "^3")
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text

# Global OCR service instance
ocr_service = None

def get_ocr_service():
    global ocr_service
    if ocr_service is None:
        ocr_service = OCRService()
    return ocr_service