import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import requests
import os
import logging
from typing import Dict, Any, Optional, Tuple, List
import re
from skimage import restoration, filters
import pytesseract

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Advanced image preprocessing for OCR optimization"""
    
    def __init__(self):
        self.target_height = 384  # Optimal for TrOCR
        
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply comprehensive image enhancement"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 1. Noise reduction
            cv_image = cv2.bilateralFilter(cv_image, 9, 75, 75)
            
            # 2. Contrast enhancement using CLAHE
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            cv_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 3. Sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            cv_image = cv2.filter2D(cv_image, -1, kernel)
            
            # Convert back to PIL
            enhanced = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}, returning original")
            return image
    
    def binarize_image(self, image: Image.Image) -> Image.Image:
        """Advanced binarization for better text contrast"""
        try:
            # Convert to grayscale
            gray = image.convert('L')
            cv_gray = np.array(gray)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                cv_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return Image.fromarray(binary)
            
        except Exception as e:
            logger.warning(f"Binarization failed: {e}")
            return image.convert('L')
    
    def detect_and_crop_text_region(self, image: Image.Image) -> Image.Image:
        """Detect and crop to text region"""
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get bounding box of largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.width - x, w + 2 * padding)
                h = min(image.height - y, h + 2 * padding)
                
                # Crop
                cropped = image.crop((x, y, x + w, y + h))
                
                # Only return cropped if it's reasonable size
                if cropped.width > 50 and cropped.height > 20:
                    return cropped
            
            return image
            
        except Exception as e:
            logger.warning(f"Text region detection failed: {e}")
            return image
    
    def resize_for_ocr(self, image: Image.Image, target_height: int = None) -> Image.Image:
        """Resize image to optimal dimensions for OCR"""
        if target_height is None:
            target_height = self.target_height
            
        try:
            # Calculate new dimensions maintaining aspect ratio
            aspect_ratio = image.width / image.height
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
            
            # Ensure minimum width
            if new_width < 100:
                new_width = 100
                new_height = int(100 / aspect_ratio)
            
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        except Exception as e:
            logger.warning(f"Resize failed: {e}")
            return image
    
    def preprocess_pipeline(self, image: Image.Image, for_model: str = "trocr") -> Image.Image:
        """Complete preprocessing pipeline"""
        try:
            # Step 1: Basic enhancement
            enhanced = self.enhance_image(image)
            
            # Step 2: Text region detection and cropping
            cropped = self.detect_and_crop_text_region(enhanced)
            
            # Step 3: Model-specific preprocessing
            if for_model == "trocr":
                # TrOCR works better with enhanced color images
                processed = self.resize_for_ocr(cropped)
            elif for_model == "easyocr":
                # EasyOCR works well with both color and binary
                processed = self.resize_for_ocr(cropped, target_height=480)
            elif for_model == "tesseract":
                # Tesseract often works better with binary images
                binary = self.binarize_image(cropped)
                processed = self.resize_for_ocr(binary, target_height=300)
            else:
                processed = self.resize_for_ocr(cropped)
            
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            return image

class MathpixOCR:
    """Mathpix API integration for specialized math OCR"""
    
    def __init__(self):
        self.api_key = os.getenv('MATHPIX_API_KEY')
        self.app_id = os.getenv('MATHPIX_APP_ID')
        self.enabled = bool(self.api_key and self.app_id)
        
        if not self.enabled:
            logger.warning("Mathpix API credentials not found. Math-specific OCR disabled.")
    
    def extract_text(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Extract text using Mathpix API"""
        if not self.enabled:
            return None
            
        try:
            # Convert image to base64
            import io
            import base64
            
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            # API request
            response = requests.post(
                "https://api.mathpix.com/v3/text",
                files={"file": img_data},
                data={
                    "options_json": '{"math_inline_delimiters": ["$", "$"], "rm_spaces": true}'
                },
                headers={
                    "app_id": self.app_id,
                    "app_key": self.api_key
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", 0.0),
                    "latex": result.get("latex", ""),
                    "source": "mathpix"
                }
            else:
                logger.warning(f"Mathpix API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Mathpix OCR failed: {e}")
            return None

class EnhancedOCRService:
    """Enhanced OCR service with multiple models and preprocessing"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocessor = ImagePreprocessor()
        self.mathpix = MathpixOCR()
        
        # Initialize models
        self._load_trocr_large()
        self._load_easyocr()
        self._init_math_symbols()
        
        logger.info(f"Enhanced OCR service initialized on {self.device}")
    
    def _load_trocr_large(self):
        """Load TrOCR Large model"""
        try:
            logger.info("Loading TrOCR Large model...")
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
            self.trocr_model.to(self.device)
            logger.info("✅ TrOCR Large loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TrOCR Large: {e}")
            # Fallback to base model
            try:
                self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
                self.trocr_model.to(self.device)
                logger.info("✅ TrOCR Base loaded as fallback")
            except Exception as e2:
                logger.error(f"Failed to load any TrOCR model: {e2}")
                self.trocr_processor = None
                self.trocr_model = None
    
    def _load_easyocr(self):
        """Load EasyOCR"""
        try:
            logger.info("Loading EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            logger.info("✅ EasyOCR loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            self.easyocr_reader = None
    
    def _init_math_symbols(self):
        """Initialize math symbol replacement dictionary"""
        self.math_symbol_map = {
            # Common OCR misreads for math symbols
            'x': '*',  # multiplication
            'X': '*',
            '×': '*',
            '÷': '/',
            '∫': 'integral',
            '∑': 'sum',
            '∆': 'delta',
            '√': 'sqrt',
            '±': '+/-',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '∞': 'infinity',
            'π': 'pi',
            'θ': 'theta',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            '²': '^2',
            '³': '^3',
            '⁴': '^4',
            '⁰': '^0',
            '¹': '^1',
            # Common OCR errors
            'O': '0',  # Letter O to zero
            'l': '1',  # Letter l to one
            'I': '1',  # Letter I to one
            'S': '5',  # Letter S to five (in certain contexts)
            'o': '0',  # Letter o to zero
        }
    
    def extract_with_trocr(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Extract text using TrOCR"""
        if not self.trocr_model:
            return None
            
        try:
            # Preprocess for TrOCR
            processed_image = self.preprocessor.preprocess_pipeline(image, "trocr")
            
            # Generate text
            pixel_values = self.trocr_processor(images=processed_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            generated_ids = self.trocr_model.generate(pixel_values)
            text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Calculate confidence (simplified)
            confidence = 0.8 if len(text.strip()) > 0 else 0.1
            
            return {
                "text": text.strip(),
                "confidence": confidence,
                "source": "trocr_large"
            }
            
        except Exception as e:
            logger.error(f"TrOCR extraction failed: {e}")
            return None
    
    def extract_with_easyocr(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Extract text using EasyOCR"""
        if not self.easyocr_reader:
            return None
            
        try:
            # Preprocess for EasyOCR
            processed_image = self.preprocessor.preprocess_pipeline(image, "easyocr")
            
            # Convert to numpy array
            cv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
            
            # Extract text
            results = self.easyocr_reader.readtext(cv_image)
            
            if results:
                # Combine all detected text
                texts = []
                confidences = []
                
                for (bbox, text, confidence) in results:
                    texts.append(text)
                    confidences.append(confidence)
                
                combined_text = " ".join(texts)
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                return {
                    "text": combined_text.strip(),
                    "confidence": avg_confidence,
                    "source": "easyocr"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return None
    
    def extract_with_tesseract(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Extract text using Tesseract (fallback)"""
        try:
            # Preprocess for Tesseract
            processed_image = self.preprocessor.preprocess_pipeline(image, "tesseract")
            
            # Extract text
            text = pytesseract.image_to_string(processed_image)
            
            if text.strip():
                return {
                    "text": text.strip(),
                    "confidence": 0.6,  # Conservative confidence for Tesseract
                    "source": "tesseract"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return None
    
    def normalize_math_expression(self, text: str) -> str:
        """Normalize mathematical expressions"""
        try:
            # Apply symbol replacements
            normalized = text
            for symbol, replacement in self.math_symbol_map.items():
                normalized = normalized.replace(symbol, replacement)
            
            # Clean whitespace
            normalized = " ".join(normalized.split())
            
            # Fix common patterns
            normalized = re.sub(r'(\d)\s*\*\s*(\d)', r'\1*\2', normalized)  # Fix spacing in multiplication
            normalized = re.sub(r'(\d)\s*\+\s*(\d)', r'\1+\2', normalized)  # Fix spacing in addition
            normalized = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', normalized)   # Fix spacing in subtraction
            normalized = re.sub(r'(\d)\s*/\s*(\d)', r'\1/\2', normalized)   # Fix spacing in division
            normalized = re.sub(r'(\d)\s*=\s*(\d)', r'\1=\2', normalized)   # Fix spacing around equals
            
            return normalized
            
        except Exception as e:
            logger.error(f"Math normalization failed: {e}")
            return text
    
    def extract_text_ensemble(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text using ensemble of OCR methods"""
        results = []
        
        # Try Mathpix first (best for math)
        mathpix_result = self.mathpix.extract_text(image)
        if mathpix_result and mathpix_result["confidence"] > 0.8:
            results.append(mathpix_result)
        
        # Try TrOCR Large
        trocr_result = self.extract_with_trocr(image)
        if trocr_result:
            results.append(trocr_result)
        
        # Try EasyOCR
        easyocr_result = self.extract_with_easyocr(image)
        if easyocr_result:
            results.append(easyocr_result)
        
        # Try Tesseract as last resort
        if not results:
            tesseract_result = self.extract_with_tesseract(image)
            if tesseract_result:
                results.append(tesseract_result)
        
        # Select best result
        if results:
            # Sort by confidence
            best_result = max(results, key=lambda x: x["confidence"])
            
            # Normalize the text
            normalized_text = self.normalize_math_expression(best_result["text"])
            
            return {
                "text": best_result["text"],
                "normalized_text": normalized_text,
                "confidence": best_result["confidence"],
                "source": best_result["source"],
                "all_results": results,
                "success": True
            }
        
        return {
            "text": "",
            "normalized_text": "",
            "confidence": 0.0,
            "source": "none",
            "all_results": [],
            "success": False,
            "error": "All OCR methods failed"
        }

# Global enhanced OCR service instance
enhanced_ocr_service = None

def get_enhanced_ocr_service():
    global enhanced_ocr_service
    if enhanced_ocr_service is None:
        enhanced_ocr_service = EnhancedOCRService()
    return enhanced_ocr_service