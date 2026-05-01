"""
Image Quality Inspector
Responsible for checking image quality (sharpness, lighting) and content validation.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple, Optional

class QualityInspector:
    
    @staticmethod
    def check_sharpness(img_pil: Image.Image, threshold: float = 7.0) -> Tuple[bool, float, str]:
        """
        Check if image is blurry using Laplacian variance.
        
        Args:
            img_pil: PIL Image
            threshold: Variance threshold (default 100.0)
            
        Returns:
            (passed, score, message)
        """
        # Convert to grayscale numpy array
        img_np = np.array(img_pil.convert('RGB'))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Calculate Laplacian variance
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        passed = variance > threshold
        message = "Image is sharp enough" if passed else "Image is too blurry"
        
        return passed, variance, message

    @staticmethod
    def check_exposure(img_pil: Image.Image) -> Tuple[bool, float, str]:
        """
        Check if image is too dark or overexposed.
        
        Args:
            img_pil: PIL Image
            
        Returns:
            (passed, mean_intensity, message)
        """
        # Convert to grayscale
        img_np = np.array(img_pil.convert('L'))
        mean_intensity = np.mean(img_np)
        
        if mean_intensity < 20:
            return False, mean_intensity, "Image is too dark (underexposed)"
        elif mean_intensity > 235:
            return False, mean_intensity, "Image is too bright (overexposed)"
            
        return True, mean_intensity, "Lighting is good"

    @staticmethod
    def detect_unexpected_text(ocr_reader, img_pil: Image.Image) -> Tuple[bool, int, str]:
        """
        Check if a 'fabric' image contains too much text (likely a label/screenshot).
        
        Args:
            ocr_reader: EasyOCR reader instance
            img_pil: PIL Image
            
        Returns:
            (passed, text_area_count, message)
        """
        if ocr_reader is None:
            return True, 0, "OCR not available, skipping check"

        try:
            img_np = np.array(img_pil)
            # detect=True returns bounding boxes without full recognition (faster)
            # But readtext is easier to interface with depending on cache structure.
            # We assume ocr_reader is the EasyOCR Reader object.
            
            # Use detail=0 to just get text list if possible, but we need boxes to estimate area?
            # Standard readtext returns (bbox, text, prob).
            results = ocr_reader.readtext(img_np)
            
            # Heuristic: If we find > 3 separate text blocks, or one very long text, it might be a document
            # But fabrics can have text prints.
            # Stricter heuristic: Text confidence high AND multiple lines.
            
            text_blocks = [res for res in results if res[2] > 0.5] # filter low conf
            count = len(text_blocks)
            
            if count > 5:
                return False, count, f"Detected {count} text blocks. Is this a document/label?"
                
            return True, count, "No significant text detected"
            
        except Exception as e:
            print(f"Text detection failed: {e}")
            return True, 0, "Text detection error"

    @classmethod
    def inspect_image(cls, img_pil: Image.Image, ocr_reader=None) -> Dict:
        """
        Run all quality checks on an image.
        """
        sharp_pass, sharp_score, sharp_msg = cls.check_sharpness(img_pil)
        exp_pass, exp_score, exp_msg = cls.check_exposure(img_pil)
        
        # Text detection is optional and heavy, maybe skip for now or only if requested
        # We will run it if ocr_reader is provided
        text_pass, text_count, text_msg = True, 0, "Skipped"
        if ocr_reader:
            text_pass, text_count, text_msg = cls.detect_unexpected_text(ocr_reader, img_pil)
            
        passed_all = sharp_pass and exp_pass and text_pass
        
        issues = []
        if not sharp_pass: issues.append(f"Blurry (Score: {sharp_score:.1f})")
        if not exp_pass: issues.append(f"Bad Lighting (Score: {exp_score:.1f})")
        if not text_pass: issues.append(f"Contains Text (Count: {text_count})")
        
        return {
            "passed": passed_all,
            "issues": issues,
            "details": {
                "sharpness": {"score": round(sharp_score, 1), "passed": sharp_pass, "message": sharp_msg, "threshold": 7.0},
                "exposure": {"score": round(exp_score, 1), "passed": exp_pass, "message": exp_msg, "range": "20-235"},
                "text_check": {"count": text_count, "passed": text_pass, "message": text_msg}
            }
        }
