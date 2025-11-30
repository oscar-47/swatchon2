"""
OCR Field Parser
从OCR文本中提取结构化字段（成分、重量、花纹等）
"""

import re
from typing import Dict, Optional


class OCRFieldParser:
    """OCR文本字段解析器"""
    
    @staticmethod
    def parse_composition(text: str) -> Optional[str]:
        """
        提取纤维成分
        示例: "Cotton 100%", "Polyester 65% Cotton 35%"
        """
        patterns = [
            r'(?:composition|fibre|fiber|content)[:\s]+([^,\n]+)',
            r'(\d+%\s+\w+(?:\s+\d+%\s+\w+)*)',
            r'((?:\w+\s+\d+%(?:\s+)?)+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                composition = match.group(1).strip()
                # 清理格式
                composition = re.sub(r'\s+', ' ', composition)
                return composition
        
        return None
    
    @staticmethod
    def parse_weight(text: str) -> Optional[str]:
        """
        提取重量
        示例: "132 gsm", "200g/m²"
        """
        patterns = [
            r'(?:weight|gsm)[:\s]*(\d+\.?\d*\s*g?s?m)',
            r'(\d+\.?\d*\s*g/m[²2])',
            r'(\d+\.?\d*\s*gsm)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    @staticmethod
    def parse_pattern(text: str) -> Optional[str]:
        """
        提取花纹类型
        示例: "Plaid", "Checked", "Stripe", "Solid"
        """
        known_patterns = [
            'plaid', 'checked', 'check', 'stripe', 'striped', 
            'solid', 'plain', 'floral', 'dot', 'polka',
            'geometric', 'abstract', 'print', 'digital print'
        ]
        
        # 查找pattern字段
        pattern_match = re.search(r'pattern[:\s]+(\w+)', text, re.IGNORECASE)
        if pattern_match:
            return pattern_match.group(1).capitalize()
        
        # 在文本中搜索已知花纹
        text_lower = text.lower()
        for pattern in known_patterns:
            if pattern in text_lower:
                return pattern.capitalize()
        
        return None
    
    @staticmethod
    def parse_width(text: str) -> Optional[str]:
        """
        提取幅宽
        示例: "56 inch", "150 cm"
        """
        patterns = [
            r'(?:width|wide)[:\s]*(\d+\.?\d*\s*(?:inch|in|cm|m))',
            r'(\d+\.?\d*\s*(?:inch|in))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    @staticmethod
    def parse_thickness(text: str) -> Optional[str]:
        """
        提取厚度
        示例: "0.43 mm"
        """
        patterns = [
            r'(?:thickness|thick)[:\s]*(\d+\.?\d*\s*mm)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    @staticmethod
    def parse_sustainability(text: str) -> Optional[str]:
        """
        提取可持续性信息
        示例: "Dead Stock", "Organic", "Recycled"
        """
        sustainability_keywords = [
            'dead stock', 'deadstock', 'organic', 'recycled', 
            'sustainable', 'eco-friendly', 'gots', 'oeko-tex'
        ]
        
        text_lower = text.lower()
        for keyword in sustainability_keywords:
            if keyword in text_lower:
                return keyword.title()
        
        return None
    
    @staticmethod
    def parse_origin(text: str) -> Optional[str]:
        """
        提取产地/发货地
        示例: "Ships from: Korea", "Made in China"
        """
        patterns = [
            r'(?:ships from|origin|made in)[:\s]+(\w+(?:\s+\w+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def parse_all_fields(self, ocr_text: str) -> Dict[str, Optional[str]]:
        """
        解析所有字段
        
        Args:
            ocr_text: OCR识别的原始文本
        
        Returns:
            包含所有解析字段的字典
        """
        return {
            "composition": self.parse_composition(ocr_text),
            "weight": self.parse_weight(ocr_text),
            "pattern": self.parse_pattern(ocr_text),
            "width": self.parse_width(ocr_text),
            "thickness": self.parse_thickness(ocr_text),
            "sustainability": self.parse_sustainability(ocr_text),
            "origin": self.parse_origin(ocr_text),
            "raw_text": ocr_text  # 保留原始文本
        }
