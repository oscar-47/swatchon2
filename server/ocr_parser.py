"""
OCR Field Parser
从OCR文本中提取结构化字段（成分、重量、花纹等）
"""

import re
from typing import Dict, Optional


class OCRFieldParser:
    """OCR文本字段解析器"""
    
    FIBER_NAMES = (
        'cotton|polyester|nylon|wool|silk|rayon|viscose|linen|acrylic|modal|'
        'spandex|elastane|lycra|tencel|lyocell|bamboo|cashmere|hemp|jute|'
        'ramie|acetate|cupro|polypropylene|kevlar|nomex|pla|ptfe'
    )

    @classmethod
    def parse_composition(cls, text: str) -> Optional[str]:
        """
        提取纤维成分
        示例: "Cotton 100%", "Polyester 65% Cotton 35%", "Nylon 76% Viscose Rayon 24%"
        """
        fibers = cls.FIBER_NAMES

        # Pattern A: "Fiber Composition: Nylon 76% Viscose Rayon 24%" — labeled field
        # Supports "fiber composition", "fibre content", "composition" with optional ":" and whitespace
        m = re.search(
            r'(?:fib(?:er|re)\s+composition|fib(?:er|re)\s+content|composition|fibre|fiber|content)\s*:?\s*'
            r'((?:(?:' + fibers + r')[\s\-]*(?:' + fibers + r')?\s*\d+\s*%?\s*)+)',
            text, re.IGNORECASE
        )
        if m:
            return re.sub(r'\s+', ' ', m.group(1)).strip()

        # Pattern B: "Fiber% Fiber%" anywhere — e.g. "Nylon 76% Viscose Rayon 24%"
        parts = re.findall(
            r'(?:' + fibers + r')(?:[\s\-]*(?:' + fibers + r'))?\s*\d+%?',
            text, re.IGNORECASE
        )
        if parts:
            return ', '.join(p.strip() for p in parts[:4])  # max 4 components

        # Pattern C: "CO 100" / "P 49 CO 36" shorthand
        shorts = re.findall(r'\b([A-Z]{1,3})\s+(\d{1,3})\b', text)
        short_map = {'CO':'Cotton','P':'Polyester','N':'Nylon','SP':'Spandex',
                     'SE':'Silk','WO':'Wool','VI':'Viscose','LI':'Linen'}
        if shorts:
            decoded = []
            for code, pct in shorts:
                name = short_map.get(code.upper())
                if name:
                    decoded.append(f"{name} {pct}%")
            if decoded:
                return ', '.join(decoded[:4])

        return None
    
    @staticmethod
    def parse_weight(text: str) -> Optional[str]:
        """
        提取重量
        示例: "132 gsm", "200g/m²"
        """
        patterns = [
            r'weight\s*[:\-]?\s*(\d+\.?\d*\s*(?:gsm|g/m[²2]|g\s*/\s*m[²2]))',
            r'weight\s*[:\-]?\s*(\d+\.?\d*)\s+gsm',
            r'(\d+\.?\d*\s*g/m[²2])',
            r'(\d+\.?\d*\s*gsm)',
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                if i == 1 and 'gsm' not in val.lower():
                    val = val + ' gsm'
                return val

        return None

    @staticmethod
    def parse_pattern(text: str) -> Optional[str]:
        """
        提取花纹类型
        示例: "Plaid", "Checked", "Stripe", "Solid"
        """
        known_patterns = [
            'plaid', 'checked', 'check', 'stripe', 'striped',
            'solid', 'solio', 'plain', 'floral', 'dot', 'polka',
            'geometric', 'abstract', 'print', 'digital print',
            'jacquard', 'herringbone', 'tweed', 'houndstooth',
        ]
        
        # OCR typo corrections
        typo_map = {'solio': 'Solid'}

        # 查找pattern字段
        pattern_match = re.search(r'pattern[:\s]+(\w+)', text, re.IGNORECASE)
        if pattern_match:
            val = pattern_match.group(1).strip().lower()
            return typo_map.get(val, val.capitalize())

        # 在文本中搜索已知花纹
        text_lower = text.lower()
        for pattern in known_patterns:
            if pattern in text_lower:
                return typo_map.get(pattern, pattern.capitalize())

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
        示例: "Ships from: Korea, Republic of", "Made in China"
        """
        # Accept multi-word country names and stop at newline / sentinel keyword
        pattern = (
            r'(?:ships\s+from|country\s+of\s+origin|origin|made\s+in)\s*[:\-]?\s*'
            r'([A-Za-z][A-Za-z ,.\'\-]{1,60}?)'
            r'(?=\s*(?:\n|\r|\.|,\s*[A-Z]{3,}|weight|width|thickness|pattern|fiber|fibre|composition|care|pricing|price|amount|$))'
        )
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            out = match.group(1).strip().rstrip(',.;:')
            # Clean obvious trailing words like "1", emojis
            return re.sub(r'\s+', ' ', out)
        return None
    
    @staticmethod
    def parse_care(text: str) -> Optional[str]:
        """
        提取洗涤说明
        示例: "Hand wash only, Do not bleach, Line dry"
        """
        # Try to find a care instructions block
        care_match = re.search(
            r'(?:care\s*(?:instructions?)?|washing)[:\s]+(.*?)(?=\n\n|\Z|ships?\s+from|origin|made\s+in|price|pricing)',
            text, re.IGNORECASE | re.DOTALL
        )
        if care_match:
            care = care_match.group(1).strip()
            care = re.sub(r'\s+', ' ', care)
            return care

        # Fallback: collect individual care-related phrases
        care_phrases = []
        care_patterns = [
            r'(?:hand\s+)?wash\s+[^,.\n]+',
            r'(?:do\s+not\s+)?bleach[^,.\n]*',
            r'(?:tumble\s+)?dry\s+[^,.\n]*',
            r'(?:line\s+)?dry[^,.\n]*',
            r'iron\s+[^,.\n]*',
            r'(?:do\s+not\s+)?wring[^,.\n]*',
            r'(?:dry\s+)?clean\s+[^,.\n]*',
            r'turn\s+inside\s+out[^,.\n]*',
            r'wash\s+separately[^,.\n]*',
        ]
        for p in care_patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                phrase = m.group(0).strip().rstrip(',;.')
                if phrase and len(phrase) > 3:
                    care_phrases.append(phrase.capitalize())

        if care_phrases:
            # Deduplicate
            seen = set()
            unique = []
            for p in care_phrases:
                key = p.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            return ', '.join(unique)

        return None

    @staticmethod
    def parse_pricing(text: str) -> Optional[list]:
        """
        提取价格信息
        示例: "1-49 yds $6.80" -> [{"amount": "1-49 yds", "price": "$6.80", "shipping": "$0.71"}]
        """
        rows = []
        # Match patterns like "1 - 49 yds $6.80 $0.71"
        for m in re.finditer(
            r'(\d+\s*[-–]\s*\d+\s*(?:yds?|yards?|m|meters?)(?:\s+and\s+up)?)\s+\$?([\d.]+)\s+\$?([\d.]+)',
            text, re.IGNORECASE
        ):
            rows.append({
                "amount": m.group(1).strip(),
                "price": f"${m.group(2)}",
                "shipping": f"${m.group(3)}"
            })
        # Also match "100 yds and up $4.41 $0.58"
        for m in re.finditer(
            r'(\d+\s*(?:yds?|yards?|m|meters?)\s+and\s+up)\s+\$?([\d.]+)\s+\$?([\d.]+)',
            text, re.IGNORECASE
        ):
            rows.append({
                "amount": m.group(1).strip(),
                "price": f"${m.group(2)}",
                "shipping": f"${m.group(3)}"
            })
        return rows if rows else None

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
            "care": self.parse_care(ocr_text),
            "sustainability": self.parse_sustainability(ocr_text),
            "origin": self.parse_origin(ocr_text),
            "pricing": self.parse_pricing(ocr_text),
            "raw_text": ocr_text
        }
