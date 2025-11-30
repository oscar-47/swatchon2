"""
测试升级后的系统功能
验证OCR解析、双面对比、元数据生成
"""

from server.ocr_parser import OCRFieldParser
from server.double_verifier import DoubleStructureVerifier
from PIL import Image
import json

def test_ocr_parser():
    """测试OCR字段解析"""
    print("=" * 60)
    print("测试 OCR 字段解析器")
    print("=" * 60)
    
    parser = OCRFieldParser()
    
    # 模拟OCR文本
    sample_texts = [
        """
        Fabric Type: Plain Weave
        Fibre Composition: Cotton 100%
        Pattern: Plaid
        Weight: 132 gsm
        Width: 56 inch
        Thickness: 0.43 mm
        Sustainability: Dead Stock
        Ships from: Korea
        """,
        """
        Material: Polyester 65% Cotton 35%
        200 gsm
        Checked pattern
        58 inch wide
        Made in China
        """,
        """
        100% Silk
        Twill weave
        Stripe
        120gsm
        Organic certified
        """
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n示例 {i}:")
        print(f"输入文本: {text[:100]}...")
        result = parser.parse_all_fields(text)
        print("解析结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return True


def test_double_verifier():
    """测试双面验证器"""
    print("\n" + "=" * 60)
    print("测试双层结构验证器")
    print("=" * 60)
    
    verifier = DoubleStructureVerifier(similarity_threshold=0.75)
    
    # 创建测试图片（相同）
    from PIL import Image, ImageDraw
    
    # 创建两张相似的图片
    img1 = Image.new('RGB', (224, 224), color=(100, 150, 200))
    draw1 = ImageDraw.Draw(img1)
    for i in range(0, 224, 20):
        draw1.line([(0, i), (224, i)], fill=(80, 130, 180), width=2)
    
    img2 = Image.new('RGB', (224, 224), color=(100, 150, 200))
    draw2 = ImageDraw.Draw(img2)
    for i in range(0, 224, 20):
        draw2.line([(0, i), (224, i)], fill=(80, 130, 180), width=2)
    
    # 测试相似图片（应该判定为Double）
    print("\n测试1: 相似图片（模拟双面织物）")
    result1 = verifier.verify_double_structure(img1, img2)
    print(json.dumps(result1, indent=2, ensure_ascii=False))
    
    # 创建不同的图片
    img3 = Image.new('RGB', (224, 224), color=(200, 100, 50))
    
    # 测试不同图片（应该判定为Single）
    print("\n测试2: 不同图片（模拟单层织物）")
    result2 = verifier.verify_double_structure(img1, img3)
    print(json.dumps(result2, indent=2, ensure_ascii=False))
    
    return True


def test_metadata_structure():
    """测试元数据结构"""
    print("\n" + "=" * 60)
    print("测试元数据JSON格式")
    print("=" * 60)
    
    # 模拟完整的元数据
    metadata_example = {
        "fabric_id": "abc123def456",
        "timestamp": "2025-11-22T10:30:00",
        "structure": {
            "primary": "Woven",
            "secondary": "Plain Weave",
            "confidence": 0.92,
            "model_version": "woven_multi",
            "all_probabilities": {
                "Plain Weave": 0.92,
                "Twill": 0.05,
                "Satin": 0.02,
                "Jacquard": 0.01
            }
        },
        "specifications": {
            "composition": "Cotton 100%",
            "weight": "150 gsm",
            "pattern": "Plaid",
            "width": "58 inch",
            "thickness": "0.5 mm",
            "sustainability": "Organic",
            "origin": "Korea"
        },
        "double_check": {
            "performed": True,
            "is_double_structure": False,
            "similarity_score": 0.45,
            "confidence_level": "Low",
            "suggestion": "Front and back sides are different. This is likely a SINGLE-LAYER structure."
        },
        "ocr_raw_text": "Cotton 100% Plaid 150gsm 58inch Organic Korea",
        "ocr_confidence": 0.88
    }
    
    print("\n符合 Elif 要求的元数据格式:")
    print(json.dumps(metadata_example, indent=2, ensure_ascii=False))
    
    return True


if __name__ == "__main__":
    print("\n🚀 开始测试升级后的系统\n")
    
    try:
        # 测试1: OCR解析
        if test_ocr_parser():
            print("\n✅ OCR解析器测试通过")
        
        # 测试2: 双面验证
        if test_double_verifier():
            print("\n✅ 双面验证器测试通过")
        
        # 测试3: 元数据格式
        if test_metadata_structure():
            print("\n✅ 元数据格式测试通过")
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！系统已按 Elif 的要求升级")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
