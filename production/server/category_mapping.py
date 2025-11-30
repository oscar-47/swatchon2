"""
Category Mapping System
将现有的9类Woven和10类Knit简化为Elif要求的5类系统
"""

# Elif要求的5类Woven分类
WOVEN_TARGET_CATEGORIES = {
    "Plain Weave": ["Plain", "Dobby", "Eyelet"],  # 平纹及其变体
    "Twill": ["Twill Weave"],  # 斜纹
    "Satin": ["Satin Weave"],  # 缎纹
    "Jacquard": ["Jacquard Weave"],  # 提花
    "Pile Weave": ["Pile Weave"]  # 起绒
}

# Elif要求的5类Knit分类
KNIT_TARGET_CATEGORIES = {
    "Single Jersey": ["Single"],  # 单面针织
    "Rib": ["Pique"],  # 罗纹（将Pique归入）
    "Jacquard Knit": ["Jacquard Knit"],  # 提花针织
    "Pile Knit": ["Pile Knit", "Crepe Knit", "Low Gauge Knit"],  # 起绒针织（合并多种）
    "Mesh": ["Mesh", "Lace Knit", "Tricot"]  # 网眼（合并相关）
}

# 需要特殊处理的Double结构（不再作为独立类别）
DOUBLE_STRUCTURES = ["Double Weave", "Double"]

# 反向映射：从细分类到5大类
def build_reverse_mapping(target_categories):
    """构建反向映射字典"""
    reverse_map = {}
    for target, sources in target_categories.items():
        for source in sources:
            reverse_map[source] = target
    return reverse_map

WOVEN_REVERSE_MAP = build_reverse_mapping(WOVEN_TARGET_CATEGORIES)
KNIT_REVERSE_MAP = build_reverse_mapping(KNIT_TARGET_CATEGORIES)


def map_to_target_category(prediction: str, model_type: str) -> dict:
    """
    将细分类映射到目标类别
    
    Args:
        prediction: 模型预测的原始类别
        model_type: 模型类型 ("woven_multi" 或 "knit_multi")
    
    Returns:
        dict: {
            "mapped_category": 映射后的类别,
            "original_category": 原始类别,
            "is_double_structure": 是否是Double结构,
            "needs_back_verification": 是否需要反面验证
        }
    """
    result = {
        "mapped_category": prediction,
        "original_category": prediction,
        "is_double_structure": False,
        "needs_back_verification": False
    }
    
    # 检查是否是Double结构
    if prediction in DOUBLE_STRUCTURES:
        result["is_double_structure"] = True
        result["needs_back_verification"] = True
        # Double结构不做映射，而是标记需要双面验证
        result["mapped_category"] = "Unknown - Requires Back Side Verification"
        return result
    
    # 根据模型类型选择映射表
    if model_type == "woven_multi":
        mapping = WOVEN_REVERSE_MAP
    elif model_type == "knit_multi":
        mapping = KNIT_REVERSE_MAP
    else:
        # 如果是woven_vs_knit，不需要映射
        return result
    
    # 执行映射
    if prediction in mapping:
        result["mapped_category"] = mapping[prediction]
    else:
        # 未找到映射，保持原样但标记
        result["mapped_category"] = f"Unmapped: {prediction}"
    
    return result


def get_simplified_classes(model_type: str) -> list:
    """获取简化后的类别列表"""
    if model_type == "woven_multi":
        return list(WOVEN_TARGET_CATEGORIES.keys())
    elif model_type == "knit_multi":
        return list(KNIT_TARGET_CATEGORIES.keys())
    elif model_type == "woven_vs_knit":
        return ["Knit", "Woven"]
    return []
