"""
Double Structure Verifier
通过正反面对比检测Double结构
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict


class DoubleStructureVerifier:
    """双层结构验证器"""
    
    def __init__(self, similarity_threshold: float = 0.75):
        """
        Args:
            similarity_threshold: 相似度阈值，超过此值认为可能是Double结构
        """
        self.similarity_threshold = similarity_threshold
    
    @staticmethod
    def compute_image_similarity(img1: Image.Image, img2: Image.Image) -> float:
        """
        计算两张图片的相似度
        使用简单的像素级相似度（可以后续升级为特征相似度）
        
        Args:
            img1: 正面图像
            img2: 反面图像
        
        Returns:
            相似度分数 (0-1)
        """
        # 统一尺寸
        size = (224, 224)
        img1_resized = img1.resize(size).convert('RGB')
        img2_resized = img2.resize(size).convert('RGB')
        
        # 转换为numpy数组
        arr1 = np.array(img1_resized).astype(np.float32)
        arr2 = np.array(img2_resized).astype(np.float32)
        
        # 计算归一化相关系数
        arr1_flat = arr1.reshape(-1)
        arr2_flat = arr2.reshape(-1)
        
        # 均值归一化
        arr1_norm = arr1_flat - arr1_flat.mean()
        arr2_norm = arr2_flat - arr2_flat.mean()
        
        # 计算相关系数
        correlation = np.corrcoef(arr1_norm, arr2_norm)[0, 1]
        
        # 转换到0-1范围
        similarity = (correlation + 1) / 2
        
        return float(similarity)
    
    @staticmethod
    def compute_histogram_similarity(img1: Image.Image, img2: Image.Image) -> float:
        """
        计算直方图相似度（颜色分布相似度）
        
        Args:
            img1: 正面图像
            img2: 反面图像
        
        Returns:
            相似度分数 (0-1)
        """
        from PIL import ImageStat
        
        # 转换为RGB
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')
        
        # 获取直方图
        hist1 = img1.histogram()
        hist2 = img2.histogram()
        
        # 计算直方图交集
        intersection = sum(min(h1, h2) for h1, h2 in zip(hist1, hist2))
        total = sum(hist1)
        
        similarity = intersection / total if total > 0 else 0
        return float(similarity)
    
    def verify_double_structure(
        self, 
        front_image: Image.Image, 
        back_image: Image.Image
    ) -> Dict:
        """
        验证是否为双层结构
        
        Args:
            front_image: 正面图像
            back_image: 反面图像
        
        Returns:
            验证结果字典
        """
        # 计算像素相似度
        pixel_similarity = self.compute_image_similarity(front_image, back_image)
        
        # 计算直方图相似度
        histogram_similarity = self.compute_histogram_similarity(front_image, back_image)
        
        # 综合评分（加权平均）
        overall_similarity = (pixel_similarity * 0.6 + histogram_similarity * 0.4)
        
        # 判断是否为Double结构
        is_double = overall_similarity >= self.similarity_threshold
        
        # 生成建议
        if is_double:
            suggestion = "✅ Front and back sides are highly similar. This is likely a DOUBLE structure (Double Weave or Double Knit)."
            confidence_level = "High"
        elif overall_similarity >= 0.5:
            suggestion = "⚠️ Front and back sides show moderate similarity. Possibly a reversible fabric or double structure with different surface treatments."
            confidence_level = "Medium"
        else:
            suggestion = "❌ Front and back sides are different. This is likely a SINGLE-LAYER structure."
            confidence_level = "Low"
        
        return {
            "is_double_structure": is_double,
            "pixel_similarity": round(pixel_similarity, 3),
            "histogram_similarity": round(histogram_similarity, 3),
            "overall_similarity": round(overall_similarity, 3),
            "threshold": self.similarity_threshold,
            "confidence_level": confidence_level,
            "suggestion": suggestion
        }
