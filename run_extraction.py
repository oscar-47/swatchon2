"""
运行纯纺织物提取
"""
from extract_pure_fabric import FabricExtractor
import time

def main():
    print("="*80)
    print("纯纺织物图片提取工具")
    print("="*80)
    print()
    print("功能：从训练图片中移除标尺和标签，提取纯纺织物区域")
    print()
    
    # 创建提取器
    extractor = FabricExtractor(strategy='hybrid')
    
    # 输入输出目录
    input_dirs = [
        'outputs/knit_category_details',
        'outputs/woven_category_details'
    ]
    output_dir = 'outputs/pure_fabric'
    
    print("输入目录:")
    for d in input_dirs:
        print(f"  - {d}")
    print(f"\n输出目录: {output_dir}")
    print()
    
    # 确认
    response = input("开始处理？(y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 开始处理
    start_time = time.time()
    
    extractor.batch_process(
        input_dirs=input_dirs,
        output_dir=output_dir,
        copy_json=True
    )
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    
    print(f"\n总耗时: {elapsed_time:.2f} 秒")
    print(f"平均速度: {extractor.stats['total']/elapsed_time:.2f} 张/秒")
    print()
    print("完成！请检查输出目录。")

if __name__ == "__main__":
    main()

