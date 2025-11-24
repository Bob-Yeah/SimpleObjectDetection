import os
import cv2
import numpy as np
from image_mask_generator import ImageMaskGenerator

def test_save_mask_functionality():
    # 创建一个测试用的ImageMaskGenerator实例
    generator = ImageMaskGenerator()
    # 设置必要的属性
    generator.image_path = "test_image.jpg"
    
    # 创建一个简单的mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # 在中心创建一个白色方块
    
    # 测试保存mask
    print("开始测试save_mask方法...")
    result = generator.save_mask(mask)
    
    # 检查是否保存成功
    expected_path = os.path.join(generator.mask_output_dir, "test_image_mask.png")
    if result and os.path.exists(expected_path):
        print(f"测试成功！mask已保存到: {expected_path}")
        # 验证文件是否真的是PNG格式
        img = cv2.imread(expected_path)
        if img is not None:
            print(f"成功读取保存的PNG文件，尺寸: {img.shape}")
            return True
        else:
            print("无法读取保存的文件，可能格式有问题")
            return False
    else:
        print(f"测试失败！未能在预期路径找到文件: {expected_path}")
        return False

if __name__ == "__main__":
    test_save_mask_functionality()