import cv2
import numpy as np
import os
from buildColorHistogram import color_index, calculate_histograms, load_images_and_masks

# 定义常量（与buildColorHistogram.py保持一致）
CLR_RSB = 3
TAB_WIDTH = (1 << (8 - CLR_RSB))
TAB_WIDTH_2 = TAB_WIDTH * TAB_WIDTH
TAB_SIZE = TAB_WIDTH * TAB_WIDTH_2

def test_color_index_function():
    """测试颜色索引计算函数"""
    print("\n测试颜色索引计算函数:")
    
    # 测试1: 黑色像素 (0,0,0)
    black_pixel = [0, 0, 0]  # BGR格式
    black_index = color_index(black_pixel)
    print(f"黑色像素 [0,0,0] 的索引值: {black_index}")
    
    # 测试2: 白色像素 (255,255,255)
    white_pixel = [255, 255, 255]
    white_index = color_index(white_pixel)
    print(f"白色像素 [255,255,255] 的索引值: {white_index}")
    
    # 测试3: 红色像素 (0,0,255)
    red_pixel = [0, 0, 255]
    red_index = color_index(red_pixel)
    print(f"红色像素 [0,0,255] 的索引值: {red_index}")
    
    # 测试4: 特定值像素，验证索引计算的准确性
    test_pixel = [100, 150, 200]
    expected_index = (test_pixel[0] >> CLR_RSB) * TAB_WIDTH_2 + \
                     (test_pixel[1] >> CLR_RSB) * TAB_WIDTH + \
                     (test_pixel[2] >> CLR_RSB)
    actual_index = color_index(test_pixel)
    print(f"测试像素 {test_pixel} 的索引值: {actual_index} (预期: {expected_index})")
    print(f"索引计算 {'正确' if actual_index == expected_index else '错误'}")

def test_histogram_calculation():
    """测试直方图计算功能"""
    print("\n测试直方图计算功能:")
    
    # 创建一个简单的测试图像和mask
    test_image = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # 在图像中设置不同颜色的区域
    test_image[0:5, 0:5] = [255, 0, 0]    # 蓝色区域
    test_image[0:5, 5:10] = [0, 255, 0]   # 绿色区域
    test_image[5:10, 0:5] = [0, 0, 255]   # 红色区域
    test_image[5:10, 5:10] = [255, 255, 255]  # 白色区域
    
    # 创建测试mask
    test_mask = np.zeros((10, 10), dtype=np.uint8)
    test_mask[0:5, :] = 255  # 上半部分为前景
    
    # 计算直方图
    foreground_hist, background_hist = calculate_histograms(test_image, test_mask)
    
    # 验证结果
    foreground_pixels = np.sum(test_mask == 255)
    background_pixels = np.sum(test_mask == 0)
    
    total_foreground_hist = np.sum(foreground_hist)
    total_background_hist = np.sum(background_hist)
    
    print(f"前景像素数量: {foreground_pixels}, 前景直方图总和: {total_foreground_hist}")
    print(f"背景像素数量: {background_pixels}, 背景直方图总和: {total_background_hist}")
    print(f"直方图计算 {'正确' if total_foreground_hist == foreground_pixels and total_background_hist == background_pixels else '错误'}")

def test_file_matching():
    """测试文件匹配功能"""
    print("\n测试文件匹配功能:")
    
    try:
        image_mask_pairs = load_images_and_masks()
        print(f"成功匹配到 {len(image_mask_pairs)} 对图像和mask文件")
        
        for i, (img_path, mask_path) in enumerate(image_mask_pairs, 1):
            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)
            print(f"对 {i}: 图像={img_name}, mask={mask_name}")
            
        return len(image_mask_pairs) > 0
    except Exception as e:
        print(f"文件匹配失败: {e}")
        return False

def test_with_real_data():
    """使用实际数据测试完整功能"""
    print("\n使用实际数据测试完整功能:")
    
    try:
        # 导入主函数
        from buildColorHistogram import main
        
        # 保存原始输出，捕获打印内容
        import io
        import sys
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        # 运行主函数
        main()
        
        # 恢复标准输出
        sys.stdout = old_stdout
        output = new_stdout.getvalue()
        
        # 检查输出内容
        print(output)
        print("测试完成！请检查输出结果是否符合预期。")
        return True
    except Exception as e:
        print(f"使用实际数据测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始测试 buildColorHistogram.py 的功能")
    print("=" * 50)
    
    test_color_index_function()
    test_histogram_calculation()
    file_matching_success = test_file_matching()
    
    if file_matching_success:
        test_with_real_data()
    else:
        print("\n由于文件匹配失败，跳过实际数据测试")
    
    print("\n" + "=" * 50)
    print("测试完成！")

if __name__ == "__main__":
    main()