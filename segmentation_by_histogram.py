import cv2
import numpy as np
import os

# 设置颜色索引计算的常量参数（与buildColorHistogram.py保持一致）
CLR_RSB = 3
TAB_WIDTH = (1 << (8 - CLR_RSB))  # 32
TAB_WIDTH_2 = TAB_WIDTH * TAB_WIDTH  # 1024
TAB_SIZE = TAB_WIDTH * TAB_WIDTH_2  # 32768

def color_index(pix):
    """
    计算颜色索引
    
    参数:
        pix: 包含BGR三个通道值的数组
    返回:
        计算得到的颜色索引
    """
    if len(pix) < 3:
        raise ValueError("pix必须包含至少3个元素（BGR通道值）")
    
    # 将像素值转换为int32以避免整数溢出
    b_val = int(pix[0]) >> CLR_RSB
    g_val = int(pix[1]) >> CLR_RSB
    r_val = int(pix[2]) >> CLR_RSB
    
    index = (b_val * TAB_WIDTH_2 +
             g_val * TAB_WIDTH +
             r_val)
    
    return index

def load_histograms(foreground_path, background_path):
    """
    加载前景和背景直方图
    
    参数:
        foreground_path: 前景直方图文件路径
        background_path: 背景直方图文件路径
    返回:
        (foreground_hist, background_hist): 前景和背景直方图
    """
    try:
        foreground_hist = np.load(foreground_path)
        background_hist = np.load(background_path)
        print(f"成功加载直方图文件")
        print(f"前景直方图形状: {foreground_hist.shape}, 数据类型: {foreground_hist.dtype}")
        print(f"背景直方图形状: {background_hist.shape}, 数据类型: {background_hist.dtype}")
        return foreground_hist, background_hist
    except Exception as e:
        print(f"加载直方图文件失败: {e}")
        raise

def calculate_pixel_probabilities(image, foreground_hist, background_hist):
    """
    计算图像中每个像素是前景的概率
    
    参数:
        image: 输入彩色图像 (BGR格式)
        foreground_hist: 前景直方图
        background_hist: 背景直方图
    返回:
        probability_map: 前景概率图 (0-1范围)
    """
    height, width = image.shape[:2]
    probability_map = np.zeros((height, width), dtype=np.float32)
    
    # 使用向量化操作提高计算效率
    # 将图像数据重塑为2D数组，每行代表一个像素的BGR值
    pixels = image.reshape(-1, 3)
    
    # 计算所有像素的颜色索引
    b_vals = (pixels[:, 0] >> CLR_RSB).astype(np.int32)
    g_vals = (pixels[:, 1] >> CLR_RSB).astype(np.int32)
    r_vals = (pixels[:, 2] >> CLR_RSB).astype(np.int32)
    indices = b_vals * TAB_WIDTH_2 + g_vals * TAB_WIDTH + r_vals
    
    # 获取前景和背景的直方图值
    foreground_counts = foreground_hist[indices]
    background_counts = background_hist[indices]
    
    # 计算概率：前景值 / (前景值 + 背景值)
    # 添加一个小的epsilon避免除零错误
    epsilon = 1e-10
    total_counts = foreground_counts + background_counts + epsilon
    probabilities = foreground_counts / total_counts
    
    # 将概率值重塑回图像形状
    probability_map = probabilities.reshape(height, width)
    
    return probability_map

def save_probability_visualization(probability_map, output_path):
    """
    保存概率图的可视化结果
    
    参数:
        probability_map: 前景概率图
        output_path: 输出图像路径
    """
    # 将概率值映射到0-255范围用于可视化
    probability_vis = (probability_map * 255).astype(np.uint8)
    # 应用热力图色彩映射
    heatmap = cv2.applyColorMap(probability_vis, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap)
    print(f"概率可视化图像已保存到: {output_path}")

def save_binary_mask(probability_map, output_path, percentile_threshold=80):
    """
    基于概率图生成并保存二值化mask
    使用前20%的概率值作为阈值（即80%分位数）
    
    参数:
        probability_map: 前景概率图
        output_path: 输出mask路径
        percentile_threshold: 百分位数阈值，默认为80（表示前20%）
    """
    # 计算指定百分位数的阈值
    threshold = np.percentile(probability_map, percentile_threshold)
    print(f"使用阈值: {threshold:.4f} (第{percentile_threshold}百分位数)")
    
    # 二值化：概率高于阈值的为前景(255)，否则为背景(0)
    binary_mask = (probability_map > threshold).astype(np.uint8) * 255
    
    # 保存二值化mask
    cv2.imwrite(output_path, binary_mask)
    print(f"二值化mask已保存到: {output_path}")
    
    # 统计mask信息
    foreground_pixels = np.sum(binary_mask > 0)
    total_pixels = binary_mask.size
    foreground_ratio = foreground_pixels / total_pixels
    
    print(f"前景像素数: {foreground_pixels}")
    print(f"总像素数: {total_pixels}")
    print(f"前景比例: {foreground_ratio:.4f}")
    
    return binary_mask, threshold

def main():
    """
    主函数：加载直方图和测试图像，计算前景概率并可视化
    """
    # 文件路径设置
    foreground_hist_path = "d:/Projects/DetectionColorHistogram/foreground_histogram.npy"
    background_hist_path = "d:/Projects/DetectionColorHistogram/background_histogram.npy"
    test_image_path = "d:/Projects/DetectionColorHistogram/test.jpg"
    output_visualization_path = "d:/Projects/DetectionColorHistogram/probability_visualization.jpg"
    
    print("开始像素前景概率计算...")
    
    # 1. 加载直方图
    foreground_hist, background_hist = load_histograms(foreground_hist_path, background_hist_path)
    
    # 2. 读取测试图像
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        raise FileNotFoundError(f"无法读取测试图像: {test_image_path}")
    
    print(f"成功读取测试图像: {test_image_path}")
    print(f"测试图像形状: {test_image.shape}")
    
    # 3. 计算前景概率图
    print("正在计算像素前景概率...")
    probability_map = calculate_pixel_probabilities(test_image, foreground_hist, background_hist)
    
    # 4. 保存概率可视化结果
    save_probability_visualization(probability_map, output_visualization_path)
    
    # 5. 生成并保存二值化mask（基于前20%概率值）
    binary_mask_path = "d:/Projects/DetectionColorHistogram/binary_mask.png"
    binary_mask, threshold = save_binary_mask(probability_map, binary_mask_path, percentile_threshold=99)
    
    # 5. 统计概率分布信息
    mean_probability = np.mean(probability_map)
    max_probability = np.max(probability_map)
    min_probability = np.min(probability_map)
    
    print(f"概率计算完成！")
    print(f"概率统计信息:")
    print(f"- 平均概率: {mean_probability:.4f}")
    print(f"- 最大概率: {max_probability:.4f}")
    print(f"- 最小概率: {min_probability:.4f}")
    
    # 6. 保存原始概率数据
    probability_data_path = "d:/Projects/DetectionColorHistogram/pixel_probabilities.npy"
    np.save(probability_data_path, probability_map)
    print(f"原始概率数据已保存到: {probability_data_path}")
    
    # 7. 显示一些示例像素的概率值
    print(f"\n示例像素的前景概率:")
    height, width = probability_map.shape
    # 选择图像中心和四个角落的像素
    sample_points = [(0, 0), (0, width-1), (height-1, 0), (height-1, width-1), (height//2, width//2)]
    
    for y, x in sample_points:
        prob = probability_map[y, x]
        bgr = test_image[y, x]
        index = color_index(bgr)
        foreground_count = foreground_hist[index]
        background_count = background_hist[index]
        
        print(f"坐标 ({y}, {x}):")
        print(f"  - BGR值: {bgr}")
        print(f"  - 颜色索引: {index}")
        print(f"  - 前景计数: {foreground_count}")
        print(f"  - 背景计数: {background_count}")
        print(f"  - 前景概率: {prob:.4f}")

if __name__ == "__main__":
    main()