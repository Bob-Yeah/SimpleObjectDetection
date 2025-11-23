import cv2
import numpy as np
import os
import glob

# 设置颜色索引计算的常量参数
CLR_RSB = 3
TAB_WIDTH = (1 << (8 - CLR_RSB))  # 1 << (8-3) = 1 << 5 = 32
TAB_WIDTH_2 = TAB_WIDTH * TAB_WIDTH  # 32 * 32 = 1024
TAB_SIZE = TAB_WIDTH * TAB_WIDTH_2  # 32 * 1024 = 32768

print(f"常量参数设置:")
print(f"CLR_RSB = {CLR_RSB}")
print(f"TAB_WIDTH = {TAB_WIDTH}")
print(f"TAB_WIDTH_2 = {TAB_WIDTH_2}")
print(f"TAB_SIZE = {TAB_SIZE}")


def color_index(pix):
    """
    计算颜色索引
    根据用户提供的函数转换而来：
    private int _color_index(byte[] pix)
    {
        return ((pix[0] >> CLR_RSB) * TAB_WIDTH_2 +
                (pix[1] >> CLR_RSB) * TAB_WIDTH +
                (pix[2] >> CLR_RSB));
    }
    
    在OpenCV中，图像的颜色通道顺序是BGR，与函数中的顺序一致
    
    参数:
        pix: 包含BGR三个通道值的数组
    返回:
        计算得到的颜色索引
    """
    # 确保pix是有效的三通道值
    if len(pix) < 3:
        raise ValueError("pix必须包含至少3个元素（BGR通道值）")
    
    # 按照公式计算颜色索引
    # pix[0]对应蓝色通道，pix[1]对应绿色通道，pix[2]对应红色通道
    index = ((pix[0] >> CLR_RSB) * TAB_WIDTH_2 +
             (pix[1] >> CLR_RSB) * TAB_WIDTH +
             (pix[2] >> CLR_RSB))
    
    return index


# 测试颜色索引函数
print("\n测试颜色索引函数:")
# 测试黑色像素 (0, 0, 0)
black_pixel = [0, 0, 0]
print(f"黑色像素 {black_pixel} 的索引: {color_index(black_pixel)}")
# 测试白色像素 (255, 255, 255)
white_pixel = [255, 255, 255]
print(f"白色像素 {white_pixel} 的索引: {color_index(white_pixel)}")
# 测试红色像素 (0, 0, 255) - 注意OpenCV是BGR顺序
red_pixel = [0, 0, 255]
print(f"红色像素 {red_pixel} 的索引: {color_index(red_pixel)}")


def get_file_prefix(filename):
    """
    获取文件名的前缀（不包含扩展名和_mask部分）
    
    参数:
        filename: 文件名
    返回:
        提取的前缀
    """
    # 移除扩展名
    base_name = os.path.splitext(filename)[0]
    # 移除_mask后缀（如果存在）
    if base_name.endswith('_mask'):
        base_name = base_name[:-5]  # 移除'_mask'
    return base_name


def load_images_and_masks(images_dir, masks_dir):
    """
    加载图片和对应的mask文件
    
    参数:
        images_dir: 图片文件夹路径
        masks_dir: mask文件夹路径
    返回:
        image_mask_pairs: 包含(图片路径, mask路径)的列表
    """
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    # 获取所有mask文件
    mask_files = []
    for ext in image_extensions:
        mask_files.extend(glob.glob(os.path.join(masks_dir, ext)))
    
    # 创建mask文件名前缀到路径的映射
    mask_prefix_map = {}
    for mask_file in mask_files:
        mask_filename = os.path.basename(mask_file)
        mask_prefix = get_file_prefix(mask_filename)
        mask_prefix_map[mask_prefix] = mask_file
    
    # 匹配图片和对应的mask
    image_mask_pairs = []
    for image_file in image_files:
        image_filename = os.path.basename(image_file)
        image_prefix = get_file_prefix(image_filename)
        
        if image_prefix in mask_prefix_map:
            image_mask_pairs.append((image_file, mask_prefix_map[image_prefix]))
            print(f"匹配成功: {image_filename} -> {os.path.basename(mask_prefix_map[image_prefix])}")
        else:
            print(f"警告: 没有找到 {image_filename} 对应的mask文件")
    
    print(f"\n总共匹配到 {len(image_mask_pairs)} 对图片和mask文件")
    return image_mask_pairs


def read_image_and_mask(image_path, mask_path):
    """
    读取图片和对应的mask文件
    
    参数:
        image_path: 图片路径
        mask_path: mask路径
    返回:
        (image, mask): 图片和mask的numpy数组
    """
    # 读取图片（彩色）
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片文件: {image_path}")
    
    # 读取mask（灰度）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取mask文件: {mask_path}")
    
    # 确保mask的大小与图片一致
    if image.shape[:2] != mask.shape:
        print(f"警告: 图片和mask的大小不一致，将调整mask大小")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # 确保mask是二值图像（255表示前景，0表示背景）
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return image, mask


def calculate_histograms(image, mask):
    """
    计算前景和背景区域的颜色直方图
    
    参数:
        image: 输入彩色图像 (BGR格式)
        mask: 二值mask (255表示前景，0表示背景)
    返回:
        (foreground_hist, background_hist): 前景和背景的颜色直方图
    """
    # 初始化直方图数组
    foreground_hist = np.zeros(TAB_SIZE, dtype=np.int32)
    background_hist = np.zeros(TAB_SIZE, dtype=np.int32)
    
    # 分离前景和背景像素
    # 前景像素 (mask == 255)
    foreground_pixels = image[mask == 255]
    # 背景像素 (mask == 0)
    background_pixels = image[mask == 0]
    
    print(f"\n统计信息:")
    print(f"前景像素数量: {len(foreground_pixels)}")
    print(f"背景像素数量: {len(background_pixels)}")
    
    # 计算前景直方图
    if len(foreground_pixels) > 0:
        # 使用向量化操作计算颜色索引
        # 将RGB值右移CLR_RSB位并转换为int32避免溢出
        b_vals = (foreground_pixels[:, 0] >> CLR_RSB).astype(np.int32)
        g_vals = (foreground_pixels[:, 1] >> CLR_RSB).astype(np.int32)
        r_vals = (foreground_pixels[:, 2] >> CLR_RSB).astype(np.int32)
        
        # 计算索引
        foreground_indices = b_vals * TAB_WIDTH_2 + g_vals * TAB_WIDTH + r_vals
        
        # 统计直方图
        for idx in np.unique(foreground_indices):
            foreground_hist[idx] = np.sum(foreground_indices == idx)
    
    # 计算背景直方图
    if len(background_pixels) > 0:
        # 使用向量化操作计算颜色索引
        b_vals = (background_pixels[:, 0] >> CLR_RSB).astype(np.int32)
        g_vals = (background_pixels[:, 1] >> CLR_RSB).astype(np.int32)
        r_vals = (background_pixels[:, 2] >> CLR_RSB).astype(np.int32)
        
        # 计算索引
        background_indices = b_vals * TAB_WIDTH_2 + g_vals * TAB_WIDTH + r_vals
        
        # 统计直方图
        for idx in np.unique(background_indices):
            background_hist[idx] = np.sum(background_indices == idx)
    
    # 验证直方图统计是否正确
    foreground_sum = np.sum(foreground_hist)
    background_sum = np.sum(background_hist)
    print(f"前景直方图像素总数: {foreground_sum}")
    print(f"背景直方图像素总数: {background_sum}")
    
    if foreground_sum != len(foreground_pixels):
        print(f"警告: 前景像素统计不一致")
    if background_sum != len(background_pixels):
        print(f"警告: 背景像素统计不一致")
    
    return foreground_hist, background_hist


def save_histograms(foreground_hist, background_hist, output_dir="."):
    """
    保存直方图数据到文件
    
    参数:
        foreground_hist: 前景直方图
        background_hist: 背景直方图
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    foreground_path = os.path.join(output_dir, "foreground_histogram.npy")
    background_path = os.path.join(output_dir, "background_histogram.npy")
    
    np.save(foreground_path, foreground_hist)
    np.save(background_path, background_hist)
    
    print(f"\n直方图已保存:")
    print(f"前景直方图: {foreground_path}")
    print(f"背景直方图: {background_path}")


def print_histogram_info(foreground_hist, background_hist):
    """
    打印直方图的统计信息
    
    参数:
        foreground_hist: 前景直方图
        background_hist: 背景直方图
    """
    print(f"\n最终直方图统计:")
    print(f"前景像素总数: {np.sum(foreground_hist)}")
    print(f"背景像素总数: {np.sum(background_hist)}")
    print(f"前景非零颜色数量: {np.count_nonzero(foreground_hist)}")
    print(f"背景非零颜色数量: {np.count_nonzero(background_hist)}")
    
    # 找出前景中出现频率最高的前5个颜色索引
    foreground_top_indices = np.argsort(foreground_hist)[::-1][:5]
    foreground_top_values = foreground_hist[foreground_top_indices]
    print(f"\n前景中出现频率最高的5个颜色索引:")
    for idx, count in zip(foreground_top_indices, foreground_top_values):
        print(f"  索引 {idx}: {count} 次")
    
    # 找出背景中出现频率最高的前5个颜色索引
    background_top_indices = np.argsort(background_hist)[::-1][:5]
    background_top_values = background_hist[background_top_indices]
    print(f"\n背景中出现频率最高的5个颜色索引:")
    for idx, count in zip(background_top_indices, background_top_values):
        print(f"  索引 {idx}: {count} 次")


def main():
    """
    主函数，整合所有功能并输出结果
    """
    print("===== 颜色直方图统计工具 =====\n")
    
    # 默认文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "images")
    masks_dir = os.path.join(current_dir, "masks")
    
    print(f"使用的文件夹路径:")
    print(f"图片文件夹: {images_dir}")
    print(f"Mask文件夹: {masks_dir}")
    print()
    
    # 验证文件夹是否存在
    if not os.path.exists(images_dir):
        print(f"错误: 图片文件夹不存在: {images_dir}")
        return
    if not os.path.exists(masks_dir):
        print(f"错误: Mask文件夹不存在: {masks_dir}")
        return
    
    # 加载图片和mask文件对
    image_mask_pairs = load_images_and_masks(images_dir, masks_dir)
    
    if not image_mask_pairs:
        print("没有找到匹配的图片和mask文件，程序退出")
        return
    
    # 初始化总直方图
    total_foreground_hist = np.zeros(TAB_SIZE, dtype=np.int32)
    total_background_hist = np.zeros(TAB_SIZE, dtype=np.int32)
    
    # 处理每对图片和mask
    for i, (image_path, mask_path) in enumerate(image_mask_pairs):
        print(f"\n\n===== 处理第 {i+1}/{len(image_mask_pairs)} 对文件 =====")
        print(f"图片: {os.path.basename(image_path)}")
        print(f"Mask: {os.path.basename(mask_path)}")
        
        try:
            # 读取图片和mask
            image, mask = read_image_and_mask(image_path, mask_path)
            print(f"图片大小: {image.shape}")
            print(f"Mask大小: {mask.shape}")
            
            # 使用优化版本计算直方图
            foreground_hist, background_hist = calculate_histograms(image, mask)
            
            # 累加总直方图
            total_foreground_hist += foreground_hist
            total_background_hist += background_hist
            
            # 打印当前图片的直方图信息
            print(f"\n当前图片直方图统计:")
            print(f"前景像素数: {np.sum(foreground_hist)}")
            print(f"背景像素数: {np.sum(background_hist)}")
            
        except Exception as e:
            print(f"处理出错: {str(e)}")
            continue
    
    # 打印最终结果
    print(f"\n\n===== 最终统计结果 =====")
    print_histogram_info(total_foreground_hist, total_background_hist)
    
    # 保存直方图数据
    save_histograms(total_foreground_hist, total_background_hist)
    
    print(f"\n程序执行完成！")
    print(f"最终输出:")
    print(f"1. 前景直方图: 大小为 {TAB_SIZE} 的int数组")
    print(f"2. 背景直方图: 大小为 {TAB_SIZE} 的int数组")


if __name__ == "__main__":
    main()