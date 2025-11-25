import cv2
import numpy as np
import os
import time

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
    
    # 将像素值转换为int以避免整数溢出
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

def overlay_heatmap_on_image(image, probability_map, alpha=0.5):
    """
    将概率热图叠加到原始图像上
    
    参数:
        image: 原始彩色图像 (BGR格式)
        probability_map: 前景概率图 (0-1范围)
        alpha: 热图透明度 (0-1)
    返回:
        overlayed_image: 叠加了热图的图像
    """
    # 将概率值映射到0-255范围用于可视化
    probability_vis = (probability_map * 255).astype(np.uint8)
    
    # 应用热力图色彩映射
    heatmap = cv2.applyColorMap(probability_vis, cv2.COLORMAP_JET)
    
    # 将原始图像与热图叠加
    overlayed_image = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    
    return overlayed_image

def process_video_stream(foreground_hist_path, background_hist_path, camera_id=0, 
                        display_original=True, display_heatmap=True, display_overlay=True,
                        display_canny=True, canny_threshold1=50, canny_threshold2=150,
                        display_contour=True, contour_threshold_percentile=90,
                        resize_width=None, resize_height=None, alpha=0.5):
    """
    处理视频流并实时显示前景概率检测结果
    
    参数:
        foreground_hist_path: 前景直方图文件路径
        background_hist_path: 背景直方图文件路径
        camera_id: 摄像头ID，默认为0（内置摄像头）
        display_original: 是否显示原始图像
        display_heatmap: 是否显示热图
        display_overlay: 是否显示叠加了热图的图像
        display_canny: 是否显示Canny边缘检测结果
        canny_threshold1: Canny边缘检测的第一个阈值（低阈值）
        canny_threshold2: Canny边缘检测的第二个阈值（高阈值）
        display_contour: 是否显示轮廓检测结果
        contour_threshold_percentile: 轮廓检测的百分位数阈值（默认90%）
        resize_width: 调整视频帧宽度
        resize_height: 调整视频帧高度
        alpha: 热图透明度
    """
    # 加载直方图
    foreground_hist, background_hist = load_histograms(foreground_hist_path, background_hist_path)
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_id}")
    
    print("视频流处理开始...")
    print("按 'q' 键退出")
    
    # 用于计算帧率
    prev_time = 0
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧，退出")
            break
        
        # 调整图像大小（如果指定了）
        if resize_width is not None and resize_height is not None:
            frame = cv2.resize(frame, (resize_width, resize_height))
        
        # 计算前景概率图
        start_time = time.time()
        probability_map = calculate_pixel_probabilities(frame, foreground_hist, background_hist)
        processing_time = time.time() - start_time
        
        # 计算并显示帧率
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # 根据设置显示不同的窗口
        if display_original:
            # 在原始图像上添加帧率信息
            original_with_info = frame.copy()
            cv2.putText(original_with_info, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(original_with_info, "Original", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Original", original_with_info)
        
        if display_heatmap:
            # 创建热图并添加处理时间信息
            probability_vis = (probability_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(probability_vis, cv2.COLORMAP_JET)
            cv2.putText(heatmap, f"Process: {processing_time*1000:.1f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(heatmap, "Probability Heatmap", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Probability Heatmap", heatmap)
        
        if display_overlay:
            # 创建叠加图像并添加信息
            overlayed_image = overlay_heatmap_on_image(frame, probability_map, alpha)
            cv2.putText(overlayed_image, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(overlayed_image, "Overlay", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Overlay", overlayed_image)
            
        if display_canny:
            # 将probability_map从float32 (0-1)转换为uint8 (0-255)以便进行Canny边缘检测
            # probability_uint8 = (probability_map * 255).astype(np.uint8)
            probability_uint8 = frame
            # 应用Canny边缘检测
            edges = cv2.Canny(probability_uint8, canny_threshold1, canny_threshold2)
            
            # 将边缘图像转换为彩色以便添加文本
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # 添加信息文本
            cv2.putText(edges_colored, f"Threshold1: {canny_threshold1}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(edges_colored, f"Threshold2: {canny_threshold2}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(edges_colored, "Canny Edge Detection", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示Canny边缘检测结果
            cv2.imshow("Canny Edge Detection", edges_colored)
        
        if display_contour:
            # 使用百分位数计算二值化阈值
            threshold_value = np.percentile(probability_map, contour_threshold_percentile)
            
            # 二值化处理
            binary_mask = (probability_map > threshold_value).astype(np.uint8) * 255
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 计算每个轮廓的面积并按面积排序（降序）
            contours_with_area = [(cnt, cv2.contourArea(cnt)) for cnt in contours]
            contours_with_area.sort(key=lambda x: x[1], reverse=True)
            
            # 只保留面积最大的前5个轮廓
            max_contours_to_show = 5
            largest_contours = [cnt for cnt, area in contours_with_area[:max_contours_to_show]]
            largest_areas = [area for cnt, area in contours_with_area[:max_contours_to_show]]
            
            # 创建轮廓显示图像
            contour_image = frame.copy()
            
            # 绘制前5个最大面积的轮廓
            if largest_contours:
                cv2.drawContours(contour_image, largest_contours, -1, (0, 255, 0), 2)
                
                # 在最大的轮廓上显示其面积和中心圆形
                if largest_areas:
                    cnt = largest_contours[0]
                    area = largest_areas[0]
                    # 计算轮廓的边界矩形
                    x, y, w, h = cv2.boundingRect(cnt)
                    # 在轮廓附近显示面积
                    cv2.putText(contour_image, f"Largest Area: {area:.1f}", 
                                (x, max(0, y - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # 计算轮廓的中心坐标
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # 在轮廓中心绘制红色圆形
                        cv2.circle(contour_image, (cx, cy), 5, (0, 0, 255), -1)  # 红色圆形，半径5像素
            
            # 添加信息
            cv2.putText(contour_image, f"Threshold: {threshold_value:.3f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(contour_image, f"Total Contours: {len(contours)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(contour_image, f"Showing Top {min(max_contours_to_show, len(contours))}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(contour_image, "Object Contours", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示轮廓检测结果
            cv2.imshow("Object Contours", contour_image)
        
        # 等待按键事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户退出")
            break
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("视频流处理结束")

def main():
    """
    主函数：设置参数并启动视频流处理
    """
    # 文件路径设置
    foreground_hist_path = "foreground_histogram.npy"
    background_hist_path = "background_histogram.npy"

    
    # 视频处理参数
    camera_id = 0  # 使用默认摄像头
    display_original = True  # 显示原始图像
    display_heatmap = True   # 显示热图
    display_overlay = True   # 显示叠加图像
    display_canny = True     # 显示Canny边缘检测
    canny_threshold1 = 50    # Canny低阈值
    canny_threshold2 = 150   # Canny高阈值
    display_contour = True   # 显示轮廓检测
    contour_threshold_percentile = 90  # 轮廓阈值百分位数
    resize_width = 640       # 调整宽度以提高处理速度
    resize_height = 480      # 调整高度以提高处理速度
    alpha = 0.6              # 热图透明度
    
    try:
        # 启动视频流处理
        process_video_stream(
            foreground_hist_path=foreground_hist_path,
            background_hist_path=background_hist_path,
            camera_id=camera_id,
            display_original=display_original,
            display_heatmap=display_heatmap,
            display_overlay=display_overlay,
            display_canny=display_canny,
            canny_threshold1=canny_threshold1,
            canny_threshold2=canny_threshold2,
            display_contour=display_contour,
            contour_threshold_percentile=contour_threshold_percentile,
            resize_width=resize_width,
            resize_height=resize_height,
            alpha=alpha
        )
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {e}")

if __name__ == "__main__":
    main()