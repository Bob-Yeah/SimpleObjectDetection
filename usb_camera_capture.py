import cv2
import os
import datetime

def ensure_directory_exists(directory):
    """确保指定的目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def main():
    # 设置图像保存目录
    save_directory = "images"
    ensure_directory_exists(save_directory)
    
    print("USB摄像头拍照程序")
    print("按 's' 键拍照并保存")
    print("按 'q' 键退出程序")
    
    # 打开USB摄像头
    # 通常0是默认的摄像头设备ID，如果有多个摄像头，可能需要尝试不同的ID
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    try:
        # 设置摄像头分辨率（可选）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            # 读取一帧图像
            ret, frame = cap.read()
            
            # 检查是否成功读取图像
            if not ret:
                print("错误：无法获取图像帧")
                break
            
            
            # 显示图像
            cv2.imshow('USB Camera', frame)
            
            # 等待按键
            key = cv2.waitKey(1) & 0xFF
            
            # 按下's'键拍照
            if key == ord('s'):
                try:
                    # 生成带时间戳的文件名
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    # 先使用.bin后缀保存
                    temp_filename = os.path.join(save_directory, f"capture_{timestamp}.bin")
                    # 最终的.jpg路径
                    final_filename = os.path.join(save_directory, f"capture_{timestamp}.jpg")
                    
                    # 使用内存缓冲区的方式保存图像
                    # 先将图像编码为JPEG格式的内存数据
                    success, buffer = cv2.imencode('.jpg', frame)
                    if not success:
                        print("错误：无法将图像编码为JPEG格式")
                        continue
                    
                    # 将内存中的JPEG数据写入到.bin文件
                    with open(temp_filename, 'wb') as f:
                        f.write(buffer)
                    
                    print(f"图像已保存: {temp_filename}")
                    # 在图像上显示保存成功的提示
                    temp_frame = frame.copy()
                    cv2.imshow('USB Camera', temp_frame)
                    cv2.waitKey(500)  # 显示0.5秒
                except Exception as e:
                    print(f"保存图像过程中发生错误: {str(e)}")
                    # 清理可能的临时文件
                    if 'temp_filename' in locals() and os.path.exists(temp_filename):
                        try:
                            os.remove(temp_filename)
                        except:
                            pass
            
            # 按下'q'键退出
            elif key == ord('q'):
                print("程序退出")
                break
    
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    finally:
        # 释放摄像头和关闭所有窗口
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()