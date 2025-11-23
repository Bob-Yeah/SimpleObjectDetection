import cv2
import numpy as np
import os
import glob

class ImageMaskGenerator:
    def __init__(self):
        self.points = []
        self.current_image = None
        self.original_image = None
        self.image_path = None
        self.mask_output_dir = "masks"
        self.image_list = []
        self.current_index = 0
        
        # 创建输出目录
        if not os.path.exists(self.mask_output_dir):
            os.makedirs(self.mask_output_dir)
    
    def load_images(self, folder_path):
        """加载指定文件夹中的所有图片"""
        # 支持的图片格式
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
        self.image_list = sorted(image_paths)
        if not self.image_list:
            print("没有找到图片文件")
            return False
        
        print(f"找到 {len(self.image_list)} 张图片")
        return True
    
    def load_image(self, index):
        """加载指定索引的图片"""
        if 0 <= index < len(self.image_list):
            self.image_path = self.image_list[index]
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                print(f"无法加载图片: {self.image_path}")
                return False
            
            self.current_image = self.original_image.copy()
            self.points = []
            print(f"加载图片 {index + 1}/{len(self.image_list)}: {os.path.basename(self.image_path)}")
            return True
        return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，处理点击事件"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 添加点击点
            self.points.append((x, y))
            
            # 在图像上标记点
            cv2.circle(self.current_image, (x, y), 3, (0, 255, 0), -1)
            
            # 连接相邻点
            if len(self.points) > 1:
                cv2.line(self.current_image, self.points[-2], self.points[-1], (0, 255, 0), 2)
            
            cv2.imshow("Image", self.current_image)
    
    def create_mask(self):
        """创建mask，勾画区域为255，其他为0"""
        if len(self.points) < 3:
            print("需要至少3个点来创建闭合区域")
            return None
        
        # 创建一个全0的mask
        mask = np.zeros_like(self.original_image[:, :, 0])
        
        # 填充多边形区域
        pts = np.array(self.points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        
        return mask
    
    def save_mask(self, mask):
        """保存mask文件"""
        if mask is None:
            return False
        
        # 生成保存路径
        base_name = os.path.basename(self.image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        mask_path = os.path.join(self.mask_output_dir, f"{name_without_ext}_mask.png")
        
        # 保存mask
        if cv2.imwrite(mask_path, mask):
            print(f"Mask已保存到: {mask_path}")
            return True
        else:
            print(f"无法保存mask: {mask_path}")
            return False
    
    def process_image(self):
        """处理当前图像"""
        # 创建窗口并设置鼠标回调
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)
        
        while True:
            cv2.imshow("Image", self.current_image)
            key = cv2.waitKey(1) & 0xFF
            
            # 按 's' 键闭合区域并保存mask
            if key == ord('s'):
                if len(self.points) < 3:
                    print("需要至少3个点来创建闭合区域")
                    continue
                
                # 闭合区域
                cv2.line(self.current_image, self.points[-1], self.points[0], (0, 255, 0), 2)
                cv2.imshow("Image", self.current_image)
                
                # 创建并保存mask
                mask = self.create_mask()
                self.save_mask(mask)
                break
            
            # 按 'r' 键重置
            elif key == ord('r'):
                self.current_image = self.original_image.copy()
                self.points = []
                cv2.imshow("Image", self.current_image)
                print("已重置，重新选择点")
            
            # 按 'n' 键下一张图片
            elif key == ord('n'):
                break
            
            # 按 'q' 键退出
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def run(self, folder_path):
        """运行主程序"""
        if not self.load_images(folder_path):
            return
        
        while self.current_index < len(self.image_list):
            if not self.load_image(self.current_index):
                self.current_index += 1
                continue
            
            print("使用指南:")
            print("1. 点击鼠标选择点")
            print("2. 按 's' 键闭合区域并保存mask")
            print("3. 按 'r' 键重置当前图片的选择")
            print("4. 按 'n' 键处理下一张图片")
            print("5. 按 'q' 键退出程序")
            
            if not self.process_image():
                break
            
            self.current_index += 1
        
        print("所有图片处理完成")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图片区域勾画mask生成工具')
    parser.add_argument('folder', nargs='?', default='.', help='图片文件夹路径 (默认: 当前目录)')
    args = parser.parse_args()
    
    generator = ImageMaskGenerator()
    generator.run(args.folder)


if __name__ == "__main__":
    main()