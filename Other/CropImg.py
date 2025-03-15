from PIL import Image
import os

def crop_images_in_folder(folder_path, x, y, w, h):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)

            # 裁剪图片
            cropped_image = image.crop((x, y, x+w, y+h))

            # 保存裁剪后的图片到原路径
            cropped_image.save(image_path)

folder_path = 'F:/Paper/deeplearning/data'  # 替换为你的文件夹路径
x = 360  # 设置裁剪起始点的横坐标
y = 290  # 设置裁剪起始点的纵坐标
w = 1300  # 设置裁剪区域的宽度
h = 890  # 设置裁剪区域的高度

crop_images_in_folder(folder_path, x, y, w, h)
