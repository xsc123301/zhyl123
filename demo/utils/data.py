import kagglehub

# Download latest version
path = kagglehub.dataset_download("tuanledinh/monuseg2018")

print("Path to dataset files:", path)

from sklearn.model_selection import train_test_split
import os
import shutil


def reorganize_dataset(data_root='E:/zhyl/data'):
    # 合并所有样本
    all_images = []
    for dataset in ['kmms_training', 'kmms_test']:
        img_dir = os.path.join(data_root, dataset, 'images')
        all_images.extend([os.path.join(dataset, 'images', f) for f in os.listdir(img_dir)])

    # 按8:1:1划分
    train, val_test = train_test_split(all_images, test_size=0.3, random_state=42)
    val, test = train_test_split(val_test, test_size=0.3, random_state=42)

    # 创建新目录结构
    new_structure = {
        'train': {'images': [], 'masks': []},
        'val': {'images': [], 'masks': []},
        'test': {'images': [], 'masks': []}
    }

    # 重新组织文件路径
    for split, files in zip(['train', 'val', 'test'], [train, val, test]):
        for img_path in files:
            dataset, _, img_name = img_path.split(os.sep)
            mask_path = img_path.replace('images', 'masks')

            new_structure[split]['images'].append((dataset, img_name))
            new_structure[split]['masks'].append((dataset, img_name))

    # 创建新目录并复制文件（示例代码）
    for split in new_structure:
        os.makedirs(f'E:/zhyl/data_reorganized/{split}/images', exist_ok=True)
        os.makedirs(f'E:/zhyl/data_reorganized/{split}/masks', exist_ok=True)

        for dataset, img_name in new_structure[split]['images']:
            src = f'E:/zhyl/data/{dataset}/images/{img_name}'
            dst = f'E:/zhyl/data_reorganized/{split}/images/{img_name}'
            shutil.copy(src, dst)

        for dataset, mask_name in new_structure[split]['masks']:
            src = f'E:/zhyl/data/{dataset}/masks/{mask_name}'
            dst = f'E:/zhyl/data_reorganized/{split}/masks/{mask_name}'
            shutil.copy(src, dst)


# 执行重组
reorganize_dataset()


import os
from PIL import Image


# def convert_tif_to_png(input_dir, output_dir):
#     """
#     将指定目录下的TIFF(.tif)图片转换为PNG格式
#
#     参数:
#         input_dir (str): 包含TIFF图片的输入目录路径 (如 'kmms_training/images')
#         output_dir (str): 保存PNG图片的输出目录路径
#     """
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 遍历输入目录中的所有文件
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
#             # 构建完整的输入和输出路径
#             input_path = os.path.join(input_dir, filename)
#             output_filename = os.path.splitext(filename)[0] + '.png'
#             output_path = os.path.join(output_dir, output_filename)
#
#             try:
#                 # 打开TIFF图像并转换为PNG
#                 with Image.open(input_path) as img:
#                     img.save(output_path, 'PNG')
#                 print(f"转换成功: {filename} -> {output_filename}")
#             except Exception as e:
#                 print(f"转换失败 {filename}: {str(e)}")
#
#
# # 使用示例
# if __name__ == "__main__":
#     input_directory = "E:/zhyl/data/kmms_test/images"
#     output_directory = "E:/zhyl/data/kmms_test/images_png"  # 你可以修改为你想保存的目录
#
#     convert_tif_to_png(input_directory, output_directory)

import os
import shutil

# 处理训练集掩码
# mask_dir = "E:/zhyl/data/kmms_training/masks"
# for f in os.listdir(mask_dir):
#     if ' .png' in f:  # 包含空格的png文件
#         new_name = f.replace(' .png', '.png')  # 去掉空格
#         shutil.move(os.path.join(mask_dir, f),
#                    os.path.join(mask_dir, new_name))

# 处理测试集掩码
# mask_dir = "E:/zhyl/data/kmms_test/masks"
# for f in os.listdir(mask_dir):
#     if ' .png' in f:  # 两个空格的png文件
#         new_name = f.replace(' .png', '.png')  # 去掉空格
#         shutil.move(os.path.join(mask_dir, f),
#                    os.path.join(mask_dir, new_name))

# import os
# import shutil
#
# # 修复测试集图像文件名（去掉空格）
# image_dir = "E:/zhyl/data/kmms_test/masks"
# for f in os.listdir(image_dir):
#     if ' ' in f and f.endswith('.png'):
#         new_name = f.replace(' ', '')
#         shutil.move(os.path.join(image_dir, f),
#                    os.path.join(image_dir, new_name))
#         print(f"重命名: {f} -> {new_name}")