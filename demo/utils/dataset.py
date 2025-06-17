import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import staintools  # 用于H&E染色标准化


class MoNuSegDataset(Dataset):
    """MoNuSeg 细胞核分割数据集加载器（适配重组后数据集结构）"""

    def __init__(self, root_dir, mode='train', target_size=(1024, 1024),
                 use_he_norm=True, augment=True):
        """
        Args:
            root_dir: 数据集根目录（包含 train/val/test 子文件夹）
            mode: 'train', 'val' 或 'test'
            target_size: 输出图像尺寸
            use_he_norm: 是否进行H&E染色标准化
            augment: 是否启用数据增强
        """
        self.root_dir = root_dir
        self.mode = mode
        self.target_size = target_size
        self.use_he_norm = use_he_norm
        self.augment = augment

        # 设置路径（适配新的目录结构）
        self.image_dir = os.path.join(root_dir, mode, 'images')
        self.mask_dir = os.path.join(root_dir, mode, 'masks')

        # 获取匹配的图像-掩码对
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])

        # 严格验证匹配
        assert len(self.image_files) == len(
            self.mask_files), f"图像与掩码数量不匹配！{len(self.image_files)} images vs {len(self.mask_files)} masks"
        for img, msk in zip(self.image_files, self.mask_files):
            assert img == msk, f"文件名不匹配: {img} vs {msk}"

        print(f"成功加载 {len(self.image_files)} 对 {mode} 数据")

        # H&E 标准化器初始化（仅训练集需要参考图像）
        if use_he_norm and mode == 'train':
            self.he_normalizer = staintools.StainNormalizer(method='macenko')
            # 随机选择一张训练集图像作为参考
            ref_image_path = os.path.join(self.image_dir, random.choice(self.image_files))
            ref_image = cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB)
            self.he_normalizer.fit(ref_image)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像和掩码
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # 确保RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 调整尺寸（保持长宽比进行padding）
        image, mask = self.pad_resize(image, mask)

        # H&E 染色标准化（仅训练模式）
        if self.use_he_norm and self.mode == 'train':
            image = self.he_normalizer.transform(image)

        # 数据增强（仅训练模式）
        if self.augment and self.mode == 'train':
            image, mask = self.augment_sample(image, mask)

        # 转为Tensor并归一化
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)  # 二值化

        return image, mask

    # 以下方法保持不变...
    def pad_resize(self, image, mask):
        """保持长宽比的resize+padding"""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size

        # 计算缩放比例
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 缩放
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 计算padding
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top = pad_h // 2
        left = pad_w // 2

        # 填充（使用反射填充减少边界伪影）
        image = cv2.copyMakeBorder(image, top, pad_h - top, left, pad_w - left,
                                   cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, top, pad_h - top, left, pad_w - left,
                                  cv2.BORDER_CONSTANT, value=0)
        return image, mask

    def augment_sample(self, image, mask):
        """医学图像专用数据增强"""
        # 随机水平/垂直翻转
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # 随机旋转（90度倍数）
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # 弹性变形（模拟细胞核形变）
        if random.random() > 0.7:
            image, mask = self.elastic_transform(image, mask, alpha=50, sigma=7)

        # 颜色扰动（H&E染色变化）
        if random.random() > 0.5:
            image = self.augment_he_stain(image)

        return image, mask

    def elastic_transform(self, image, mask, alpha, sigma):
        """弹性变形增强"""
        h, w = image.shape[:2]

        # 生成随机位移场
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha

        # 创建网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        # 应用变形
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return image, mask

    def augment_he_stain(self, image):
        """模拟H&E染色变化"""
        # 在HSV空间扰动饱和度
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = hsv[..., 1] * random.uniform(0.8, 1.2)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # 添加光照变化
        image = image * random.uniform(0.9, 1.1)
        return np.clip(image, 0, 255).astype(np.uint8)


def get_loaders(data_dir, batch_size=4, num_workers=4):
    """获取训练、验证和测试数据加载器"""
    train_set = MoNuSegDataset(
        root_dir=data_dir,
        mode='train',
        target_size=(1024, 1024),
        use_he_norm=True,
        augment=True
    )

    val_set = MoNuSegDataset(
        root_dir=data_dir,
        mode='val',
        target_size=(1024, 1024),
        use_he_norm=False,
        augment=False
    )

    test_set = MoNuSegDataset(
        root_dir=data_dir,
        mode='test',
        target_size=(1024, 1024),
        use_he_norm=False,
        augment=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,  # 测试集通常batch_size=1
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载
    train_loader, val_loader, test_loader = get_loaders('E:/zhyl/data_reorganized')

    for images, masks in train_loader:
        print(f"Train Batch images shape: {images.shape}")  # [B, 3, 1024, 1024]
        print(f"Train Batch masks shape: {masks.shape}")  # [B, 1, 1024, 1024]
        break