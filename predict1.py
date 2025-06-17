import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import argparse
from torch.cuda.amp import autocast

from model.deep import DeepLabv3Plus
from utils.dataset import get_loaders


def calculate_metrics(preds, targets, threshold=0.5):
    """计算评价指标"""
    preds = torch.sigmoid(preds)
    preds_bin = (preds > threshold).float()
    targets_bin = targets

    intersection = (preds_bin * targets_bin).sum()
    union = preds_bin.sum() + targets_bin.sum()

    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    precision = (intersection + 1e-6) / (preds_bin.sum() + 1e-6)
    recall = (intersection + 1e-6) / (targets_bin.sum() + 1e-6)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


def predict_single_image(model, image_path, device, target_size=(1024, 1024)):
    """对单张图像进行预测"""
    model.eval()
    
    # 加载图像
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # 预处理
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        with autocast():
            pred = model(image_tensor)
            pred = torch.sigmoid(pred)
    
    # 后处理
    pred_np = pred.squeeze().cpu().numpy()
    pred_resized = cv2.resize(pred_np, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    
    return pred_resized, image


def create_overlay(image, mask, alpha=0.6):
    """创建叠加效果"""
    # 创建彩色掩码
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0.5] = [255, 0, 0]  # 红色
    
    # 叠加
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    return overlay


def evaluate_model(model, test_loader, device, save_dir):
    """评估模型在测试集上的表现"""
    model.eval()
    
    total_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': []}
    
    print("开始评估测试集...")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="评估进度")):
            images = images.to(device)
            masks = masks.to(device)
            
            with autocast():
                outputs = model(images)
                batch_metrics = calculate_metrics(outputs, masks)
            
            # 累积指标
            for k in total_metrics:
                total_metrics[k].append(batch_metrics[k])
            
            # 保存预测结果
            for i in range(images.size(0)):
                # 获取预测结果
                pred = torch.sigmoid(outputs[i]).cpu().numpy().squeeze()
                mask = masks[i].cpu().numpy().squeeze()
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                
                # 反归一化图像
                image = (image * 255).astype(np.uint8)
                
                # 创建二值化预测
                pred_binary = (pred > 0.5).astype(np.uint8) * 255
                
                # 创建叠加图
                overlay = create_overlay(image, pred_binary)
                
                # 保存结果
                sample_idx = batch_idx * images.size(0) + i
                
                # 保存原图
                plt.imsave(os.path.join(save_dir, f'{sample_idx:02d}_original.png'), image)
                
                # 保存真实掩码
                plt.imsave(os.path.join(save_dir, f'{sample_idx:02d}_ground_truth.png'), mask, cmap='gray')
                
                # 保存预测掩码
                plt.imsave(os.path.join(save_dir, f'{sample_idx:02d}_prediction.png'), pred, cmap='gray')
                
                # 保存叠加图
                plt.imsave(os.path.join(save_dir, f'{sample_idx:02d}_overlay.png'), overlay)
                
                # 保存二值化预测
                plt.imsave(os.path.join(save_dir, f'{sample_idx:02d}_pred_binary.png'), pred_binary, cmap='gray')
    
    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
    
    return avg_metrics


def create_comparison_grid(save_dir, num_samples=8):
    """创建对比网格图"""
    # 获取所有图像文件
    original_files = sorted([f for f in os.listdir(save_dir) if f.endswith('_original.png')])
    
    if len(original_files) == 0:
        print("没有找到预测结果文件")
        return
    
    # 选择前num_samples个样本
    selected_files = original_files[:min(num_samples, len(original_files))]
    
    # 创建网格
    fig, axes = plt.subplots(len(selected_files), 4, figsize=(16, 4*len(selected_files)))
    if len(selected_files) == 1:
        axes = axes.reshape(1, -1)
    
    for i, filename in enumerate(selected_files):
        base_name = filename.replace('_original.png', '')
        
        # 加载图像
        original = plt.imread(os.path.join(save_dir, f'{base_name}_original.png'))
        ground_truth = plt.imread(os.path.join(save_dir, f'{base_name}_ground_truth.png'))
        prediction = plt.imread(os.path.join(save_dir, f'{base_name}_prediction.png'))
        overlay = plt.imread(os.path.join(save_dir, f'{base_name}_overlay.png'))
        
        # 显示图像
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(ground_truth, cmap='gray')
        axes[i, 1].set_title('truth_mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(prediction, cmap='gray')
        axes[i, 2].set_title('pred_mask')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='模型预测和评估')
    parser.add_argument('--model_path', type=str, default='E:/zhyl/results/saved_models/20250617_164455/best_model_dice0.7901.pth', help='模型权重文件路径')
    parser.add_argument('--data_path', type=str, default='E:/zhyl/data_reorganized', help='数据集路径')
    parser.add_argument('--save_dir', type=str, default='E:/zhyl/results/predictions', help='结果保存路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    model = DeepLabv3Plus(num_classes=1, backbone='resnet101').to(device)
    
    # 加载权重
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("模型加载完成")
    
    # 加载测试数据
    print("加载测试数据...")
    _, _, test_loader = get_loaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # 评估模型
    metrics = evaluate_model(model, test_loader, device, args.save_dir)
    
    # 打印结果
    print("\n" + "="*50)
    print("测试集评估结果:")
    print("="*50)
    print(f"Dice系数: {metrics['dice']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print("="*50)
    
    # 创建对比网格
    print("创建对比网格图...")
    create_comparison_grid(args.save_dir)
    
    print(f"\n结果已保存到: {args.save_dir}")
    print("包含以下文件:")
    print("- *_original.png: 原图")
    print("- *_ground_truth.png: 真实掩码")
    print("- *_prediction.png: 预测掩码")
    print("- *_overlay.png: 叠加结果")
    print("- *_pred_binary.png: 二值化预测")
    print("- comparison_grid.png: 对比网格图")


if __name__ == "__main__":
    main()