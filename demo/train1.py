from matplotlib import pyplot as plt
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm

from model.deep import DeepLabv3Plus
from utils.dataset import MoNuSegDataset, get_loaders  # 修改导入
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime
from torch.nn import functional as F
import time

def plot_loss_curve(train_loss, val_loss=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('E:/zhyl/results_1/loss_curve.png')
    plt.close()

def plot_metrics(train_metrics, val_metrics, save_dir):
    """绘制并保存训练曲线"""
    os.makedirs(save_dir, exist_ok=True)
    metrics = ['dice', 'iou', 'precision', 'recall']

    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(train_metrics[metric], label=f'Train {metric}')
        plt.plot(val_metrics[metric], label=f'Validation {metric}')
        plt.title(f'Training and Validation {metric.upper()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{metric}_curve.png'))
        plt.close()


def train_net(net, device, data_path, epochs=40, batch_size=4, lr=0.00001):
    # 创建结果目录
    results_dir = 'E:/zhyl/results_1'
    os.makedirs(results_dir, exist_ok=True)

    # 加载数据集 - 使用新的get_loaders函数
    train_loader, val_loader, test_loader = get_loaders(
        data_path,
        batch_size=batch_size,
        num_workers=4
    )

    # 定义优化器和学习率调度器
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = {
        'plateau': ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True),
        'cosine': CosineAnnealingLR(optimizer, T_max=epochs // 2, eta_min=1e-6)
    }
    criterion = DiceFocalLoss(alpha=0.7)
    scaler = GradScaler()

    # 初始化记录器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(results_dir, 'logs', timestamp))

    # 训练变量初始化
    best_dice = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': {'dice': [], 'iou': [], 'precision': [], 'recall': []},
        'val_metrics': {'dice': [], 'iou': [], 'precision': [], 'recall': []}
    }

    for epoch in range(epochs):
        start_time = time.time()
        net.train()
        epoch_train_loss = 0.0
        train_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': []}

        # 使用tqdm进度条
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

            for image, label in tepoch:
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                # 混合精度训练
                with autocast():
                    pred = net(image)
                    loss = criterion(pred, label)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 记录指标
                epoch_train_loss += loss.item()
                batch_metrics = calculate_metrics(pred.detach(), label)
                for k in train_metrics:
                    train_metrics[k].append(batch_metrics[k])

                tepoch.set_postfix(loss=loss.item(), dice=batch_metrics['dice'])

        # 计算平均训练指标
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}

        # 验证阶段
        val_loss, val_metrics = validate_model(net, val_loader, criterion, device)

        # 更新学习率
        scheduler['plateau'].step(val_metrics['dice'])
        scheduler['cosine'].step()

        # 记录历史数据
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        for k in avg_train_metrics:
            history['train_metrics'][k].append(avg_train_metrics[k])
            history['val_metrics'][k].append(val_metrics[k])

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for k in avg_train_metrics:
            writer.add_scalar(f'{k}/train', avg_train_metrics[k], epoch)
            writer.add_scalar(f'{k}/val', val_metrics[k], epoch)

        # 打印epoch结果
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1} Summary ({epoch_time:.1f}s):")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Dice: {avg_train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型和最后模型
        model_dir = os.path.join(results_dir, 'saved_models', timestamp)
        os.makedirs(model_dir, exist_ok=True)

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'config': {'backbone': 'resnet101', 'num_classes': 1}
            }, os.path.join(model_dir, f'best_model_dice{best_dice:.4f}.pth'))
            print(f"Saved new best model with Dice: {best_dice:.4f}")

        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(model_dir, f'checkpoint_epoch{epoch + 1}.pth'))

    # 训练结束后保存最终模型
    torch.save(net.state_dict(), os.path.join(model_dir, 'final_model.pth'))

    # 绘制并保存训练曲线
    plot_loss_curve(history['train_loss'], history['val_loss'])
    plot_metrics(history['train_metrics'], history['val_metrics'], results_dir)

    writer.close()

    # 在测试集上评估最佳模型
    print("\nEvaluating on test set...")
    net.load_state_dict(torch.load(os.path.join(model_dir, f'best_model_dice{best_dice:.4f}.pth'))['model_state_dict'])
    test_loss, test_metrics = validate_model(net, test_loader, criterion, device)

    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f} | Test Dice: {test_metrics['dice']:.4f}")
    print(
        f"Test IoU: {test_metrics['iou']:.4f} | Test Precision: {test_metrics['precision']:.4f} | Test Recall: {test_metrics['recall']:.4f}")

    return net, history


def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': []}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            batch_metrics = calculate_metrics(outputs, labels)
            for k in metrics:
                metrics[k].append(batch_metrics[k])

    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return val_loss / len(dataloader), avg_metrics


def calculate_metrics(preds, targets, threshold=0.5):
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


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        # Dice Loss
        preds_sigmoid = torch.sigmoid(preds)
        intersection = (preds_sigmoid * targets).sum()
        union = preds_sigmoid.sum() + targets.sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)

        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()

        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DeepLabv3Plus(num_classes=1, backbone='resnet101').to(device=device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")

    data_path = "E:/zhyl/data_reorganized"
    trained_net, training_history = train_net(
        net,
        device,
        data_path,
        epochs=100,
        batch_size=4,
        lr=1e-4
    )