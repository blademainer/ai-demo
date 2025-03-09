import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    """自定义数据集，用于加载和预处理图像数据"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        """
        初始化数据集
        
        Args:
            data_dir (str): 数据目录路径
            transform (callable, optional): 图像转换操作
            split (str): 数据集类型，'train', 'val', 或 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # 假设数据目录结构为：data_dir/split/class_name/image_files
        self.split_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(self.split_dir):
            raise RuntimeError(f"数据目录 {self.split_dir} 不存在")
        
        # 获取所有类别
        self.classes = sorted(os.listdir(self.split_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集所有图像文件路径和对应的标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.split_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if self._is_image_file(img_name):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        img_path, label = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def _is_image_file(self, filename):
        """检查文件是否为图像文件"""
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        return filename.lower().endswith(IMG_EXTENSIONS)


def get_data_transforms():
    """获取数据转换操作"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transform, 'val': val_transform, 'test': val_transform}


def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """创建数据加载器"""
    transforms_dict = get_data_transforms()
    
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            datasets[split] = ImageDataset(
                data_dir=data_dir,
                transform=transforms_dict[split],
                split=split
            )
            
            shuffle = (split == 'train')
            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
        except RuntimeError as e:
            print(f"无法加载 {split} 数据集: {e}")
    
    return dataloaders, datasets