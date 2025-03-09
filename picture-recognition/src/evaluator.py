import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
from torchvision import transforms

class Evaluator:
    """模型评估器"""
    
    def __init__(self, model, dataloader, device=None, classes=None):
        """
        初始化评估器
        
        Args:
            model: 待评估的模型
            dataloader: 数据加载器
            device: 评估设备
            classes: 类别名称列表
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = classes
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self):
        """评估模型性能"""
        all_preds = []
        all_labels = []
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = running_corrects.double() / total_samples
        print(f'测试准确率: {accuracy:.4f}')
        
        return {
            'accuracy': accuracy.item(),
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
    
    def plot_confusion_matrix(self, save_path='./results/confusion_matrix.png'):
        """绘制混淆矩阵"""
        results = self.evaluate()
        
        cm = confusion_matrix(results['labels'], results['predictions'])
        plt.figure(figsize=(10, 8))
        
        if self.classes:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        
        # 保存图表
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()
    
    def print_classification_report(self):
        """打印分类报告"""
        results = self.evaluate()
        
        if self.classes:
            report = classification_report(results['labels'], results['predictions'], target_names=self.classes)
        else:
            report = classification_report(results['labels'], results['predictions'])
        
        print('分类报告:')
        print(report)
        
        return report


class Predictor:
    """图像预测器"""
    
    def __init__(self, model, device=None, classes=None, transform=None):
        """
        初始化预测器
        
        Args:
            model: 预测模型
            device: 预测设备
            classes: 类别名称列表
            transform: 图像转换操作
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = classes
        
        # 默认转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()
    
    def predict_image(self, image_path):
        """预测单张图像"""
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 应用转换
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted_idx = torch.max(outputs, 1)
        
        # 获取预测结果
        predicted_idx = predicted_idx.item()
        probability = probabilities[predicted_idx].item()
        
        if self.classes:
            predicted_class = self.classes[predicted_idx]
        else:
            predicted_class = str(predicted_idx)
        
        return {
            'class_id': predicted_idx,
            'class_name': predicted_class,
            'probability': probability,
            'probabilities': probabilities.cpu().numpy()
        }
    
    def predict_batch(self, image_paths):
        """批量预测图像"""
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append(result)
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """可视化预测结果"""
        # 预测图像
        result = self.predict_image(image_path)
        
        # 加载原始图像
        image = Image.open(image_path).convert('RGB')
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 显示图像
        ax1.imshow(image)
        ax1.set_title(f"预测: {result['class_name']}\n概率: {result['probability']:.4f}")
        ax1.axis('off')
        
        # 显示概率条形图
        if self.classes:
            class_names = self.classes
        else:
            class_names = [str(i) for i in range(len(result['probabilities']))]
        
        # 获取前5个最高概率
        top_k = min(5, len(result['probabilities']))
        top_indices = np.argsort(result['probabilities'])[-top_k:]
        top_probs = result['probabilities'][top_indices]
        top_classes = [class_names[i] for i in top_indices]
        
        # 反转顺序，使最高概率在顶部
        top_indices = top_indices[::-1]
        top_probs = top_probs[::-1]
        top_classes = top_classes[::-1]
        
        ax2.barh(range(top_k), top_probs, color='skyblue')
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels(top_classes)
        ax2.set_xlabel('概率')
        ax2.set_title('预测概率 (Top-5)')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()
        
        return result