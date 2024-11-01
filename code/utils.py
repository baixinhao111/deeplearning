import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
    
import tarfile

import os
if not os.path.exists('../dataset/cifar-10-batches-py'):  # 检测路径下是否已存在文件，若否，则执行以下代码
    os.mkdir('data')  # 创建数据存储路径，即data文件夹
    with tarfile.open('/public/data/image/cifar10/cifar-10-batches-py.tar.gz') as tar:  # 矩池云上内置CIFAR10数据集压缩文件的存储路径
        tar.extractall(path='data')  # 将数据集压缩文件解压，并保存至data文件夹中
        
import torchvision.transforms as transforms
import torchvision

def get_cifar10(batch_size = 32):

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 变形为AlexNet所需的输入形状（ 32 * 32）
        transforms.ToTensor(),  # 转换为tensor（注意，此处的tensor默认在CPU上储存）
    ])

    train_set = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                batch_size=batch_size, shuffle=True,num_workers=12)  # 进行打乱

    val_set = torchvision.datasets.CIFAR10(root='../dataset',
                                train=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set,
                                batch_size=batch_size, shuffle=False,num_workers=12) 
# 不进行打乱

    return train_loader,val_loader







# 计算模型预测准确率
def accuracy(outputs, labels):
    preds = torch.max(outputs, dim=1)[1]  # 获取预测类别
    return torch.sum(preds == labels).item() / len(preds)  # 计算准确率

# 模型验证
def validate(model, val_loader):
    
    val_loss = 0
    val_acc = 0
    
    model.eval()
    
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将tensor切换到GPU存储模式
        outputs = model(inputs)  # 计算模型输出
        
        loss = F.cross_entropy(outputs, labels)  # 计算交叉熵损失函数
        val_loss += loss.item()  # item方法提取出tensor中的数字
        
        acc = accuracy(outputs, labels)  # 计算准确率
        val_acc += acc
    
    val_loss /= len(val_loader)  # 计算平均损失
    val_acc /= len(val_loader)  # 计算平均准确率
    
    return val_loss, val_acc

# 打印训练结果
def print_log(epoch, train_time, train_loss, train_acc, val_loss, val_acc,epochs = 10):
    print(f"Epoch [{epoch+1}/{epochs}], time: {train_time:.2f}s, loss: {train_loss:.4f}, acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
    
# 模型训练
import time
def train(model,optimizer, train_loader, val_loader, epochs=1):
    
    train_losses = [];train_accs = [];
    val_losses = [];val_accs = [];
    
    model.train()
    
    for epoch in range(epochs):
        
        train_loss = 0
        train_acc = 0
        
        start = time.time()  # 记录本epoch开始时间
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将tensor切换到GPU存储模式

            optimizer.zero_grad()  # 将模型所有参数tensor的梯度变为0（否则之后计算的梯度会与先前存在的梯度叠加）

            outputs = model(inputs)  # 计算模型输出

            loss = F.cross_entropy(outputs, labels)  # 计算交叉熵损失函数
            train_loss += loss.item()  # item方法提取出tensor中的数字

            acc = accuracy(outputs, labels)  # 计算准确率
            train_acc += acc

            loss.backward()  # 调用PyTorch的autograd自动求导功能，计算loss相对于模型各参数的导数
            optimizer.step()  # 根据模型中各参数相对于loss的导数，以及指定的学习率，更新参数
        
        end = time.time()  # 记录本epoch结束时间
        train_time = end - start  # 计算本epoch的训练耗时
        
        train_loss /= len(train_loader)  # 计算平均损失
        train_acc /= len(train_loader)  # 计算平均准确率
        
        val_loss, val_acc = validate(model, val_loader)  # 计算测试集上的损失函数和准确率
        
        train_losses.append(train_loss);train_accs.append(train_acc)
        val_losses.append(val_loss);val_accs.append(val_acc)
        
        print_log(epoch, train_time, train_loss, train_acc, val_loss, val_acc,epochs = epochs)  # 打印训练结果
    return train_losses,train_accs,val_losses,val_accs


# 模型预测
def predict(model, inputs):
    outputs = model(inputs.to(device))
    preds = torch.max(outputs, dim=1)[1]
    return preds