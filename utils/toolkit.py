import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10, i2l = None):
    assert len(y_pred) == len(y_true), "Data length error."
    if i2l is None:
        i2l = lambda x: x
    all_acc = {}
    all_acc["total"] = np.around(
        (i2l(y_pred) == i2l(y_true)).sum() * 100 / len(y_true), decimals=2
    )
    class_id = 0
    # Grouped accuracy, for initial classes
    for incre_ in increment:
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + incre_)
        )[0]
        if len(idxes)==0:
            class_id+=incre_
            continue
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + incre_ - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (i2l(y_pred)[idxes] == i2l(y_true)[idxes]).sum() * 100 / len(idxes), decimals=2
        )
        class_id += incre_

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (i2l(y_pred)[idxes] == i2l(y_true)[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (i2l(y_pred)[idxes] == i2l(y_true)[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png', 
                         figsize=(10, 8), dpi=300, cmap='Blues', 
                         class_names=None, title='混淆矩阵'):
    """
    绘制并保存带有数值标记的混淆矩阵。
    
    参数:
        y_true: 真实标签数组
        y_pred: 预测标签数组
        save_path: 图像保存路径，默认为'confusion_matrix.png'
        figsize: 图像大小，默认为(10, 8)
        dpi: 图像分辨率，默认为300
        cmap: 热图颜色映射，默认为'Blues'
        class_names: 类别名称列表，默认为None (使用数字索引)
        title: 图表标题，默认为'混淆矩阵'
    
    返回:
        None，但会保存图像并显示图表
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 获取类别数量
    num_classes = cm.shape[0]
    
    # 如果没有提供类别名称，使用数字索引
    if class_names is None:
        class_names = range(num_classes)
    
    # 创建图像
    plt.figure(figsize=figsize)
    
    # 使用seaborn创建热图
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True,
                xticklabels=class_names,
                yticklabels=class_names)
    
    # 添加标签和标题
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title(title)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    # Use environment variable for save path, default to original value
    env_save_path = os.getenv("CONFUSION_MATRIX_PATH", save_path)
    plt.savefig(env_save_path, dpi=dpi, bbox_inches='tight')
    
    # 显示图像
    plt.show()
    
    # 打印分类指标
    print(f"混淆矩阵已保存至: {env_save_path}")
