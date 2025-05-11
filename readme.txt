数据集及预训练参数：

通过网盘分享的文件：辅修毕设
链接: https://pan.baidu.com/s/1W3buvInpct3GJUx7qMwQ7A?pwd=3spn 提取码: 3spn 
--来自百度网盘超级会员v5的分享

1、作者的实验环境如下：

PyTorch  1.10.0
Python  3.8(ubuntu20.04)
CUDA  11.3
GPU  RTX 4090(24GB) * 1
CPU  16 vCPU Intel(R) Xeon(R) Gold 6430

一般需要再pip install timm scikit-learn easydict这几个库

2、预定义超参数

在实验前请根据您的显存容量修改EasyDict中的batch_size、epochs，填写root_path路径，以及第219行的model_name。

具体model_name列表可查询timm官方库。

3、加载预训练参数

直接运行main.py，即可从零开始训练train中图像并以test中图像评估模型性能。

若想加载官方预训练参数，需将第220行代码注释，替换为第222-227行代码，同时将第227行file路径填写为预训练参数文件路径，如（1_tiny.npz、2_small.npz、3_base.bin）

若想下载其他模型的预训练参数文件，可通过如下代码打印模型信息，并通过打印列表中的url地址下载：

model = timm.create_model('vit_tiny_patch16_224')
print(model.default_cfg)