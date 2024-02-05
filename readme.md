## 基于深度学习的故障诊断

### 第一章：安装以及环境配置

#### 1.Pycharm安装

```
官网: https://www.jetbrains.com/pycharm/
```

#### 2.Anaconda安装与配置

```
官网：https://www.anaconda.com/download#downloads

配置环境变量（按照自己的安装路径进行配置即可）
D:\software\anaconda\install
D:\software\anaconda\install\Scripts
D:\software\anaconda\install\Library\bin
D:\software\anaconda\install\Library\mingw-w64\bin
```

#### 3.Tensorflow安装

```
官网：https://tensorflow.google.cn/?hl=zh-cn

创建深度学习环境并命名为tf2，指定python版本为3.11:
conda create --name tf2 python=3.11

Anaconda激活tf2环境:
conda activate tf2

Anaconda退出激活环境:
conda deactivate


安装Tensorflow并指定版本为2.15，使用豆瓣源进行加速
pip install tensorflow==2.15.0 -i https://pypi.douban.com/simple/

安装scikit-learn，使用豆瓣源进行加速
pip install scikit-learn -i https://pypi.douban.com/simple/
```



### 第二章：代码讲解与运行调试

#### 0.数据集说明

```
凯斯西楚大学轴承数据集官网：
https://engineering.case.edu/bearingdatacenter/download-data-file

数据集说明(翻译于官网数据集说明)：
收集了正常轴承、单点驱动端和风扇端缺陷的数据。驱动端轴承实验以12000个样本/秒和48000个样本/秒的速度采集数据。所有风机端轴承数据以12000个样本/秒的速度采集。

数据文件为Matlab格式。每个文件包含风扇和驱动端振动数据以及电机转速。对于所有文件，变量名中的以下项表示:

数据集包括：
(1)正常基线数据
(2)12k驱动端轴承故障数据
(3)48k传动端轴承故障数据
(4)风机端轴承故障数据

实验数据集仅使用12k驱动端轴承故障数据，其余数据集预处理与实验数据预处理一致。
```



#### 1.基于卷积神经网络的故障诊断（1DCNN、2DCNN）

```
文件夹说明：
data：数据集存放路径
save_picture/1DCNN:存放1DCNN的图片
save_picture/2DCNN:存放2DCNN的图片

model/1DCNN.h5: 保存的1DCNN模型文件
model/2DCNN.h5: 保存的2DCNN模型文件

code/1DCNN.py: 1DCNN代码运行文件
code/2DCNN.py: 2DCNN代码运行文件
code/preprocessing.py: 数据预处理文件
```

#### 2.基于卷积与循环神经网络的故障诊断(1DCNN-LSTM、1DCNN-GRU、2DCNN-LSTM、2DCNN-GRU)

```
文件夹说明：
data：数据集存放路径

save_picture/1DCNN_GRU:存放1DCNN_GRU的图片
save_picture/1DCNN_LSTM:存放1DCNN_LSTM的图片
save_picture/2DCNN_GRU:存放2DCNN_GRU的图片
save_picture/2DCNN_LSTM:存放2DCNN_LSTM的图片


model/1DCNN_GRU.h5: 保存的1DCNN_GRU模型文件
model/1DCNN_LSTM.h5: 保存的1DCNN_LSTM模型文件
model/2DCNN_GRU.h5: 保存的2DCNN_GRU模型文件
model/2DCNN_LSTM.h5: 保存的2DCNN_LSTM模型文件

code/1DCNN_GRU.py: 1DCNN_GRU代码运行文件
code/1DCNN_LSTM.py: 1DCNN_LSTM代码运行文件
code/2DCNN_GRU.py: 2DCNN_GRU代码运行文件
code/2DCNN_LSTM.py: 2DCNN_LSTM代码运行文件
code/preprocessing.py: 数据预处理文件
```

#### 3.基于卷积与连续小波变换的故障诊断(1DCNN-CWT、2DCNN-CWT)

```
1.运行时注意事项：对于pywt库的安装，应使用：
pip install PyWavelets -i https://pypi.douban.com/simple/

2.首先在创建cwt_picture-train-valid-test文件夹，并运行sign_cwt文件，在train、valid、test文件夹生成相应的连续小波变换的图片后，再运行1DCNN_CWT/2DCNN_CWT进行分类实验  

文件夹说明：
data：数据集存放路径

cwt_picture/train: 存放训练集的连续小波图片
cwt_picture/test: 存放测试集的连续小波图片
cwt_picture/valid: 存放验证集的连续小波图片

save_picture/1DCNN_CWT:存放1DCNN_CWT的图片
save_picture/2DCNN_CWT:存放2DCNN_CWT的图片

model/1DCNN_CWT.h5: 保存的1DCNN_CWT模型文件
model/2DCNN_CWT.h5: 保存的2DCNN_CWT模型文件

code/1DCNN_CWT.py: 1DCNN_CWT代码运行文件
code/2DCNN_CWT.py: 2DCNN_CWT代码运行文件
code/gen_cwt_pic.py: 生成连续小波变换图片代码运行文件
code/read_picture.py: 读取连续小波图片代码运行文件
code/preprocessing.py: 数据预处理文件
```

#### 4.基于特征融合的故障诊断（SIGN-FFT）

```
文件夹说明：
data：数据集存放路径

save_picture/1DCNN_SIGN_FFT:存放1DCNN_SIGN_FFT的图片

model/1DCNN_SIGN_FFT.h5: 保存的1DCNN_SIGN_FFT模型文件

code/1DCNN_SIGN_FFT.py: 1DCNN_SIGN_FFT代码运行文件
code/preprocessing.py: 数据预处理文件
```

#### 5.基于抗噪方法的故障诊断（1DCNN、2DCNN-DRSN、1DCNN-SVD）

```
文件夹说明：
data：数据集存放路径

save_picture/1DCNN:存放1DCNN的图片
save_picture/1DCNN_SVD:存放1DCNN_SVD的图片
save_picture/2DCNN_DRSN:存放2DCNN_DRSN的图片

model/1DCNN.h5: 保存的1DCNN模型文件
model/1DCNN_SVD.h5: 保存的1DCNN_SVD模型文件
model/2DCNN_DRSN.h5: 保存的2DCNN_DRSN模型文件

code/1DCNN.py: 1DCNN代码运行文件
code/1DCNN_SVD.py: 1DCNN_SVD代码运行文件
code/2DCNN_DRSN.py: 2DCNN_DRSN代码运行文件
code/plot_svd.py: 因为可能需要图片，就可视化一条样本，用于样本分析
code/preprocessing.py: 数据预处理文件
```

#### 6.基于迁移学习的故障诊断（模型）

```
文件夹说明：
data：数据集存放路径

save_picture/1DCNN_Transfer:存放1DCNN_Transfer迁移的图片

model/1DCNN_Transfer.h5: 保存的1DCNN_Transfer模型文件

code/1DCNN_Transfer.py: 1DCNN_Transfer代码运行文件
code/preprocessing.py: 数据预处理文件
```

#### 7.基于通用模型代码的故障诊断（GRU、Inception、LSTM、RandomForest、SVM）

```
文件夹说明：
data：数据集存放路径

save_picture/GRU:存放GRU的图片
save_picture/Inception:存放Inception的图片
save_picture/LSTM:存放LSTM的图片

model/GRU.h5: 保存的GRU模型文件
model/Inception.h5: 保存的Inception模型文件
model/LSTM.h5: 保存的LSTM模型文件

code/preprocessing.py: 数据预处理文件
code/GRU.py: GRU代码运行文件
code/Inception.py: Inception代码运行文件
code/LSTM.py: LSTM代码运行文件
code/RandomForest.py: RandomForest代码运行文件
code/SVM.py: SVM代码运行文件
```











