# 2025CCF-网易雷火联合基金课题数据库
## 项目说明
本仓库为“基于人机协作的角色表情动画重定向”项目的实现，对所提供的30段基于MetaHuman绑定系统的表情动画进行重定向到少于70维度控制器的自定义头模面部。为了满足项目设置的控制器维度数目不超过70的要求，我们选择目标头模其中的45个控制器属性对其面部进行控制，目标头模文件位于下方网盘链接中，使用的45个控制器名称及属性信息位于`/Asset/mery_rig.txt`

## 实现方案
项目使用具有空间特征的MediaPipe生成的landmark图片作为中间量，实现MetaHuman图片 --> MetaHuman landmark image --> Alignment Module --> Target landmark image --> Target Controller Value的预测过程，项目的代码部分包括Alignment Module的实现(对应于step1_prepare.py)，目标头模landmark图片到控制器参数的映射使用ConvNeXt-Tiny作为backbone进行搭建，训练数据集为使用blendshape绑定系统下的真人动捕数据集，使用手动制作目标头模的52个基底表情间接得到粗糙的目标绑定系统下的数据集。

## 参考文件下载
从以下网盘链接中下载 `https://pan.baidu.com/s/1vMqURBKHt-PMjUVLb46FvQ?pwd=xp7t`
1. 目标头模文件`Mery.mb` 
2. pretrained_retargeting.pth并将该预训练参数放入到项目的`Model/pretrained_retargeting.pth`


## Demo
`/Demo`中包含30段课题提供的测试数据及对应的重定向结果视频，每个视频为1024*512尺寸，其中左侧为原始提供的MetaHuman动画，右侧为在自定义头模上的表情重定向结果

## Inference
### 1. 数据准备阶段
将待重定向的原始MetaHuman驱动表情动画逐帧提取为512*512尺寸的图片，并使用`%06d.jpg`的格式按序存储在`/Code/data/source`文件夹中

### 2. 模型推理阶段
```bash
python ./Code/step1_prepare.py
python ./Code/step2_inference.py
python ./Code/step3_postprocess.py
```

### 3. 结果渲染
在maya中导入本项目使用的目标头模文件`/Asset/Mery.mb`，在maya的python脚本编辑器中运行`/Code/maya_render.py`文件进行渲染（注意修改其中的文件路径）。

