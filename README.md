# Da-sac for Ascend

## 添加功能

- [x] 1. 修改网络结构与推理配置，适配无 CUDA 环境
- [x] 2. 实现相机实时输入推理
- [ ] 3. 实现 onnx 模型推理

## 环境配置

参考原项目：<https://github.com/visinf/da-sac#installation>

数据集下载部分，由于只验证了 Cityscapes 数据集，因此可只下载 `./data/cityscapes/gtFine2/` 和 `./data/cityscapes/leftImg8bit/` 两个文件。

另外，为实现视频流输入输出，需要安装 opencv：

```bash
conda install opencv-python
```

## 模型下载

参考原项目：<https://github.com/visinf/da-sac#training>

可以使用自己训练的模型或提供的模型，后者可以用脚本下载，注释掉不需要的模型即可：

```bash
cp tools/download_baselines.sh snapshots/cityscapes/baselines/
cd snapshots/cityscapes/baselines/
bash ./download_baselines.sh
```

## 运行推理

通过脚本运行：

```bash
launch/infer_camera.sh
```

模型、数据集、输出路径等参数可在脚本中修改。

### 相机的连接

视频的输入是通过 opencv 的 `cv2.VideoCapture()` 方法实现的，因此需要先连接相机，然后在 `infer_camera.py` 修改相机对应的设备号：

```python
cap = CamCap(DEVICE_ID)
```

### GPU 与 CPU 推理的切换

由于原项目是基于 CUDA 的，因此做了对应改动使其能在无 CUDA 环境下运行。要切换回 GPU 推理，需要进行修改：

- 在 `infer_camera.py` 中将 `model = nn.DataParallel(model).cpu()` 修改为 `model = nn.DataParallel(model).cuda()`

- 将推理用的 backbone 中的部分 `BatchNorm2d` 算子修改为 `SyncBatchNorm`，以 `models/basenet.py` 为例，需要将：

    ```python
    _trainable = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.BatchNorm2d)
    _batchnorm = (nn.BatchNorm2d, nn.BatchNorm2d, nn.GroupNorm)
    ```

    修改为：

    ```python
    _trainable = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.SyncBatchNorm)
    _batchnorm = (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)
    ```

- *(可选的)* 在 `opts.py` 中的 `get_arguments` 方法中加入参数检查：`check_global_arguments(args)`
