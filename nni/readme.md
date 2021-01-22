# NNI

**NNI (Neural Network Intelligence)** 
是一个轻量但强大的工具包，帮助用户**自动**的进行 [特征工程](https://nni.readthedocs.io/zh/latest/FeatureEngineering/Overview.html)，[神经网络架构搜索](https://nni.readthedocs.io/zh/latest/NAS/Overview.html)， [超参调优](https://nni.readthedocs.io/zh/latest/Tuner/BuiltinTuner.html)以及[模型压缩](https://nni.readthedocs.io/zh/latest/Compression/Overview.html)。   

## NNI工作流程

> 输入：搜索空间，试用代码，配置文件
> 输出：一种最佳的超参数配置  
>
> 1：对于t = 0、1、2，...，maxTrialNum，
>
> 2：超参数=从搜索空间中选择一组参数
>
> 3：最终结果= run_trial_and_evaluate（超参数）
>
> 4：向NNI报告最终结果
>
> 5：如果达到上限时间，停止实验
>
> 返回具有最佳最终结果的超参数值

## mnist示例

直接运行mnist_nni.py一次只能尝试一组参数。如果要调整学习率，则需要手动修改超参数并一次又一次地开始试验。

1. 使用JSON编写文件，包括需要搜索的所有超参数（离散值或连续值）

   > [search_space.json](https://github.com/microsoft/nni/blob/85c0d841a6a15d64f32d8237e29616227fd03425/examples/trials/mnist-pytorch/search_space.json)

2. 修改代码以从NNI获取超参数集，并将最终结果报告给NNI。

   > mnist_nni.py 已修改

3. `定义一个config.yml文件，该文件包括路径搜索空间和文件声明，还提供其他信息，例如调整算法、最大持续时间参数等。

   > config.yml

从**命令行**运行**config.yml**文件以开始MNIST实验。

```bash
nnictl create --config ./nni/config.yml
# The Web UI urls are: http://127.0.0.1:8080
```

更多命令[nnictl](https://nni.readthedocs.io/en/latest/Tutorial/Nnictl.html)

```bash
# 恢复已停止的实验
nnictl resume [experiment_id]
# 查看已停止的实验
nnictl view [OPTIONS]
# 停止正在运行的实验或多个实验
nnictl stop [Options]
## 显示所有（运行中）实验的信息。##
nnictl experiment list --all
# 删除一个或所有实验，其中包括日志，结果，环境信息和缓存。
nnictl experiment delete [OPTIONS]
###将实验的结果和超参数导出到csv或json文件中。###
nnictl experiment export [experiment_id] --filename [file_path] --type json --intermediate
# 导入
nnictl experiment import [experiment_id] -f experiment_data.json
# 保存nni实验元数据和代码数据。
nnictl experiment save [experiment_id] --saveCodeDir
# 载入nni实验。
nnictl experiment load --path [path] --codeDir [codeDir]
####显示实验的WebUI网址###
nnictl webui url
```

