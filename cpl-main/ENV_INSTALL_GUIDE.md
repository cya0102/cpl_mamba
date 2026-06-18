# CPL + AMP 虚拟环境安装指南

这份环境用于运行 `cpl-main` 里的 CPL/AMP 实验，也就是：

```bash
python train.py --config-path config/charades/amp.json --log_dir logs --tag amp_charades
python train.py --config-path config/activitynet/amp.json --log_dir logs --tag amp_activitynet
```

不要直接在这个环境里安装 `hieramamba-main/requirements.txt`。CPL 需要 `fairseq`，而 `fairseq==0.12.2` 常用的 `omegaconf/hydra-core` 版本与 HieraMamba 全量 requirements 里的 `omegaconf>=2.2.3` 容易冲突。这里安装的是 CPL+AMP 最小可跑环境。

## 推荐版本

| 组件 | 推荐版本 | 说明 |
| --- | --- | --- |
| OS | Linux x86_64 | Mamba CUDA kernel 主要面向 Linux + NVIDIA GPU |
| NVIDIA Driver | 支持 CUDA 12.0 即可 | 你的服务器当前驱动可用；不用强行升级到 CUDA 12.1 |
| CUDA Runtime | 11.8 | 由 PyTorch `cu118` wheel/conda 包提供；源码编译 Mamba 时最好服务器也有 CUDA Toolkit 11.8 |
| Python | 3.10.13 | 避开 Python 3.11/3.12 下 fairseq 旧依赖的坑 |
| PyTorch | 2.2.2+cu118 | 与 HieraMamba 的 `torch>=2.1.0` 要求兼容，也适配你当前驱动 |
| torchvision | 0.17.2+cu118 | 与 PyTorch 2.2.2 配套 |
| torchaudio | 2.2.2+cu118 | 与 PyTorch 2.2.2 配套 |
| mamba-ssm | 2.2.3 | 项目要求 `>=2.2.3`，这里锁定版本便于复现实验 |
| causal-conv1d | 1.4.0 | Mamba 官方推荐先装 PyTorch，再以 `--no-build-isolation` 安装 CUDA 扩展 |
| fairseq | 0.12.2 | CPL 的 attention softmax 依赖 |
| pip | 24.0 | 旧版 fairseq / omegaconf 在过新的 pip 上可能安装失败 |
| numpy | 1.26.4 | 稳定兼容 Python 3.10 / PyTorch 2.2 |
| h5py | 3.10.0 | 读取 Charades / ActivityNet HDF5 特征 |
| nltk | 3.8.1 | 数据集文本分词和 POS tagging |

## 你的服务器推荐方案：CUDA 11.8 / cu118

如果 `nvidia-smi` 里显示的最高 CUDA 版本是 `12.0`，不建议硬装 `cu121`。NVIDIA driver 对低版本 CUDA runtime 向后兼容，所以更稳的方案是用 PyTorch 的 `cu118` wheel。

你的两张卡是：

| GPU | 架构 | Compute Capability |
| --- | --- | --- |
| TITAN RTX | Turing | 7.5 |
| RTX 3090 | Ampere | 8.6 |

如果 `mamba-ssm` / `causal-conv1d` 需要本地编译，安装前设置：

```bash
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
```

这样编译出的 CUDA 扩展会同时支持 TITAN RTX 和 RTX 3090。建议服务器环境用下面这套：

| 组件 | 推荐版本 |
| --- | --- |
| NVIDIA Driver | 支持 CUDA 12.0 的当前驱动即可 |
| CUDA Toolkit / nvcc | 11.8 |
| Python | 3.10.13 |
| PyTorch | 2.2.2+cu118 |
| torchvision | 0.17.2+cu118 |
| torchaudio | 2.2.2+cu118 |
| mamba-ssm | 2.2.3 |
| causal-conv1d | 1.4.0 |

对应 PyTorch 安装命令：

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

如果集群用 module 管理 CUDA，优先加载 CUDA 11.8：

```bash
module load cuda/11.8
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
```

如果没有 CUDA 11.8 toolkit / `nvcc`，PyTorch 本身仍可运行，但 Mamba CUDA 扩展在没有预编译 wheel 时可能无法安装。此时有三个选择：

1. 让管理员安装或加载 CUDA Toolkit 11.8。
2. 用 NVIDIA Docker / Apptainer 跑 `cuda:11.8-devel` 容器，宿主机驱动支持 CUDA 12.0 即可。
3. 临时把 `config/*/amp.json` 里的 `"use_mamba": false`，先用卷积 fallback 排查训练流程；正式实验仍建议装好 Mamba。

如果安装 `causal-conv1d` / `mamba-ssm` 前忘了设置 `TORCH_CUDA_ARCH_LIST`，影响取决于安装方式：

- 如果 pip 下载的是预编译 wheel，通常没有影响，wheel 已经带有编译好的 CUDA kernel。
- 如果 pip 是本地源码编译，PyTorch 会尽量根据当前可见 GPU 编译。若安装时只暴露了 RTX 3090，扩展可能只支持 `sm_86`；之后放到 TITAN RTX 上跑可能报 `no kernel image is available for execution on the device`。反过来也类似。
- 如果安装时两张卡都对进程可见，通常会同时编译两张卡架构，但显式设置 `TORCH_CUDA_ARCH_LIST="7.5;8.6"` 更稳。

补救方式是强制重装 CUDA 扩展：

```bash
conda activate cpl_amp
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
pip uninstall -y causal-conv1d mamba-ssm
pip install causal-conv1d==1.4.0 --no-build-isolation --no-cache-dir
pip install mamba-ssm==2.2.3 --no-build-isolation --no-cache-dir
```

重装后分别在两张卡上测试：

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; from mamba_ssm import Mamba2; m=Mamba2(d_model=32,d_state=64,d_conv=4,expand=2,headdim=16).cuda(); x=torch.randn(2,16,32,device='cuda'); print(torch.cuda.get_device_name(0), m(x).shape)"
CUDA_VISIBLE_DEVICES=1 python -c "import torch; from mamba_ssm import Mamba2; m=Mamba2(d_model=32,d_state=64,d_conv=4,expand=2,headdim=16).cuda(); x=torch.randn(2,16,32,device='cuda'); print(torch.cuda.get_device_name(0), m(x).shape)"
```

## Hydra / 预训练权重说明

当前 `cpl-main/models/modules/amp_backbone.py` 里的 AMP 适配版没有直接 import HieraMamba 原仓库的 `Hydra` 类；它默认使用 `mamba_ssm.Mamba2` 做双向扫描，或者在没有 Mamba 扩展时使用卷积 fallback。因此运行 CPL+AMP 不需要额外安装 Hydra 包，也不需要下载 Hydra 专用预训练权重。

`config/*/amp.json` 默认是：

```json
"pretrained_path": null,
"freeze_backbone": false
```

也就是说会从头训练 AMP backbone。后续如果你有 HieraMamba checkpoint，可以把路径填到 `pretrained_path`。加载器会按 key 和 shape 尽量部分加载，输入投影层因为 CPL 特征维度不同，通常不会完全匹配；branch / Mamba 层若维度一致才会加载。

## 方式一：conda/mamba 创建环境

```bash
conda create -n cpl_amp python=3.10.13 -y
conda activate cpl_amp

python -m pip install --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0 packaging==24.0 ninja==1.11.1.1

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

pip install numpy==1.26.4 h5py==3.10.0 nltk==3.8.1 tqdm==4.66.4 scipy==1.11.4

pip install Cython==0.29.36 hydra-core==1.0.7 omegaconf==2.0.6
pip install fairseq==0.12.2

# CPL+AMP 不依赖 transformers；如果环境里已有新版 transformers，它可能要求 torch>=2.4。
# 保守做法是卸载，或固定到较旧版本。
pip uninstall -y transformers tokenizers
# 可选：只有你额外运行 HieraMamba 原仓库脚本时才需要 transformers。
# pip install transformers==4.38.2 tokenizers==0.15.2

export TORCH_CUDA_ARCH_LIST="7.5;8.6"
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm==2.2.3 --no-build-isolation

python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

如果 `causal-conv1d` 或 `mamba-ssm` 编译失败，先确认：

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

常见原因是服务器只有 CUDA driver，没有 `nvcc`；如果没有预编译 wheel，Mamba 扩展会需要 CUDA Toolkit。集群上通常可以先执行类似：

```bash
module load cuda/11.8
```

再重新安装 `causal-conv1d` 和 `mamba-ssm`。

## 方式二：environment.yml

也可以保存下面内容为 `environment.yml` 后创建环境：

```yaml
name: cpl_amp
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10.13
  - pip
  - pytorch==2.2.2
  - torchvision==0.17.2
  - torchaudio==2.2.2
  - pytorch-cuda=11.8
  - pip:
      - pip==24.0
      - setuptools==69.5.1
      - wheel==0.43.0
      - packaging==24.0
      - ninja==1.11.1.1
      - numpy==1.26.4
      - h5py==3.10.0
      - nltk==3.8.1
      - tqdm==4.66.4
      - scipy==1.11.4
      - Cython==0.29.36
      - hydra-core==1.0.7
      - omegaconf==2.0.6
      - fairseq==0.12.2
      # CPL+AMP 本身不需要 transformers；不要安装最新版 transformers，
      # 否则它可能要求 torch>=2.4 并打印 backend disabled 警告。
```

创建后再单独安装 CUDA 扩展：

```bash
conda env create -f environment.yml
conda activate cpl_amp
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm==2.2.3 --no-build-isolation
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

## 安装后验证

在服务器上运行：

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import fairseq, h5py, nltk; from mamba_ssm import Mamba2; print('deps ok')"
python -c "import torch; from mamba_ssm import Mamba2; m=Mamba2(d_model=32,d_state=64,d_conv=4,expand=2,headdim=16).cuda(); x=torch.randn(2,16,32,device='cuda'); print(m(x).shape)"
```

再验证 CPL 代码能导入：

```bash
cd /path/to/cpl_mamba/cpl-main
python -m py_compile models/cpl.py models/modules/amp_backbone.py
```

## 开始实验

```bash
cd /path/to/cpl_mamba/cpl-main

python train.py --config-path config/charades/amp.json --log_dir logs --tag amp_charades
python train.py --config-path config/activitynet/amp.json --log_dir logs --tag amp_activitynet
```

如果显存不足，优先改对应 `amp.json`：

```json
"batch_size": 16
```

如果 Mamba CUDA 扩展临时装不上，但想先检查 CPL+AMP 训练管线，可以把 `config/*/amp.json` 里的：

```json
"use_mamba": false
```

这样会走卷积 fallback，不代表最终 AMP 实验效果，只用于排查数据、loss、checkpoint 等流程。

## transformers 警告处理

如果训练启动时看到：

```text
[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.2+cu118
[transformers] PyTorch was not found. Models won't be available ...
```

这是因为环境里安装了过新的 `transformers`，它要求更高版本 PyTorch。`cpl-main` 的训练代码不使用 `transformers`，所以如果训练能继续，这个提示可以忽略；如果它影响启动，直接卸载：

```bash
pip uninstall -y transformers tokenizers
```

如果你还要在同一个环境里运行 HieraMamba 原仓库中依赖 transformers 的脚本，则不要装最新版，固定旧版：

```bash
pip install transformers==4.38.2 tokenizers==0.15.2
```

然后再验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from mamba_ssm import Mamba2; print('mamba ok')"
```
