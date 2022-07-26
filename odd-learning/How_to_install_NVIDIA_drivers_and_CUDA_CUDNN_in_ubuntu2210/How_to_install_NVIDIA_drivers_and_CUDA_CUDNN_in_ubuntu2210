## 如何在ubuntu 2210中安装NVIDIA驱动、CUDA和cuDNN

本篇讲解参考NVIDIA官方[CUDA ToolKIT DOCUMENTATION](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)和[CUDNN DUCUMENTATION](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

本机参数如下：

```markdown
Distributor ID:Ubuntu
Description:Ubuntu Kinetic Kudu (development branch)
Release:22.10
Codename:kinetic
Linux内核版本:5.15.0-27-generic
显卡型号：NVIDIA GeForce RTX3060 Mobile / Max-Q
```



### 安装NVIDIA显卡驱动

采用ubuntu自带的驱动安装，也可指定安装的驱动。

```bash
sudo ubuntu-drivers autoinstall
```

安装完后重启

```bash
sudo reboot
```

重启后输入以下命令查询显卡驱动是否安装成功，我们在这张图中可以看到当前驱动版本位515.48.07，CUDA版本最高可支持到11.7。**由于我们的系统是最新的，所以我们默认全都装最高版本**。

```bash
nvidia-smi
```

![](/home/chairc/typora/How_to_install_NVIDIA_drivers_and_CUDA_CUDNN_in_ubuntu2210/img/1.png)

### 安装CUDA 11.7

本机参数如下：

使用以下命令验证当前GPU的能力：

```bash
lspci | grep -i nvidia
```

使用以下命令检查当前Linux的版本支持：

```bash
uname -m && cat /etc/*release
```

查询内核支持：

```bash
sudo apt-get install linux-headers-$(uname -r)
```

**以下安装均以初次安装，不提供卸载讲解**

NVIDIA CUDA工具下载网站：[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

![](/home/chairc/typora/How_to_install_NVIDIA_drivers_and_CUDA_CUDNN_in_ubuntu2210/img/2.png)

使用以下命令删除过时的签名的密钥：

```bash
sudo apt-key del 7fa2af80
```

由于我们使用的是下载到本地安装方式，所以在 **下载位置** 输入以下指令：

```bash
sudo dpkg -i cuda-repo-你下载的那个文件（一般用Tab补齐就行）.deb
```

安装完后会提示你需要注册临时公共秘钥，**控制台中会给你一个命令，复制控制台上出现的!!**

```bash
sudo cp /var/cuda-repo-(复制控制台上的命令)-local/cuda-*-keyring.gpg /usr/share/keyrings/
```

更新APT仓库缓存

```bash
sudo apt-get update
```

安装CUDA SDK

```bash
sudo apt-get install cuda
```

安装nvidia-gds

```bash
sudo apt-get install nvidia-gds
```

重启电脑

```bash
sudo reboot
```

开始设置环境变量

```bash
sudo gedit ~/.bashrc
```

在文件最后写入

```bash
export CUDA_DIR=/usr/local/cuda-11.7/bin${CUDA_DIR:+:${CUDA_DIR}}
export LD_CUDA_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_CUDA_LIBRARY_PATH:+:${LD_CUDA_LIBRARY_PATH}}
```

我们输入以下命令查询CUDA安装情况

```bash
nvcc -V
```

![](/home/chairc/typora/How_to_install_NVIDIA_drivers_and_CUDA_CUDNN_in_ubuntu2210/img/3.png)

### 安装cuDNN 8.4

由于官方需要注册账号下载，下载地址：https://developer.nvidia.com/rdp/cudnn-download

下载完成后进入你的cuDNN路径中，解压CUDNN文件

```bash
tar -xvf cudnn-linux-x86_64-8.x.x.x（cuDNN版本）_cuda版本-archive.tar.xz
```

将以下文件复制到CUDA目录。

```bash
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

