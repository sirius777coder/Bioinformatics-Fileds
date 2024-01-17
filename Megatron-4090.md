# Megatron-4090机器搭配

- 李沐动手用机器学习 reading
  - 装机基础
  - 环境安装
  - 性能测试
  - 存储测试
- 装机清单

## 采购计划

| 清单                            | 型号                          | 价格  | 购买 |
| ------------------------------- | ----------------------------- | ----- | ---- |
| cpu                             | AMD r9 7950x - 16核，32线程   | 4699  |      |
|                                 |                               |       |      |
|                                 |                               |       |      |
|                                 |                               |       |      |
| gpu                             | 技嘉水冷                      | 18000 | ✅    |
|                                 |                               |       |      |
|                                 |                               |       |      |
|                                 |                               |       |      |
| 主板(motherboard or host)       | 技嘉 超级雕 B650E             | 2499  |      |
| 内存（RAM）                     | ? 32G * 4                     | 2800  |      |
| SSD(Solid State Drive) 固态硬盘 | 技嘉 M.2 大雕2TB              | 2789  |      |
| 电源                            | 微星 额定1000W ATX3.0         | 1200  |      |
| 散热                            | 瓦尔基里 E360                 | 1000  |      |
| 机箱                            | 爱国者YOGO K100 黑色 防尘降噪 | 1000  |      |
|                                 |                               |       |      |
|                                 |                               |       |      |
|                                 |                               |       |      |
|                                 |                               |       |      |
|                                 |                               |       |      |

> 1. 显卡的牌子
>
> - Asus 华硕 (2.5个slot)
> - EVGA
> - Gigabyte Technology 技嘉科技 (标准的两个slot)
>
> 2. 显卡和CPU之间，显卡和显卡之间标准怎么交互------ PCIE 
>
> PCIe（Peripheral Component Interconnect Express）是一种高速串行计算机扩展总线标准。它用于连接计算机的主要部件，如CPU、内存、硬盘、显卡等。PCIe是较早的PCI（Peripheral Component Interconnect）和AGP（Accelerated Graphics Port）标准的替代者，它提供了更高的数据传输速度和更好的性能。

如何进行深度学习训练

- 高端的服务器显卡
  - DGX2 8 A100 320GB : 15W $ , 100万人民币
  - DGX2 8 A100 640GB : 20W $, (15,000 一张卡)
  - 并不是好的选择，太贵了，溢价非常严重，主要是卡之间的通讯非常好

- 高端的游戏显卡和游戏主板
  - 需求：安静、能主动散热、GPU显存要大且带宽要足够好
  - GPU : 
    - 风扇：一台机器放多卡需要**涡轮**散热，想要安静要**水冷**散热，单张卡用显卡加三个风扇的问题不大
    - 带宽：一般的多卡卡插在PCIE的主板之间，主板进行PCIE的通讯 e.g. PCIE 4.0
  - 主板：看PCIE和价格
  - 固态硬盘 ：2TB虽然不是很大，但是用起来速度非常快；如果纯考虑存储数据量的话可以买机械硬盘
  - 内存：主板上4个插口用4个内存，32GB*4=128GB
  - 电源：比较重要，一块GPU耗电在450瓦，整台机器耗电在1000瓦以上
  - 散热：一块风扇水冷120mm，似乎可以换成两个风扇
  - 机箱：空间越大散热越好
  - 其他：万兆网卡

- 自己租服务器
  - 3年租金等于买一台回家里



拿GPU不要碰金属的地方，如果有静电会造成GPU导电，



## 环境安装

裸机装cuda, driver, PyTorch GPU版本

- 安装cuda大礼包（包含driver和cuda集成环境），再通过pip安装对应cuda的GPU版本
- 安装driver，再通过conda安装不同pytorch对应的cudatoolkit

> 只安装driver 可以通过nvidia-smi访问显卡，此时nvidia-smi显示的是driver支持的最高cuda版本，本机并没有安装cuda，可以通过nvcc -V查看本机的显卡

- 安装driver，再安装容器

Comments:

1. 如果只需要pytorch，那么本机根本不用安装cuda，只用安装driver就行了，但是如果要跑simulation等其他加速软件，似乎本机也要装一个cuda



04.30

https://discuss.pytorch.org/t/is-it-required-to-set-up-cuda-on-pc-before-installing-cuda-enabled-pytorch/60181/16

一些更正的comments

1. 只要安装了NVIDIA driver，不管有没有cuda，**conda/pip**都会将CUDA runtime和cublas等二进制包安装上

> - You would still only need to install the NVIDIA driver to run GPU workloads using the PyTorch binaries with the appropriately specified `cudatoolkit` version.
> - One limitation to this is that you would still need a locally installed CUDA toolkit to build custom CUDA extensions or PyTorch from source. (想从源代码安装PyTorch 可能需要现在电脑上安装CUDA toolkit)
> - No CUDA *toolkit* will be installed using the current binaries, but the CUDA *runtime*, which explains why you could execute GPU workloads, but not build anything from source.

PS : 现在版本（Pytorch 1.13 CUDA>=11.6）conda/pip 安装的都是CUDA runtime和二进制包而不是cudatoolkit，但是早期的PyTorch版本似乎都需要直接conda/pip安装cudatoolkit.



2. docker 安装

> Docker是一种容器化技术，表示在不同的操作系统中快速便捷的部署和运行程序
>
> - (class instance)容器是一个独立运行的、可移植的软件包，其中包含了应用程序的代码、运行时、系统工具、系统库和配置文件等。容器可以在任何支持Docker的操作系统上运行，并且具有独立的文件系统和网络等资源，与其他容器和主机系统互相隔离。
>
> - (class)镜像是用于创建容器的模板，包含了应用程序运行所需的所有文件、配置和依赖项等。镜像可以被看作是一个静态的、只读的文件系统快照，它不包含运行时状态和用户数据等。通过镜像，可以创建一个或多个容器，每个容器都是基于同一个镜像创建的，并共享该镜像的代码和配置。\
> - Mac DockerDesktop 的路径在`/Applications/Docker.app/Contents/Resources/bin` , bin目录表示可执行文件

容器运行的是操作系统ubantu，而不像conda一样仅仅是在操作系统内管理不同的环境，所以docker运行对外部环境并不敏感

```
Common Commands:
  run         Create and run a new container from an image
  exec        Execute a command in a running container
  ps          List containers
  build       Build an image from a Dockerfile
  pull        Download an image from a registry
  push        Upload an image to a registry
  images      List images
  login       Log in to a registry
  logout      Log out from a registry
  search      Search Docker Hub for images
  version     Show the Docker version information
  info        Display system-wide information
```

- `docker build -t welcome-to-docker . ` 在某一个文件夹下找到dockerfile文件 并建立一个名为dockerfile的镜像

- `docker run -i -t --name sirius_ubantu ubuntu bash` 

其他基本命令

本机的container_name 有 sirius_ubantu

```bash
docker restart container_name #open a docker
docker exex -it container_name bash #execture a runing docker with interactive terminal
docker kill container_name #kill a docker with some signature

docker ps #查看所有Running的container
docker ps -a #查看所有的container
```

