# Megatron-4090机器搭配

- 李沐动手用机器学习 reading
  - 装机基础
  - 环境安装
  - 性能测试
  - 存储测试
- 装机清单

[toc]



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



## 视屏1:

### 为什么我们需要训练10亿-100亿参数的模型

- **数据不再是瓶颈**：语言模型、对比学习的突破使得数据不再是一个瓶颈，可以很方便去网上收集到100GB甚至上TB的数据，可以不使用标号或者使用一个弱的标号来训练这个模型
- **算法不再是瓶颈**
- **主要的瓶颈是计算的资源 **





### 如何获取计算资源

- 高端的服务器显卡
  - DGX2 8 A100 320GB : 15W $ , 100万人民币
  - DGX2 8 A100 640GB : 20W $, (15,000 一张卡)
    - 8 * 60GB = 640GB
    - GPU-GPU之间通过**12 NVLinks**进行双向带宽，实现了600GB/s的双向速度
    - 所有GPU通讯，6个NVSwitch, 进行了4.8TB的带宽
  - 并不是好的选择
    - 太贵了
    - 一定需要一个机房来放这些机器，电量需求高同时噪音大
  - 优点是自带NVlink和NVswitch，卡与卡之间，机器与机器之间通信非常好



- 高端的游戏显卡和游戏主板
  - 需求：安静、能主动散热、GPU显存要大、~~带宽要足够好~~
    - 温度过高会使得GPU降频，效率低下
  - GPU : 
    - 风扇：一台机器放多卡需要**涡轮**散热，想要安静要**水冷**散热；风的走向会从涡轮进去从侧面吹出去 ----对于散热很关键
      - 水冷的卡虽然安静，但是水冷特别占机箱，**如果想要一个机箱放4块卡，最好选择只有一个涡轮风扇的卡，例如nvidia原版生产的卡，这样进风比较规律而且卡之间可以挨的比较紧**
      - 为什么不能选择4 * 三个风扇的卡？因为对于这种类型的显卡而言，风是从正面进去四面八方散热，如果卡挨的比较紧，机箱的温度就比较高
    - 带宽：一般多个显卡插在PCIE的主板之间，主板进行PCIE的通讯 e.g. PCIE 4.0，每一条通道是2GB/s，一般16条通道就是32GB/s ,20倍慢于A100的600GB/s
  - **主板：看PCIE和价格，希望能够支持多张卡** 
  - 固态硬盘 ：2TB虽然不是很大，但是用起来速度非常快；如果纯考虑存储数据量的话可以买机械硬盘
  - 内存：主板上4个插口用4个内存，32GB*4=128GB
  - **电源：比较重要，一块GPU耗电在450瓦，整台机器耗电在1000瓦以上**
  - 散热：一块风扇水冷120mm，似乎可以换成两个风扇
  - 机箱：空间越大散热越好
  - 其他：万兆网卡



风扇因为是对外面吹风，所以横着竖着无所谓

冷风从底下吹进来，从上面和 前后排出



- 自己租服务器例如AWS
  - 3年租金等于买一台回家里





1. 显卡的牌子

- Asus 华硕 (2.5个slot)
- EVGA
- Gigabyte Technology 技嘉科技 (标准的两个slot)

2. 显卡和CPU之间，显卡和显卡之间标准怎么交互------ PCIE 

PCIe（Peripheral Component Interconnect Express）是一种高速串行计算机扩展总线标准。它用于连接计算机的主要部件，如CPU、内存、硬盘、显卡等。PCIe是较早的PCI（Peripheral Component Interconnect）和AGP（Accelerated Graphics Port）标准的替代者，它提供了更高的数据传输速度和更好的性能。







拿GPU不要碰金属的地方，如果有静电会造成GPU导电，



## 视屏2 : 环境安装，Transformer单机Tfloats benchmark

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
docker exec -it container_name bash #execture a runing docker with interactive terminal
docker kill container_name #kill a docker with some signature

docker ps #查看所有Running的container
docker ps -a #查看所有的container
```





ssh 基本命令

1. 反向代理 frp ，肯定需要一个可以暴露的公网IP
2. ssh 进行端口转发 
   - 本地远程连接服务器并设置端口转发 `ssh -L 8888:localhost:7777 bozhang@xxxx` --- 表示将服务器的7777端口转发到本地(localhost)的8888端口
   - 在服务器开一个jupyter notebook `jupyter notebook -port 7777 --no-browser `---- 表示服务器打开7777端口的jupter notebook服务
   - 在本地网页打开 `localhost:8888` , 对应服务器端的7777链接



### Transformer Benchmarks

TFLOPS = Tera FLOPS 表示 ; trillion 万亿

**FLOPS** : Floating point Operations per Second 表示计算机每秒钟可以进行多少次的浮点数运算，是一个真实测量的值

**FLOPs**：表示理论上某个操作有多少个浮点数运算，只有模型架构有关，与机器、代码平台都没有关系

TFLOPS = FLOPs / (1e12 * 运行时间)



**矩阵运算的浮点数**

1. 对于两个n维向量，向量内积的浮点数运算是n次乘法以及n-1次加法，一共有2n-1次运算，大致计算为2n次运算
2. 两个矩阵相乘，(a,b) @ (b,c) 最终矩阵为(a,c) 每个矩阵元素的结果都是一次向量乘法的结果，则对于矩阵运算一共有 a * c * 2b = 2abc次 

> 矩阵乘法表示的是某个机器能够进行浮点数运算的**上限**；transformer操作本质就是矩阵乘法
>
> 卷积操作的性能和矩阵乘法很有关，但现在的卷积实现已经优化程度很高，不用矩阵乘法实现了



**Transformer Block的浮点数**

注意区分浮点运算与nn.Module参数的关系

注意这里面存在多维张量，按照李沐老师的视频，[b,s,h]的矩阵每次抽出一个[b,h]的矩阵进行矩阵乘法

1. ffn TFLOPs = 16 * b *s * h * h / 1e12

   - ```python
     ffn = nn.Sequential(
     	nn.Linear(h,4h), # 浮点运算 (b,h) @ (h,4h) -> s *b * 4h * 2h = 8 * b *s * h *h
       nn.ReLU(),
       nn.Linear(4h,h) # 浮点运算 (b,4h) @ (4h,h) -> s *b * h *2 * 4h = 8 * b *s * h *h
     )
     ```

2. atten = (4 * b * h *s *s + 8 *b * s * h * h)  / 1e12

   >  计算时先忽略批量batch和头的数目a，则QKV的中间维度为[h/a]，形状为[s, h/a]

   - 计算QKV : [s,h] @ [h, h/a] 重复三次，复杂度为3 * s *2h * h / a = 6 * s * h * h / a
   - 每一个query矩阵[s, h/a]里面长度为h/a的向量要与key矩阵的每一个向量[h/a,]做内积，得到一个向量[s]之后再与value矩阵[s,h/a]进行矩阵乘法，得到一个更新后的向量[h/a]，复杂度如下
     - s * 2h/a (得到attention score)
     - [1,s] @ [s,h/a] = 2sh/a  (得到QKV之后的向量)
   - 一个query向量更新的复杂度是4sh/a，则s个query向量更新复杂度是4s * s *h /a
   - 一个头的复杂度是6 * s * h * h /a + 4 * s * s * h /a，则a个头复杂度为6 * s * h * h + 4 *s * s * h
   - 最后把每个头的输出[s,h/a]拼接起来得到 [s, h]再做一次投影，计算为 [s,h] @ [h,h] = 2 * s * h * h
   - 则一个batch的计算为8 * s * h * h + 4 *s *s *h
   - batch=b的计算为 8 * b * s * h * h + 4 *b *s *s *h

3. forward = ffn + ( 2 if cross_attention else 1) * atten 

我们忽略掉了俺元素的操作比如ReLU，mask，LayerNorm这些操作

- 虽然这些按元素操作不占用浮点数运算，但是可能在整体时间上消耗是不低的

近似认为梯度的运算是backward运算✖️2



**一些计算的trick**

```python
from collections import defaultdict
results = defaultdict(lambda : {}) 
```

defaultdict是python中dict的一个子类，如果访问一个不存在的key时，会为这个key生成一个默认值，防止程序break up your code execution



由许多成员变量组成的数据类

```python
@dataclass
class Exp:
  name : str
  model :str
  batch_size: int
  seq_len: int = None
  
  def __post_init__(self):
    # 在生成class对象之后要进行的操作
```











## 视频3:多卡训练Bert, GPT 
