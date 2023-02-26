[toc]





## Pytorch 工程化方法

1. Tensorboard 可视化

> 不同的summary 对象，只要name一样，就可以放到同一个图里;
>
> 默认SummaryWriter的路径是

```python
from torch.utils.tensorboard import SummaryWriter
# Writer will output to ./runs/ directory by default
# 定义一个summary 对象 runs/CURRENT_DATETIME_HOSTNAME
writer = SummaryWriter(log_dir="./runs/exp1")
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter) # name{group_name/item_name}, y,x
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
# 定义第二个summary 对象
writer_2 = SummaryWriter(log_dir="./runs/exp1")
for n_iter in range(100):
    writer_2.add_scalar('Loss/train', np.random.random(), n_iter)
    writer_2.add_scalar('Loss/test', np.random.random(), n_iter)
    writer_2.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer_2.add_scalar('Accuracy/test', np.random.random(), n_iter)
```

开启tensorboard

```shell
tensorboard --logdir=runs
```





## Python使用技巧

1. 使用sorted函数对一个抽象的字典排序

`sorted(iterable,key)` 

e.g. `x = sorted(y,key=lambda i:i[2]),reverse=False`

y表示可迭代的对象如列表，字典等；key表示一个函数，以什么样的标准对可迭代对象进行排序

2. 可视化tensor splicing

- (a) 表示保留所有的dim 0 

![image-20221124182227678](/Users/sirius/Library/Application Support/typora-user-images/image-20221124182227678.png)



##  Pytorch 使用技巧

0.内存分配

- 静态内存：模型的参数
- 动态内存 --- 计算图的开销
  - 激活值：每次前向传播都需要计算并保留中间激活值
  - 梯度：backward()需要
  - 优化器的状态：定义优化器的时候需要

解决显存问题的方法

- 并行计算

- check pointing

  > **torch.utils.checkpoint.checkpoint(functions, *args,**preserve_rng_state**)**
  >
  > Checkpoint a model or part of the model
  >
  > Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, the checkpointed part does **not** save intermediate activations, and instead recomputes them in backward pass. It can be applied on any part of a model.
  >
  > Specifically, in the forward pass, `function` will run in [`torch.no_grad()`](https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad) manner, i.e., not storing the intermediate activations. Instead, the forward pass saves the inputs tuple and the `function` parameter. In the backwards pass, the saved inputs and `function` is retrieved, and the forward pass is computed on `function` again, now tracking the intermediate activations, and then the gradients are calculated using these activation values.
  >
  > The output of `function` can contain non-Tensor values and gradient recording is only performed for the Tensor values. Note that if the output consists of nested structures (ex: custom objects, lists, dicts etc.) consisting of Tensors, these Tensors nested in custom structures will not be considered as part of autograd.
  >
  > E.g.
  >
  > ```python
  > from torch.utils.checkpoint import checkpoint 
  > class toy(nn.Module):
  >   def __init__(self):
  >     self.net = nn.Linear(10,10)
  >   def forward(self,x):
  >     x = checkpoint(self.net,x) #计算该模块时利用checkpoint技术
  >     return x
  > ```
  >
  > 

- 混合精度

1.items()函数----注意对应torch.tensor.item()作用是将单个tensor元素提取出来,比如 (pred==labels).sum().item()

-返回一个字典的key-value元组集合

注意字典也可以通过一行代码生成

Eg. 探究各个样本的预测正确率`x = {class_name:0 for class_name in classes}`



2.enumarate()函数

返回一个可迭代对象的index与值

```python
for idx,value in enumarate(lt):
```

eg.

```python
for batch,(X,y) in enumerate(dataloader):
  pass
```

dataloader是一个可迭代对象 ( class ) ，enumerate每次从可迭代对象里取值batch=索引，（X,y）=list[tensorFeatures,tensorLabels]



3.iter()与next()函数

- iter()转化为迭代器
- next()从迭代器里取值



4.import tqdm

- tqdm(iterator)



5.init getitem len 是创建dataset的三个必要函数



6. 

两个冒号`::`表示按照间隔从一个列表、ndarray、tensor中进行splicing，注意右区间是可以取到的

一个冒号`:`表示取一定范围内的数字，右区间不可以取

```python
a[0::2] //
pe[:,0::2]//从0-2-4-2i<=n
```



7. Numpy、Torch里广播的机制    **从最右边开始对齐，看看是否维度compatible**

> The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes. 
>
> **Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python**.  
>
> It does this without making needless copies of data and usually leads to efficient algorithm implementations. There are, however, cases where broadcasting is a bad idea because it leads to inefficient use of memory that slows computation.

Broadcasting in NumPy follows a strict set of rules to determine the interaction between the two arrays.  When operating on two arrays, NumPy compares their shapes element-wise. It starts with the **trailing (i.e. rightmost)** dimensions and works its way left. Two dimensions are compatible when

- they are equal, or

- one of them is 1

Arrays do not need to have the same number of dimensions. So there is a third rule that 

- When either of the dimensions compared is one, the other is used. 

![截屏2022-05-09 上午10.44.31](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-05-09 上午10.44.31.png)

> A set of arrays is called “broadcastable” to the same shape if the above rules produce a valid result.
>
> a.shape is (5,1)   `a=np.array([[1],[2],[3],[4],[5]])`
>
> b.shape is (1,6)   `a=np.array([[1,2,3,4,5]])`
>
> c.shape is (6.)	 `a=np.array([1,2,3,4,5])`
>
> d.shape is () 	   `a=np.array(2)`

Eg.

> 1. outer operation like outer sum or outer product
>
> ```
> >>> a = np.array([0.0, 10.0, 20.0, 30.0])
> >>> b = np.array([1.0, 2.0, 3.0])
> >>> a[:, np.newaxis] + b
> array([[  1.,   2.,   3.],
>        [ 11.,  12.,  13.],
>        [ 21.,  22.,  23.],
>        [ 31.,  32.,  33.]])
> >>> a.reshape(-1,1) * b
> array([[ 0.,  0.,  0.],
>        [10., 20., 30.],
>        [20., 40., 60.],
>        [30., 60., 90.]])
> ```
>
> 
>
> 2. AlphaFold MSA column-wise to pair-representation (outer product)
>
> input : MSA [s, r, $c_m$]
>
> > MSA[:,i,:] $\otimes$ MSA [:,j,:] = (s,1,$c_m$)  * (s,$c_m$,1)
>
> output : pair-representation [r,r,$c_z$]
>
> 
>
> 3. AlphaFold target_feature to pair-representation. (outer sum)
>
> input :  feature[r,$c_z$]
>
> > feature $\oplus$ feature = (1,r,$c_z$)  + (r,1,$c_z$)
>
> output : pair-representation (r,r,$c_z$)
>
> 
>
> 4. Layer Normalization
>
> ```python
> # step by step LayerNorm
> class LayerNorm(nn.Module):
>     "Construct a layernorm module (See citation for details)."
> 
>     def __init__(self, features, eps=1e-6):
>    """input : (N,L,B),B=features"""   
>         super(LayerNorm, self).__init__()
>         self.a_2 = nn.Parameter(torch.ones(features))
>         self.b_2 = nn.Parameter(torch.zeros(features))
>         self.eps = eps
> 
>     def forward(self, x):
>         mean = x.mean(-1, keepdim=True)
>         std = x.std(-1, keepdim=True)
>         
>         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
> ```
>
> 



8. Assignment & Deep copy & Shallow copy

前提：对于不可变对象而言，assignment和copy全部都是指向一个新的内存地址。而可变对象而言，深拷贝需要为该obj创建一个新的地址，因此深拷贝后与原来拷贝就没关系了



直接赋值: 对象的引用,也就是给对象起别名

浅拷贝: 拷贝父对象构建一个新的复合对象，但是不会拷贝对象的内部的子对象。
深拷贝: 拷贝父对象. 递归拷贝内部的所有对象



- **对于赋值语句而言，并不产生拷贝，而是产生一种name和object的绑定关系**

  被绑定的对象分为

  - 可变对象(mutable object) : 列表list、字典dictionary、类class、集合Set
  - 不可变对象(immutable object)：字符串string、数字number、元组tuple---元组中每一个元素都不能变

PS : set{1,2,3} list[1,2,2,[2,3],3]  tuple(1,2,3)



- **要产生拷贝关系需要`copy` module**------一种真正的拷贝关系
  - shallow copy : copy.copy (obj)
  - deep copy : copy.deepcopy (obj)



- Important Points:The difference between shallow and deep copying is only relevant for **compound objects** (objects that contain other objects, like lists or class instances)

  - A shallow copy constructs a new compound object and then (to the extent possible) inserts references into it to the objects found in the original.

  - A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original.

    

    ![截屏2022-01-23 下午6.22.33](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-01-23 下午6.22.33.png)

![image-20220123182254199](/Users/sirius/Library/Application Support/typora-user-images/image-20220123182254199.png)







9. Pytorch中两种容器（containers）来组建网络[https://zhuanlan.zhihu.com/p/64990232]

- nn.ModuleList () 
  - 没有自定义执行顺序,需要在forward函数中人为指定执行顺序
  - 很好的将python内置的列表和容器结合
- nn.Sequential ()
  - 按照Sequential的顺序自动执行forward函数



e.g. 最经典的两种用法

```python
class net1(nn.Module):#ModuleList
  def __init__(self,layer,n):
    super().__init__()
    self.net = nn.ModuleList([layer for _ in range(n)])
  def forward(self,x):
    for layer in self.net: #人为指定sequential的顺序
      x = layer(x)
      # x = x+layer(x) 用ModuleList很容易实残差连接
    return x
  
  
class net2(nn.Module)://Sequential
  def __init__(self,layer,n):
    super().__init__()
    self.list = [layer for _ in range(n)]
    self.net = nn.Sequential(*self.list)
  
  def forward(self,x):
    return self.net(x)
```



10. NLP中的两种mask

- **padding mask**：处理非定长序列，区分padding和非padding部分，如在RNN等模型和Attention机制中的应用等

  > 通过mask向量来记录位置 例如最大长度为5 则一个长为3的序列对应的位置向量为 [True,True,True,False,False]

Eg.对每一个蛋白质序列进行padding

补充`np.pad(input_array,padding,mode="constant")`与`torch.nn.functional.pad(input, pad, mode='constant', value=0.0)`

- Input：代表你要对谁padding
- pad_width：**每个轴**要填充的数据的数目【每个维度前、后各要填充多少个数据】

![image-20220306160638839](/Users/sirius/Library/Application Support/typora-user-images/image-20220306160638839.png)

Note:将蛋白质序列放到一个列表中，找到最大长度，可以轻松的得到一个padding好的3d_tensor (mini_batch,max_length,20)

```python
# Normal embedding
def embed_protein(amino_acid_residues: str) -> np.ndarray :
  vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R','S', 'T', 'V', 'W', 'Y']
  index = torch.tensor([vocab.index(i) for i in amino_acid_residues]).view(-1,1)
  src = torch.ones_like(index,dtype=float)
  temp = torch.zeros((len(amino_acid_residues),len(vocab)),dtype=float)
  protein_one_hot = temp.scatter_(dim=1,index,src)
  return torch.to_numpy(protein_one_hot)

def pad_one_hot_sequence(sequence: np.ndarray,target_length: int) -> np.ndarray:
  """Pads one hot sequence [seq_len, num_aas] in the seq_len dimension."""
  sequence_length = sequence.shape[0]
  pad_length = target_length - sequence_length
  if pad_length < 0:
    raise ValueError(
        'Cannot set a negative amount of padding. Sequence length was {}, target_length was {}.'
        .format(sequence_length, target_length))
  pad_values = [[0, pad_length], [0, 0]]
  return np.pad(sequence, pad_values, mode='constant')


```

PS : 如果是复杂的embedding比如含有X标签等见：[Deep Learning to annotate proteins](https://colab.sandbox.google.com/github/google-research/google-research/blob/master/using_dl_to_annotate_protein_universe/Using_Deep_Learning_to_Annotate_the_Protein_Universe.ipynb)

PS : 对于不同长度的MSA怎么办？-----在线学习，一个MSA一个MSA的丢进去算梯度累计，最后一起进去step()



> `torch.one_hot(tensor, num_classes=- 1)`
>
> - **tensor** (*LongTensor*) – class values of any shape.
> - **num_classes** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Total number of classes. If set to -1, the number of classes will be inferred as one greater than the largest class value in the input tensor
>
> input (*)
>
> output(*,num_classes)



- **sequence mask**：防止标签泄露，如：Transformer decoder中的mask矩阵，BERT中的[Mask]位，XLNet中的mask矩阵等

  > 通过上三角矩阵实现

PS：padding mask 和 sequence mask非官方命名,在transformer decoder中两种mask需要结合起来实现



Sequence mask需要通过一个上三角矩阵来实现

```python
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0 #对应True为不需要mask的部分
```

```python
# tensor.masked_fill(mask,value)
# mask矩阵必须可以和value矩阵广播
# mask矩阵必须是一个boolean类型的矩阵
# eg. Attention

def attention(query, key, value, mask=None, dropout=None): 
# query, key, value的形状类似于(30, 8, 10, 64), (30, 8, 11, 64), 
#(30, 8, 11, 64)，例如30是batch.size，即当前batch中有多少一个序列；
# 8=head.num，注意力头的个数；
# 10=目标序列中词的个数，64是每个词对应的向量表示；
# 11=源语言序列传过来的memory中，当前序列的词的个数，
# 64是每个词对应的向量表示。
# 类似于，这里假定query来自target language sequence；
# key和value都来自source language sequence.
  "Compute 'Scaled Dot Product Attention'" 
  d_k = query.size(-1) # 64=d_k
  scores = torch.matmul(query, key.transpose(-2, -1)) / 
    math.sqrt(d_k) # 先是(30,8,10,64)和(30, 8, 64, 11)相乘，
    #（注意是最后两个维度相乘）得到(30,8,10,11)，
    #代表10个目标语言序列中每个词和11个源语言序列的分别的“亲密度”。
    #然后除以sqrt(d_k)=8，防止过大的亲密度。
    #这里的scores的shape是(30, 8, 10, 11)
  if mask is not None: 
    scores = scores.masked_fill(mask == 0, -1e9) 
    #使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，
    #然后在下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视
  p_attn = F.softmax(scores, dim = -1) 
    #对scores的最后一个维度执行softmax，得到的还是一个tensor, 
    #(30, 8, 10, 11)
  if dropout is not None: 
    p_attn = dropout(p_attn) #执行一次dropout
  return torch.matmul(p_attn, value), p_attn
#返回的第一项，是(30,8,10, 11)乘以（最后两个维度相乘）
#value=(30,8,11,64)，得到的tensor是(30,8,10,64)，
#和query的最初的形状一样。另外，返回p_attn，形状为(30,8,10,11). 
#注意，这里返回p_attn主要是用来可视化显示多头注意力机制。
```



11. zip函数----每次循环返回多个对象

**zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同

```python
a = [1,2,3]
b = [1,2,3]
zip(a,b)#得到一个zip对象 list(zip(a,b))为[(1,1),(2,2),(3,3)]
```





12. conda基础命令

```text
conda update -n base conda        //update最新版本的conda
conda create -n xxxx python=3.5   //创建python3.5的xxxx虚拟环境
conda activate xxxx               //开启xxxx环境
conda deactivate                  //关闭环境
conda env list                    //显示所有的虚拟环境

登陆结点上开启
module load anaconda/3
source activate xxx
```



13. python获取命令行

```python
sys.argv //表示命令行参数的数组
```





14. Pytorch最难的函数`torch.tensor.scatter_(dim,index,src,reduce=None)`

Eg.将label转化为one-hot编码---可以用torch.functional.one_hot

```python
label = torch.tensor([0,1,0,2,3,2]).view(len(label),-1)
src = torch.ones_like(label)
temp = torch.zeros((len(label),label_size))
x_oneHot = temp.scatter_(dim=1,index=label,src=src)
```

含义：

- _代表这个函数是一个原地操作
- 一共有三个张量：self(执行scatter的张量)、index(进行索引的张量)、src(要被scatter的张量)

```python
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```





15. os module

- `os.path.join`

  ```python
  dir = "/Users/sirius/Desktop" # 也可以是 /Users/sirius/Desktop/
  os.path.join(dir,"test.txt") 
  # /Users/sirius/Desktop/test.txt
  ```

- `os.path.exists`



16. 回顾Dataset与Dataloader

- Dataset class的长度为数据的长度，有 __ len __ .方法来得到整个数据集的长度，有 __ getitem __ 方法进行下标索引，本质上为一个可以索引的对象
  - 每次索引出来的结果为一个二元tuple对应着一个样本的特征和标签，tuple的第一个元素为tensor，形状即为图片的形状（比如3 * 64 *64），第二个元素为标签(可以为一个int)
- Dataloader class的长度为 数据的长度/batch_size取上界，是一个不可索引，但是有iter方法的可迭代对象
  - 每次迭代出来的结果为一个二元tensor的list， 第一个tensor代表来minibatch的数据特征 (minibatch, a, b, c)，是一个四维的tensor;第二个tensor是minibatch的数据标签,是一个一维的tensor （minibatch）



17. tensor visualization 

从2d-tensor向下累积得到3d-tensor

![image-20220220174333502](/Users/sirius/Library/Application Support/typora-user-images/image-20220220174333502.png)

![image-20220220174419769](/Users/sirius/Library/Application Support/typora-user-images/image-20220220174419769.png)



18. **pytorch contiguous**

- `torch.tensor.is_contiguous()` : **Tensor底层一维数组元素的存储顺序与Tensor按行优先一维展开的元素顺序是否一致**

eg.如果一个矩阵通过transpose,permute,并没有改变最初数组的底层按行存储的顺序,并不会额外开辟一个内存空间,只是改变了stride的元信息.原来数组(a,b)访问行相邻只需要1,列相邻只需要a;现在的stride为(a,1)。因此transpose之后的tensor在逻辑上是连续的，在内存上实际是不连续的

- `torch.tensor.contiguous():`额外开辟一个底层一维数组与tensor逻辑行优先展开数组存储方式相同的数据。

```python
a = torch.range(12).reshape(3,4)
b = a.transpose(0,1)
c = b.contiguous()
a.is_contiguous(),b.is_contiguous(),c.is_contiguous()
#True,False,True
a.data_ptr() == b.data_ptr()
#True,代表确实共享了内存,只是stride和shape不相同
b.data_ptr() == c.data_ptr()
#False,代表c额外开辟了内存,stride不相同,shape相同
```

- 应用：

  - 为什么需要contiguous?

    `torch.view()`等方法需要连续的Tensor （由于历史的原因，大家默认规定view就必须共享内存）。后期根据便携性,pytorch提供了`reshape`方法类似于`tensor.contiguous().view(*args)`，如果不关心底层数据是否使用了新的内存，使用reshape更加方便;reshape()可以开辟一个新的空间，也可以利用原有的空间如果原来的数组本来就是连续的

  - tensor是否连续的递归数学定义
    $$
    \forall i=0,...,k-1(i\neq k-1),stride[i]=stride[i+1]*size[i+1]\\
    \and stride[i+1]=1
    $$
    tensor.stride()得到步长,从i=k-2开始进行递归,i=k-1为最后一个维度内存和语义上均连续因此stride应该为1,size[i+1]为对应的维度元素的数量

Eg.C++/Python描述Tensor是否连续的递归版本

```c
int THTensor(isContiguous)(const THTensor *self)
{
  long z=1; //刚开始最后一个维度的stride
  int d;
 	for (d=self->nDimension-1;d>=0;d--)
  {
    if (self->size[d]!=1) //如果size为1就不用判断了
    {
      if (stride[d]==z)
        z*=self->size[d]
      else 
        return 0;
    }
  }
  return 1;
}
```

```python
def isContiguous(tensor):
  z = 1
  d = tensor.dim()-1
  size = tensor.size()
  stride = tensor.stride()
  while d>=0:
    if size[d]!=1:
      if stride[d]==z:
        z*=size[d]
       else:
        return False
    d-=1
	return True
```



eg:

torch.view()可以任意将tensor变为compatible的形状，然而permute只是交换axis而已；但是两者都会使得原来的tensor变为不是contiguous的



18. `torch.sparse` 稀疏矩阵的存储

注意到在Pytorch中`torch.tensor`用来存储多维的元素，底层存储方式为contiguous的存储，但是对于系数矩阵而言大部分都是0，那么如何存储少数非零元素就很重要，常用的数据结构有 COO，CSR/CSC，LIL等

> 1. 实际上系数矩阵也会存储一些0元素
> 2. 除非尺寸特别大，稀疏性特别强的tensor，一般contiguous memory storage（strided tensor later）都是很有效的存储方式

- Sparse COO tensor （Coordinate format）
  - 一般存储元素索引的和元素值组成的tuple
  - 索引tensor (ndim, nse) --- torch.int64
  - 值 tensor (nse,) ----自定义数据类型的内存开销
- Sparse CSR (Compressed Sparse Row)









18. nn.Module的所有模块都是带有minibatch的,只不过nn.Linear,nn.Conv2d不用写而已，你传递进去的数据全部都应该是(minibatch,dim1...)，即保证第一个维度是minibatch的维度

Eg. nn.Flatten(dim=1,dim=-1)就表示除了不把minibatch的维度展开,其余的所有维度全部展开



PS : nn.Flatten()与torch.flatten()的区别。nn.Flatten()从神经网络的角度来理解，默认第0个维度是minibatch,所以start_dim=1;torch.flatten()只考虑当前的张量，start_dim=0



18. equivariance of convolution layer  CNN对于平移等变！

> **CNNs are famously equivariant with respect to translation**. This means that translating the input to a convolutional layer will result in translating the output



19. 二维卷积细节`nn.Conv2d(in_channel,out_channel,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode="zeros")`
    - Kernel_size一般是一个二维的tuple如(3,3)或直接是一个int 3代表(3,3)
    - stride=1实际上也是代表了一个tuple如(1,1)
    - padding可以选择字符串"valid","same"或者一个int表示要padding多少层;实际上也可以是一个tuple表示高和宽
    - padding mode表述除了0之外还可以填充什么数字

二维卷积实际上的weight是四维的$(C_{out},C_{in},K_{H},K_{W})$

可以说一个卷积层只有一个卷积核，卷积核内有一个四维的tensor和$C_{out}$的bias；

个人理解：有$C_{out}$个三维卷积核，每个卷积核对应一个bias

第i个样本的第j个feature map



$(N_i,C_{out_{j}})=bias(C_{out_j})+\sum\limits_{k=0}^{C_{in}-1}weight(C_{out_j},k)\star input(N_i,k)$





20. 计算卷积

- 最简单的卷积 (n,n,3)的一张图片,kernel是(f*f),如果没有padding=0或者padding = "valid"则最后输出是**(n-f+1,n-f+1) * out_channel**
- 如果有padding比如padding的层数是p,则padding后的图片是(n+2p,n+2p,3),最后输出是**(n+2p-f+1,n+2p-f+1)*out_channel**
  - 对于**same padding**,$2p-f+1=0$,结果应该为$p=\frac{f-1}{2}$
- 如果是有strided的卷积操作，最后图片大小应该为: $Lower(\frac{n+2p-f}{s}+1)$





PS:卷积padding的意义：

- 防止图片缩小 : 除非是kernel=1的卷积只改变通道不改变图片的大小，其他的卷积全部会将图片缩小，比如same padding就是通过添加0的操作使得卷积前后图片形状不变
- 考虑边缘信息：不加padding则边缘的像素考虑的不多

PS：池化层的意义

- 调整图片大小
- 减少参数量，防止过拟合





21. Tensor.detach ：detach tensor from the current graph

`torch.detach()`       

- **Returns a new Tensor** with the same storage memory. 如果
- detached from the current graph. The result will never require gradient.

> 用法，可以和clone一起使用， label = ids.clone().detach()

PS : `torch.clone()`并不共享内存

`torch.detach_()`

- Detaches the Tensor from the graph that created it, making it a leaf. Views cannot be detached in-place.



22. Transposed Convolution 上采样、Fractionally-strided convolution、deconvolution

`nn.ConvTransposed(channel_in,channel_out,kernel_size,stride,padding,bias)`

仍然是一种卷积操作，只不过需要经历三个步骤：矩阵填充(内部相邻元素填充s-1个0，外部填充k-p-1个0)、kernel颠倒、按照kernel,s=1,p=0进行卷积操作

意义：对于比较小的图片进行上采样

输出：

- 矩阵填充后形状为：n + (n-1)(s-1) +2*(k-1-p)
- 填充之后kernel上下左右翻转、做s=1,p=0的卷积 -k+1
- 最终形状为(n-1)*s -2p +k



PS：卷积在pytorch中实现是通过构造稀疏矩阵进行矩阵乘法实现卷积操作而不是滑动窗口的方式





23. Dilated Convolution

解决问题：增大感受野 (receptive field)

含义：在kernel两个元素之间增加d-1个0

区分：空洞卷积是给kernel补0, transposed conv是给原图的内部补0

![image-20220226121235346](/Users/sirius/Library/Application Support/typora-user-images/image-20220226121235346.png)

使用方法：可以通过d=1,d=2,d=3,d=2,d=1的设置方法既可以增大感受野，又能避免gridding effect的问题

> 感受野(receptive field):
>
> Def : 第n层特征图的一个元素，对应输入层的区域大小；即feature map上一个单元对应输入层上的区域大小
>
> $r_k = (r_{k-1} -1)\times s_{k-1} +f_{k-1}$
>
> $r_n=1,r_k$表示第k层的感受野





24. Python中正则表达式处理批量数据

内部模块`re`

常用命令：`re.match(pattern,string)`返回None或者一个匹配的对象，必须从开头进行匹配, `re.search(pattern,string)`可以从任何位置开始匹配

Eg.

```python
x = re.match(r"^([\w\.\-]+)\s+.+\s+([0-9]+)\s+([0-9e\-\.]+)\s+[0-9]+$",s)
x.groups()
#通过groups可以得到不同补获组的一个tuple,只进行一次匹配
```

字符串经常的替换命令 

`a=str;a.replace("原来的字符串","要替换成什么")`

PS :

如果要进行多次匹配`re.findall(pattern_re,sequence)    `可以利用pattern查找sequence里的所有匹配字符串放入一个列表



25. Python文件IO基本操作

- 如果是csv文件等直接用pandas `df = pd.read_csv("")`
- 如果是一般的文件则需要`with open(file_path,pattern) as f: line = f.readline()`读取第一行文件 

```python
file = "./chain_set.json"
with open(file,"r") as f:
  for line in f:
    line = line.replace("\n","")#提取每一行文件放入一个字符串
```





26. 3D卷积代表输入除了宽度之外还多了一个深度depth，因此卷积核是在3D的形状下进行滑动；2D卷积代表卷积在二维的情况下进行滑动，最后的channel进行相加

input : $(N,C_{in},D,H,W)$

output : $(N,C_{out},D_{out},H_{out},W_{out})$

对于输入的$(N_i,C_{out_{j}})=bias(C_{out_j})+\sum\limits_{k=0}^{C_{in}-1}weight(C_{out_{j}},k)\star input(N_i,k)$

注意到卷积的weight: $(C_{out},C_{in},K[0],K[1],K[2])$





27. 判断array中符合某个字段的值：用sum!

y == 1将y转化为一个bool类型的数组，接着利用sum可以判断内部True的个数

(y==1).sum(axis=?)

![sklearn算法选择](https://img2018.cnblogs.com/blog/1011838/201901/1011838-20190123203347054-1083715070.png)





```python
def MinibatchAccumulation(traindata,labels,epochoptimizer,criterion,net):
    optimizer.zero_grad()
    for i in range(len(traindata)):
        loss = criterion(net(traindata[i],labels[i]))
        loss.backward()
    optimizer.step()      
```





28. json文件格式---本质是一个列表



```python
{"test": ["1bct.A", "1bf0.A", "1bjx.A", "1cyu.A", "1ehs.A", "1faf.A", "1fuw.A", "1g7e.A", "1gyz.A", "1hp8.A", "1ifw.A", "1ilo.A", "1iur.A", "1kvz.A", "1mkn.A", "1n02.A", "1nho.A", "1on4.A", "1pv0.A", "1qkl.A", "1ss3.A", "1t4y.A", "1u97.A", "1urf.A", "1v9v.A", "1v9w.A", "1vib.A", "1waz.A", "1wjk.A", "1wjz.A", "1wpi.A", "1x4t.A", "1x59.A", "1x5e.A", "1xq8.A", "1xut.A", "1yua.A", "1z7p.A", "1z9i.A", "2a2p.A", "2a4h.A", "2b5x.A", "2ctq.A", "2d9e.A", "2dat.A", "2dbc.A", "2dj0.A", "2dkw.A", "2dlx.A", "2fxt.A", "2h7a.A", "2hky.A", "2i7k.A", "2if1.A", "2j6d.A", "2jmz.A", "2jov.A", "2jua.A", "2jz4.A", "2k0m.A", "2k18.A", "2k3d.A", "2k54.A", "2k6h.A", "2k6v.A", "2k9p.A", "2kaa.A", "2kcd.A", "2km7.A", "2krt.A", "2krx.A", "2kw8.A", "2kx2.A", "2kxg.A", "2kyy.A", "2l1s.A", "2l2f.A", "2l4v.A", "2l57.A", "2lcj.A", "2lja.A", "2ljk.A", "2lkl.A", "2lku.A", "2ln7.A", "2lt5.A", "2ltk.A", "2lus.A", "2lwy.A", "2lyx.A", "2m7o.A", "2mbt.A", "2mc8.A", "2ml5.A", "2myg.A", "2mzb.A", "2pxg.A", "2rsx.A", "2wgo.A", "2y1s.A", "2yqd.A", "2ys8.A", "2yua.A"]}
```



- 读取json文件`json.load(file_object)`,将文件对象的内容变为一个列表

```python
file = ""
with open(file,"r") as f:
  data = json.load(file_object)
```

- 读取json文件`json.loads(string)`,将一个字符串的内容变为一个列表

测试文件

`chain_set_split.json`

```python
{"cath_nodes": {"2fyz.A": ["1.20.5"], "1v5j.A": ["2.60.40"], "4nav.A": ["3.40.50"], "2d9e.A": ["1.20.920"],..,}, "test": ["3fkf.A", "2d9e.A", "2lkl.A", "1ud9.A", "2rem.B", "2d8d.B", "1kll.A", "1ifw.A", "1buq.A", "1lu4.A",...,] , "train":[],"validation":[]}
```

- 如何读取一个json列表？---一个文件里有许多个字典,观察不同的字典有什么区别

```python
file1 = "./chain_set.jsonl"
data = []
with open(file1,"r") as f:
    for line in f:
        try:
            data.append(json.loads(line.replace("\n",""))) #每一行代表一个序列的字典字符串加一个'\n',对于字符串应该用json.loads()文件进行读取
        except ValueError:
            pass

```

PS : 进行蛋白质设计，给定一个jsonl 列表 + 一个json字典包含所有蛋白质的名字进行输入，直接执行一次python





29. GAN训练技巧小结：

- 因为要分别训练两个网络，需要各自两个optimizer对应这两个网络的参数
- 观察到对 f(x) 期望的梯度等于对f(x)自身的梯度
- 在训练D时：
  - Theoretically : $\max\limits_{D}  E_{x}[\log D(x)] + E_{z}[\log (1-D(G(z)))]$
  - NN 
    - 先计算Real_data  ：$loss_{real} = -\sum\limits_{i=1}^{batch}log(D(x^i))$----------标签为real label
    - 再计算Fake_data  :  $loss_{fake} = -\sum\limits_{i=1}^{batch}log(1-D(G(z^{i})))$----------标签为fake label

- 训练G时
  - theoretically ： $\min\limits_{G}E_z [log(1-D(G(z)))]$,  $\max\limits_{G}E_z [logD(G(z))]$  
  - NN中最小化将G(z)的标签全部设置为real, 最小化损失 $loss = -\sum\limits_{i=1}^{batch}log(D(G(z^{i})))$



30. `torch.fill(size:tuple,value,dtype,device)`



31. stack是新增加一个维度，cat是在原有维度上拼接

`np.stack`

`torch.stack(tensors,dim=0)`

- **tensors** (*sequence of Tensors*) – sequence of tensors to concatenate
- **dim** ([*int*](https://docs.python.org/3/library/functions.html#int)) – dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive)

输入一个seqence tensor(ndarray)，增加一个新的维度，个数是seqeunce的数目

> concat是在原有的维度上进行拼接，stack是增加一个新的维度



32. `argparse` module, 自动从sys.argv列表里数据

常用的三条命令；

```python
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument()
args = parser.parse_args()
return args #返回的是一个包含所有参数的对象
```

- 位置参数：名称不带"--"，如果add_argument()后必须要添加

```python
parser.add_argument("square", help="display a square of a given number",type=int)
```

- 可选参数：“--abc”，‘abc'表示在args这个对象中的名称，也可以包含简写"-"
  - type表示将字符串转化为什么类型的变量,一般有int等
  - 如果设定了可选参数但是没有输入，则args里该变量为None；也可以通过添加**default**的方式来更改不输入时的默认值
  - 如果想将这个可选参数变成一个flag，则应当设置**action**="store_true"表示只需要指定这个标签就为True，不需要传入额外的值
  - 可以设置**choice**= [1,2,3,4]表示限制输入的范围
  - **action**=“count”表示按照输入字符 出现的频率来统计，不输入为None，输入一次为1，两次为2以此类推



33. `np.nan` 与 `np.inf` ,`torch.nan`



33. 得到一个True False矩阵

- np.nan == np.nan永远为False
- `np.isnan()`值为nan的时候为True
- `np.isinf()` 值为inf的时候为True
- `np.isfinite()` 值不是nan或者inf的时候为True

- 忽略nan求值

```python
nansum()
nanmax()
nanmin()
nanargmax()
nanargmin()

>>> x = np.arange(10.)
>>> x[3] = np.nan
>>> x.sum()
nan
>>> np.nansum(x)
42.0
```

PS : 在numpy中更改数据类型`x = x.asstype(np.float32)`



34. Python GIL ： Global Interpretor Lock 

要解释Python全局解释器锁，首先要知道进程process和线程thread

> Process : A program in execution.(the [instance](https://en.wikipedia.org/wiki/Instance_(computer_science)) of a [computer program](https://en.wikipedia.org/wiki/Computer_program))
>
> Thread:  A thread is the unit of execution within a process. A process can have anywhere from just one thread to many threads.

**GIL**

> In **CPython**, the Global Interpreter Lock (GIL) is a mutex that allows only one thread at a time to have the control of the Python interpreter.  In other words, the lock ensures that **only one thread is running at any given time**. 

**为什么有GIL？**

> Since the **CPython’s memory management is not thread-safe**, the **GIL** prevents race conditions and **ensures thread safety**.Threads in Python share the same memory — this means that when multiple threads are running at the same time we don’t really know the precise order in which the threads will be accessing the shared data.

**什么是race condition**

> a race condition occurs when the behaviour of the system or code relies on the sequence of execution that is defined by uncontrollable events.

**Threading vs Multi-processing in Python**

> As mentioned already, a Python process cannot run threads in parallel but it can run them concurrently through context switching during I/O bound operations.
>
> 单CPU并发运行
>
> ![截屏2022-05-14 下午10.09.21](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-05-14 下午10.09.21.png)
>
> 多核进程
>
> ![截屏2022-05-14 下午10.10.10](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-05-14 下午10.10.10.png)



Pytorch中DistributedDataParallel为multiprocessing，为每一个GPU创建一个进程；然而DataParallel为一个进程创建多个线程，每一个GPU是一个线程，性能很容易受到GIL带来的影响。









35. 维度与采样

> **维度灾难 curse of dimensionality** 
>
> - 当维度非常高的时候，数据分布变得特别稀疏，同时两两之间的距离也变得十分相近，因此很难分类或者找nearst neighbor，在高维度采样也十分困难，现在大部分都是基于MCMC采样
>
> 
>
> **如何有效的找Nearest Neighbor, N sample , dim = D**
>
> - Brute Force : $O(D N^2)$
>
> - KD-tree (k dimension tree) : $O(DN\log(N))$,维度很高时与Brute Force一样，甚至由于树数据结构，整体搜索的速度会变慢
>
>   > 如果是二维数据就用 quadtree,三维就oct-tree 
>
> - Ball-tree : 不用树来表示，用超球体来表示
>
> 
>
> **NearestNeighbor方法有什么用？**
>
> > 方法来看：
> >
> > 1. K-NN 
> > 2. Radius NN 
> >
> > 标签来看
> >
> > 1. Classification : majority vote
> > 2. Regression : average



36. Simulated Annealing

- **Principle :** [Metaheuristics](https://en.wikipedia.org/wiki/Metaheuristic) use the neighbours of a solution as a way to explore the solution space, and although they prefer better neighbours, they also accept worse neighbours in order to avoid getting stuck in local optima; they can find the global optimum if run for a long enough amount of time.
- **Definition :** SA is a probabilistic technique for approximating the global optimum of a given function.  Specifically, it is a metaheuristic to approximate global optimization in a large search space for an optimization problem. 
- **When should we use it :**
  - It is often used when the search space is discrete and the objective function doesn't have gradient with respect to variable.
  - For problems where finding an approximate global optimum is more important than finding a precise local optimum in a fixed amount of time

- **Pseudocode:**

  ![截屏2022-06-22 上午10.54.45](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-22 上午10.54.45.png)

- **Parameters:**
  - state space
  - energy function `E(r)`
  - candidate generator procedure `neighbor()`
  - acceptance probability function `P()`
  - annealing schedule `temperature()`
  - initial temperature `T`



37. Hill climbing

- Greedy algorithm (heuristics) which move by finding better neighbor after neighbor and stop when they have reached a solution which has no neighbors that are better solutions.



38. `torch.multinomial()` 对每一行进行离散多项式抽样返回下标

`torch.multinomial(input,num_samples,replacement-False) -> LongTensor`

- 输入的矩阵每一行的元素都代表weights（不要求weights和为1，但必须非负，有限，并且和非零）

- 输出的矩阵每行包含num_samples个下标

  - 如果输入是一个 vector，输出也是个vector长度为num_samples

  - 如果输入为一个长度为m*n的矩阵，输出为 m * num_samples的矩阵

  - 如果replacement = True，则为有放回的抽样；默认为无放回的抽样，对于那些value = 0的位置只有最后才会抽到

    PS：无放回抽样的时候，num_samples数必须小于每一行的元素数

39. `torch.repeat_interleave(input,repeats,dim)`

- 类似`np.repeat()` ,都是对每个元素的操作，而`tensor.repeat()`和`np.arrary.repeats()` 一样，是对整体的操作
- parameter:
  - Input : 输入张量
  - repeats : int 或 一维Tensor **(repeats must have the same size as input along dim)**
  - dim : 对哪个轴进行repeats， 默认将原来张量拉平成1维再对所有元素repeat

PS：难以理解的点是 repeats 先fit 到dim维度上,然后再广播到其他元素

## Loss函数

1. CrossEntropyLoss : 从logiti -> softmax -> CrossEntropy

> `torch.nn.``CrossEntropyLoss`(*weight=None*, *size_average=None*, *ignore_index=- 100*, *reduce=None*, *reduction='mean'*, *label_smoothing=0.0*)
>
> - input x:
>   - [Batch, C ]
>   - [Batch, C, d1, d2, ..., dk]
>     - 一共有k个分类
>     - computing cross entropy loss per-pixel for 2D images.
>
> - target y : 一般而言都不是概率，因此没有C那个index维度
>
>   - [Batch, ]
>   - [Batch, d1, d2, ..., dk]
>
> - 计算方式
>
>   - `reduction=none`
>
>     > 最后输出一个每个元素cross entropy的向量，可以添加weighted和ignore
>     >
>     > $l(x,y)=L=\{l_1,...,l_N\}^T,l_n = -\omega_{y_n}\log \frac{\exp(x_{n,y_n})}{\sum\limits_{c=1}^C\exp (x_n ,c)} \times 1 \{y_n \neq ignore\}$
>
>   - `reduction='mean' ` or `reduction='sum'`
>
>     > ![image-20220604112633417](/Users/sirius/Library/Application Support/typora-user-images/image-20220604112633417.png)
>
>   - 对于添加了label smoothing的标签而言
>
>     > ![截屏2022-06-04 上午11.28.16](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-04 上午11.28.16.png)
>
> 为什么CrossEntropy更优化？一般情况target采用索引计算而不是将其看成概率分布
>
> - The performance of this criterion is generally better when target contains class indices, as this allows for optimized computation. Consider providing target as class probabilities only when a single class label per minibatch item is too restrictive.





## Torch.utils

### Torch.utils.Data

在搭建tmpnn的过程中，需要自己定义数据集Dataset和DataLoader,先学习一下pytorch官网的简介

问题1：自定义dataset

问题2:  自定义dataloader

----

Pytorch loading data最重要的一个参数就是`torch.utils.data.DataLoader`，它是一个python的可迭代对象，支持

- map-style and iterable-style datasets
- customizing data loading order
- automatic batching
- single- and multi-process data loading
- automatic memory pinning

主要由来完成

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

- Dataset Types

  - Map-style datasets （大部分的定义方式）

    > 实现了_ _ getitem _ _ ()和 _ _ len _ _ ()函数，表示一个映射 （从一个整数或非整数索引到数据样本）。比如对于一个数据集，通过dataset[idx]来映射到第idx个样本

  -  Iterable-style datasets

    > Utterable-style dataset是IterableDataset的一个子类，实现了 _ _ iter _ _ ()函数

- Data Loading Order and `sampler`



----





## CV

一般图像任务的下采样倍数为H/32,W/32, 若为224 * 224 则变化位 7 * 7

### 图像分类模型

1. AlexNet-2012

- **端到端的预测**，只对输入进行图像增强后直接输入像素预测类别
- 首次利用GPU进行网络加速
- 使用ReLU (Rectified Linear Unit线性修正单元)
  - 容易计算和求导
  - 不存在饱和区域，不存在梯度消失
- 使用Dropout防止过拟合，提升泛化性能



2. VGG - 2014

- 堆叠多个3*3卷积核来替代大卷积核来减少所需参数，同时保证receptive field大小 (2个3 * 3对应一个5 * 5，3个3 * 3对应一个 7 * 7)

> VGG-16：
>
> - input : 224 * 224 *3，conv 为3 * 3,padding 1 stride 1, maxpool stride=size=2
> - network:
>   - conv3 - 64 --- 224 * 224 * 64
>   - conv3 - 64
>   - **maxpool** --- 112 * 112 * 128
>   - conv3 - 128
>   - conv3 - 128
>   - conv3 - 128
>   - **maxpool** --- 56 * 56 * 256
>   - conv3 - 256
>   - conv3 - 256
>   - conv3 - 256
>   - **maxpool** --- 28 * 32 * 512
>   - conv3 - 512
>   - conv3 - 512
>   - conv3 - 512
>   - **maxpool** --- 14 * 14 * 256
>   - conv3 - 512
>   - conv3 - 512
>   - conv3 - 512
>   - **maxpool** --- 7 * 7 * 512
>   - FC- (7 * 7 * 512,4096) + ReLU
>   - FC- (4096,4096) + ReLU
>   - FC- (4096, 1000) + Softmax
>
> <img src="/Users/sirius/Library/Application Support/typora-user-images/image-20220603224726737.png" alt="image-20220603224726737" style="zoom:50%;" />
>
> 



3. GoogleNet - 2014

- **引入Inception结构** （融合不同尺度的特征信息）

  > 并行化的结构
  >
  > ![截屏2022-06-03 下午11.02.25](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-03 下午11.02.25.png)

- 使用1*1的卷积核来帮助改变feature map的通道数，减少计算量

- 添加两个辅助分类器帮助训练 （一共有三个输出层）

  > Auxiliary Classifier

- 丢弃全连接层，只用平均池化层







0. ConvNext - 2020 从ViT 回到卷积



### 图像分割

1. Image segmentation

   > The term “image segmentation” or simply “segmentation” refers to dividing the image into groups of pixels based on some criteria.

   - Semantic Segmentation (object detection) 语义分割---从class的角度对全图pixel分类从而达到分割

     > In Semantic Segmentation the goal is to assign a label (car, building, person, road, sidewalk, sky, trees etc.) to **every pixel in the image**.  
     >
     > 我们可以看出每个像素属于哪一类，但是不能分辨到底是不是同一个object

     Eg1.U-Net(2015 ) Encoder 下采样，Decoder 上采样 最终在原图大小预测前景和背景

     Eg2. FCN (Fully Convolutional Networks for Semantic Segmentation)---**首个end2end, Pixelwise prediction，将全连接层全部转化为卷积层，就可以处理不同大小的数据，首先可以有一个VGG backbone当作encoder，decoder可以分别由FCN-32S,FCN-16S,FCN-8S构成**

   - Instace Segmentation 实例分割---从object的角度对部分像素检测来绘制某些物体轮廓

     > Instance Segmentation is a concept closely related to Object Detection. However, unlike Object Detection, the output is a mask (or contour) containing the object instead of a bounding box. Unlike Semantic Segmentation, we do not label every pixel in the image; we are interested only in finding the boundaries of specific objects.

     Eg. Fast R-CNN,  Mask R-CNN

   - Panoptic Segmentation 全景检测

     > Panoptic segmentation is the combination of Semantic segmentation and Instance Segmentation. Every pixel is assigned a class (e.g. person), but if there are multiple instances of a class, we know which pixel belongs to which instance of the class.





# DISTRIBUTED DATA PARALLEL IN PYTORCH 

Pytorch DDP : Distributed Data Parallel

核心

- Distributed Sampler给每一个device一个non-overlapping input batch
- The model is replicated on all the devices; each replica calculates gradients and simultaneously synchronizes with the others using the [ring all-reduce algorithm](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/). (同步更新)

>**Ring all-redue algorithm**
>
>In synchronized data-parallel distributed deep learning, the major computation steps are:
>
>1. Compute the gradient of the loss function using a minibatch on each GPU.
>2. Compute the mean of the gradients by inter-GPU communication.
>3. Update the model in all GPUs.
>
>To compute the mean, we use a collective communication operation called “AllReduce.”  As of now, one of the fastest collective communication libraries for GPU clusters is NVIDIA Collective      Communication Library: NCCL. It achieves far better communication performance than MPI, which is the de-facto standard communication library in the HPC community.
>
>- There are several algorithms to implement the operation. For example, a straightforward one is to select one process as a master, gather all arrays into the master, perform reduction operations locally in the master, and then distribute the resulting array to the rest of the processes. Although this algorithm is simple and easy to implement, it is not scalable. The master process is a performance bottleneck because its communication and reduction costs increase in proportion to the number of total processes. the amount of communication of the master process is proportional to P.
>
>- Faster and more scalable algorithms have been proposed. They eliminate the bottleneck by carefully distributing the computation and communication over the participant processes.
>  Such algorithms include Ring-AllReduce and Rabenseifner’s algorithm
>
>  - Ring-all-reduce是一个同步更新来解决网络瓶颈问题的方法，每个 GPU 只从左邻居接受数据、并发送数据给右邻居。
>
>    - scatter-reduce：会逐步交换彼此的梯度并融合，最后每个 GPU 都会包含完整融合梯度的一部分
>    - allgather：GPU 会逐步交换彼此不完整的融合梯度，最后所有 GPU 都会得到完整的融合梯度
>
>    https://zhuanlan.zhihu.com/p/69797852



single machine mp.spawn code

```python
import torch.multiprocessing as mp # a wrapper around python's native multiprocessing
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP# the main workhorse
from torch.distributed import init_process_group, destroy_process_group #initial and destroy groups
import os

def ddp_setup(rank,world_size):
  """
  initial the group.
  Arguments:
  	rank : the unique identifier that is assigned to each process,usually 0, 1, 2, world_size-1
  	world_size : the total number of process to each group
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  init_process_group(backend="nccl",rank=rank,world_size=world_size)


# before initialize the model class, using ddp to wrap it 
gpu_id = list(range(torch.cuda.device_count()))
model = DDP(model,device_ids=gpu_id)

# the current model has been wrapped by ddp
# load the parameters
ckp = self.model.module.state_dict()

# save the model, only save 1 copy
if model.gpu_id == 0 :
  torch.save(ckp,"./ckp.pt")

# change the dataloader with distributed sampler and shuffle to false
DataLoader(dataset,batch_size=batch_size,shuffle=False,sampleer=DistributedSampler(dataset))

def main(rank,world_size):
  """
  第一个参数总要是rank
  """
# initial the group
  ddp_setup(rank,world_size)
  
  # running the code
  
  # destroy the group
  destroy_process_group()

if __name__ == "__main__":
  mp.spawn(main, args=(world_size),nprocs=wold_size) # given a function and spawn to all the process,注意不需要提供rank
```



single machine fault tolerance with torchrun，注意到torchrun将mp.spawn的所有变量全部使用系统变量，因此不用在python代码中体现

```python
import torch.multiprocessing as mp # a wrapper around python's native multiprocessing
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP# the main workhorse
from torch.distributed import init_process_group, destroy_process_group #initial and destroy groups
import os

def ddp_setup():
  init_process_group(backend="nccl")

gpu_id = init(os.environ["LOCAL_RANK"])

```



