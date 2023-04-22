# Bioinformatics  

[toc]

## Deep Lerning Blocks

### System

预备知识：

常见的backend通讯方式：MPI、NCCL

1. NCCL operations

All, reduce , gather

- AllReduce

The AllReduce operation is performing reductions on data (for example, sum, max) across devices and writing the result in the receive buffers of every rank.

**The AllReduce operation is rank-agnostic**. Any reordering of the ranks will not affect the outcome of the operations.![截屏2023-04-22 16.36.33](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-04-22 16.36.33.png)

- Brodcast

The Broadcast operation copies an N-element buffer on the root rank to all ranks.

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.![截屏2023-04-22 16.37.26](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-04-22 16.37.26.png)



- Reduce

The Reduce operation is **performing the same operation as AllReduce, but writes the result only in the receive buffers of a specified root rank.**

Important note : The root argument is one of the ranks (not a device number), and is therefore impacted by a different rank to device mapping.

PS : Reduce + Brodcast = AllReduce

![截屏2023-04-22 16.38.37](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-04-22 16.38.37.png)



- AllGather

In the AllGather operation, each of the K processors aggregates N values from every processor into an output of dimension K*N. The output is ordered by rank index.

![截屏2023-04-22 16.44.33](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-04-22 16.44.33.png)

The AllGather operation is impacted by a different rank or device mapping since the ranks determine the data layout.



#### Megatron-LM

- 提出了一种单机多卡的模型并行（层内张量并行）的方法，非常简单；只能针对标准架构的Transformer设计，比如GPT2和Bert
- 对MLP、Multihead Attention、Embedding分别进行了模型的拆分，核心思想是矩阵分块



### Attention is all you need

Basic Blocks

1. position encoding 来保证模型不是permutaiton equivariant

> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add “positional encodings” to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed.
>
> In this work, we use sine cond cosine functions of different frequencies:
> $$
> PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})\\
> PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})\\
> $$
> Where "pos" is the position from 0 to max_len -1 , and i is the dimension from 0 to d_model(512)
>
> ```python
> # code in the paper
> class PisitionalEncoding(nn.Module):
> def __init__(slef,d_model,dropout,max_len=5000):
>  super().__init__()
>  self.dropout = nn.Dropout(p=dropout)
> 
> #compute the positional encodings once in log space.
>  pe = torch.zeros(max_len,d_model)
>  position = torch.arange(0,max_len).unsqueeze(1) #[max_len, 1]
>  div_term = torch.exp(
>    torch.arange(0,d_model,2) * (- (math.log(10000.0)/d_model))
>  ) # [d_model/2]
>  pe[:,0::2] = torch.sin(position * div_term) # broadcast : [max_len, 1]* [d_model/2] -> [max_len, d_model/2] 
>  pe[:,1::2] = torch.cos(position * div_term)
>  pe = pe.unsqueeze(0) # add batch dimension
>  self.register_buffer("pe",pe)
> 
> def forward(self,x):
>  x = x+ self.pe[:,:x.size(1)].requires_grad_(False)
>  return self.dropout
> 
> # code in fairseq in line with tensor2tensor
> @staticmethod
> def get_embedding(
>     num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
> ):
>     """Build sinusoidal embeddings.
>     This matches the implementation in tensor2tensor, but differs slightly
>     from the description in Section 3.5 of "Attention Is All You Need".
>     """
>     half_dim = embedding_dim // 2
>     emb = math.log(10000) / (half_dim - 1)
>     emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
>     emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
>         1
>     ) * emb.unsqueeze(0)
>     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
>         num_embeddings, -1
>     )
>     if embedding_dim % 2 == 1:
>         # zero pad
>         emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
>     if padding_idx is not None:
>         emb[padding_idx, :] = 0
>     return emb



2. Optimizer

> We used the Adam optimizer with $\beta_1 = 0.9, \beta_2=0.98,\epsilon=10^{-9}$. We varied the learning rate over the course of training, according to the formula :
> $$
> lrate = d_{model}^{-0.5}\times\min(step\_num^{-0.5}, step\_num \times warmup\_steps^{-1.5})
> $$
> 注意：可以通过两种方式定义warmup 1.只定义线性增长的步数, 和transofrmer原文里一样通过model size控制lr增长  2.定义线性增长步数的同时定义最大lr ，完全自定义lr增长
> 需要的pytorch函数
>
> ```python3
> # transformer warmup
> 
> def transformer_rate(step,warmup,model_size):
>     if step == 0:
>         step = 1
>     return model_size ** (-0.5) * min(step**(-0.5),step * warmup ** (-1.5))
> 
> lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: transformer_rate(step,warmup,model_size))
> ```
>



3. Decoder的细节

> 1. 注意到有tgt处有两种mask : padding mask &  std mask (避免看到未来的单词)
> 2. Decoder的输入是tgt[:,:-1], 标签是tgt[:,1:]
>
> e.g. 完整的tgt是`<bos>,<I>,<love>,<deep>,<learning>,<.>,<eos>` 长度为N
>
> 输入就是`<bos>, <I>, <love>,<deep>,<learning>,<.>`  长度为N-1每一个token对应预测的值为:
>
> ​                `<I>, <love>,<deep>,<learning>,<.>,<eos>`    标签，长度也为N-1
>
> 所以future words mask是可以看到自己的，根据自己来预测下一个token
>
> 3. Decoder 重复N层 (GPT)
>
> ```python
> class Batch:
>     """Object for holding a batch of data with mask during training."""
> 
>     def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
>         self.src = src #[B,L_s]
>         self.src_mask = (src != pad).unsqueeze(-2) #[B, 1, L_s] 表示padding mask
>         if tgt is not None:
>             self.tgt = tgt[:, :-1] #decoder input不用考虑最后一个token <eos>
>             self.tgt_y = tgt[:, 1:] #deocoder label不用考虑第一个token <bos>
>             self.tgt_mask = self.make_std_mask(self.tgt, pad) #decoder mask : padding mask + mask future words
>             self.ntokens = (self.tgt_y != pad).data.sum()
> 
>     @staticmethod
>     def make_std_mask(tgt, pad):
>         "Create a mask to hide padding and future words."
>         tgt_mask = (tgt != pad).unsqueeze(-2) #[B, 1, L_t]
>         tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
>             tgt_mask.data
>         ) #[B, 1, L_t] & [1, L_t, L_t] = [B, L_t, L_t]
>         return tgt_mask
> 
> def subsequent_mask(size):
>     "Mask out subsequent positions."
>     attn_shape = (1, size, size)
>     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
>         torch.uint8
>     )
>     return subsequent_mask == 0
> 
> ```
>
> 



### Codex

**Q1 : average log p是推理的时候直接拿来，还是拿推理的序列进行teacher forcing forward 计算perplexity** 

> ProGPT2, ProteinMPNN, Graph transformer,ESM inverse folding 评估生成序列的 mean log probability 都是拿采样的序列再重复做一次forward, 生成N条序列：sample 函数 N次，forward函数N次（teacher forcing）
>
> ```python3
> sequence = model.sample(x) #[B,N]
> batch , seuqnce_len = sequence.size()
> ntokens = 20
> with torch.no_grad():
>   model = model.eval()
>   logits = model(x, sequence) #[B, N, 20]
>   loss = CrossEntropy(logits.reshape(-1,ntokens), logits.reshape(-1))  #计算token层面的loss, 先平均batch和token, 再计算teacher forcing的损失函数
> ppl = torch.exp(loss)
> ```

总结：利用GPT + 爬取的github公开数据集(docstring + code)

1. **微调不一定会带来精度上的提高，但是大部分时候都会让模型的收敛的更快, 所以能微调还是尽量微调**

2. Codex选取来164个任务(Human Eval)，利用pass@k的方法来表示正确率，即对于每个任务生成k条代码，有1条能通过任务就可以，如果反复生成k个方法取评价，那么这个分数variance会非常大

   - 作者来估算pass@k:   先在每个任务生成n个样本  $n\ge k$ (论文里选了n=200) ,利用一个无偏估计的期望来计算 : 1个问题生成n个答案，有c个是正确的，假设每次抽取k个，则$\frac{\binom{n-c}{k}}{\binom{n}{k}}$ 表示选取的k个答案没有一个是正确答案的结果, $1-\frac{\binom{n-c}{k}}{\binom{n}{k}}$  表示选取的k个答案存在正确答案的概率

   ![截屏2023-03-31 23.10.11](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-03-31 23.10.11.png)

   对于这个二项式计算，整数的相乘很容易超出计算机内定义的范围，作者进行化简来避免数值问题
   $$
   1-\frac{\binom{n-c}{k}}{\binom{n}{k}}=1-\frac{(n-c)!}{k!(n-c-k!)}\times\frac{k!(n-k)!}{n!}=1-\frac{(n-c)!(n-k)!}{(n-c-k)!n!}=1-\Pi_{i=n-c+1}^{n} \frac{1}{i} \times \Pi_{i=n-c-k+1}^{n-k}i
   $$
   统一后两项的下标$1-\Pi_{i=n-c+1}^{n} \frac{1}{i} \times \Pi_{i=n-c-k+1}^{n-k}i=1-\Pi_{i=n-c+1}^{n}\frac{i-k}{i}$

   ```python
   def pass_at_k(n,c,k):
     """
     np format pass@k numerical stable computation
     """
     if n - c < k : return 1.0 #every time 
   	return 1.0 - np.prod(1.0 - k/np.arange(n-c+1,n+1))
   ```

   

3. 采样的方法：top k sample, nucleus sample top p (加起来的概率大于p之后丢弃其他选项，保证多样性的同时避免了很不靠谱的答案)

4. 如果允许采样的数目比较少，温度低一点好；采样的数目比较多，温度高一点好

5. **采样出来之后利用avetage log p进行排序**

6. Codex 的微调并不是在最后换一个线性头，而是针对同一个任务同一个语言模型同一个损失函数，只不过数据集变成了代码问答数据集而不是Github爬取的数据集



### AlphaCode

**Q1 : 头数目不一样怎么做到的?**

- **非标准的attention, key-value的头和query的头数量不一样, 有效的提高了运行速度**
- 解码器是编码器的6倍
- DeepMind 习惯使用一个学习的网络来评估生成结果的置信度： Value Conditioning and predcition
  - 数据集有正确的答案，有错误的答案，训练的时候把正确与否当成一个tag进行输入，同时输出一个预测的二分类正确与否
  - 采样的时候tag全部设置为正确答案，过滤的时候只考虑那些生成答案为正确 的结果
- AlphaCode的采样和排序 
  - 采样 : 标准的random sample (相比较Codex每个问题生成10-100个答案，AlphaCode可能一次会生成1M个答案，所以已经能够保证采样的多样性了，因此温度也设置的比较低T=0.1)
  - 排序：预测的结果首先为正确，再通过强化学习进行判断 

### MoCo

Momentum Contrast for Unsupervised Visual Representation Learning

总结：利用单模态图片的未标注的数据集，通过对比学习进行无监督预训练，在下游任务上的表现超过了ImageNet有监督预训练任务

对比学习：只要定义好代理任务（什么是正负样本）+ Loss 函数（如何处理正负样本的Loss）

- 目前大部分的对比学习任务，主要受限于字典的大小 + key encoder有效性上

- End-to-End 同时更新encoder + key encoder

  - 每次字典的大小为mini-batchsize大小

  > 对于谷歌不是问题，TPU的使用可以无限增大batch size的大小 

  - 优点是key encoder可以实时更新，字典的值也可以实时更新，有效性比较高

- memory bank

  - 不考虑key encoder，提前用一个预训练好的网络得到所有key encoder特征，每次随机sample一些特征作为副样本

- Moco

  - 创立一个queue 来存储negative key的大小，将mini-batchsize和字典的大小分离开，同时由于队列FIFO的特点，去除队列的特征往往是比较老的特征，以及失去了一致性
  - 由于queue 的使用导致key encoder模型无法求梯度，尝试直接将encoder参数直接拷贝给key encoder
  - 利用动量的方法来更新key encoder $\theta_k = m\theta_k + (1-m)\theta_q$ ，使得队列里整体的正负样本 具有很好的一致性
  - other
    - Loss函数相当于一个 (K+1) 类的softmax (controlled by temperature)



![image-20230330120249015](/Users/sirius/Library/Application Support/typora-user-images/image-20230330120249015.png)



idea : 目前很火的方向更多是在考虑机器学习任务本身、Loss函数的设置、推理样本的生成等等，对模型架构的关心反而更少

```python
# f_q, f_k: encoder networks for query and key
# queue: dictionary as a queue of K keys (CxK)
# m: momentum
# t: temperature
f_k.params = f_q.params # initialize
for x in loader: # load a minibatch x with N samples
   x_q = aug(x) # a randomly augmented version
   x_k = aug(x) # another randomly augmented version
   q = f_q.forward(x_q) # queries: NxC
   k = f_k.forward(x_k) # keys: NxC
   k = k.detach() # no gradient to keys
   # positive logits: Nx1
   l_pos = bmm(q.view(N,1,C), k.view(N,C,1))
   # negative logits: NxK
   l_neg = mm(q.view(N,C), queue.view(C,K))
   # logits: Nx(1+K)
   logits = cat([l_pos, l_neg], dim=1)
   # contrastive loss, Eqn.(1)
   labels = zeros(N) # positives are the 0-th
   loss = CrossEntropyLoss(logits/t, labels)
   # SGD update: query network
   loss.backward()
   update(f_q.params)
   # momentum update: key network
   f_k.params = m*f_k.params+(1-m)*f_q.params
   # update dictionary
   enqueue(queue, k) # enqueue the current minibatch
   dequeue(queue) # dequeue the earliest minibatch
```









### CLIP

- 框架：对比学习预训练 + prompt template zero shot prediction

![image-20230325123114226](/Users/sirius/Library/Application Support/typora-user-images/image-20230325123114226.png)

- 核心：在40million的图像-文本对 利用**对比学习**进行预训练 得到自然语言的监督信号和视觉的表征，利用Prompt + template 的方式对下游discriminative任务进行zero-short learning

- 技术细节

  - 当前预训练的三种任务
    - autoregressive (language model)
    - masked language model
    - contrastive learning (更适合多模态的预训练)
  - 原来CV的预训练：给定一个大的数据集和label类别，利用图像分类做预训练之后；采样**linear-probe representation learning**的方法，保持骨干网络不变，只修改最后一个linear + softmax 预测层的方法做到下游任务
  - 为什么OpenAI不采用GPT-style来训练CLIP，即给定图片来预测文字
    - 如果采样decoder 方法进行训练，需要精准的预测每一张image对应的word，然而每一张word可能对应多个单词，同时训练起来也比较贵
  - CLIP伪代码

  ``` python
  # image_encoder 		- ResNet or ViT
  # test_encoder	  	- CBOW or Text Transformer
  # I[n, h, w, c] 		- minibatch of alignedimages
  # I[n, l] 					- minibatch of texts
  # W_i[d_i, d_e]			- learned proj of image to embed
  # W_t[d_t, d_e]			- learned proj of text to embed
  # t 								- learned temperature parameter
  
  # extract faeture representations of each modality
  I_f = image_encoder(I)
  T_f = text_encoder(T)
  
  # joint multimodal embedding [n, d_e] 将不同模态的空间信息投影到一个joint space
  I_e =  l2_normalize(np.dot(I_f,W_i), axis=1)
  T_e =  l2_normalize(np.dot(T_f,W_t), axis=1)
  
  # scaled pairwise cosine similarities [n,n], 注意到由于提前使用了归一化, cosine similarities长度即为1
  logits = np.dot(I_e, T_e.T) * np.exp(t)
  
  # symmetric loss fuction
  labels = np.arange(n) # [n,]
  loss_i = cross_entropy_loss (logits,labels,axis=0) #[B,n] vs [B,],image 是dim = 0 为图片，每一行为每一个图片对应的文字match概率
  loss_t = cross_entropy_loss (logits,labels,axis=1) #text是 dim = 1, 每一列为1段文字对应第几个图片
  loss = (loss_i+loss_t)/2
  ```

  - 实验部分

    - 为什么要做prompt  engineering (promt ensembling)

      - polysemy 多义性
      - 训练时只见过句子和图片的配对 而不是单词和图片的配对 (distribution gap)
      - prompt template 能够帮助推理 e.g. "A photo of a {label}, a type of pet"

    - 具体做的实验

      - Zero-shot 不用训练直接推理

      - Few-shot 每个label用1-2-4-8-16个样本 推理训练

      - 全样本数据集进行训练推理

        - linea probe : 冻住骨干网络，只变化线性分类头 ( CLIP模型采样的方法)

        - fine tune : 端到端的网络学习

- 总结：NLP里面利用大规模的数据集进行下游任务无关的训练方式(task-agnostic)



### DALLE2

> Hierarchical Text-Coniditional Image Generation with **CLIP Latents**
>
> 使用CLIP训练好的特征，来做层级式的(先生成小分辨率模型不断上采样)，依托于文本的图像生成

**核心是利用CLIP的text embedding 或image embedding当classifier free guidance** 

- OpenAI  文本图像生成一系列工作历史

  - 2021 01 DALLE

  - 2021 12 GLIDE (图扩散模型做像生成)

  - **2022 04 DALLE2  : GLIDE + CLIP** 
    - Unconditional Design : 根据文本描述来生成**原创性的图片(fake image)**
    - In-painting : 根据文本对图片进行编辑和修改 (diffusion model并非discriminative，而是扩散生成模型，可以生成任意多不同细节的图片)
    - 直接输入图片，生成很多图片的变体
  - 2022 05 Imagen from Google



- 摘要 ：
  - 类似CLIP的对比学习可以很好的学习到图片的semantics and style 鲁棒表征，为了利用好这种特征来做生成任务，作者提出了一种两个阶段的模型：a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding.
  - 显示的建模图像的表征  有效的提高了图片的多样性，且并不会损失图片的写实程度 以及 图片与文字的匹配性 ---- 表示prior model生成image embedding 存在的必要性
  - 因为CLIP学习到了文本和图片的多模态embedding space，所以可以zero shot来用文本对图片进行操作 (Inpainting)



- 引言

  - CLIP模型通过 scaling models on large datases of captioned images 能够获得非常robust 图片表征，可以在很多个不同的领域获得zero-shot能力

  - 目前diffusion模型已经dominate生成模型领域, 一个著名的技巧是利用guidance technique 来引导生成模型牺牲一部分多样性，达到更好的保真度(sample fidelity)

    - DDPM
    - Improved DDPM : DALLE 二作、三作 受到启发
      - 高斯分布的方差是可以学习的
      - 添加噪声从线性schedule变为余弦schedule
      - diffusion模型很适合大模型
    - Diffusion Models Beat GANs  
      - 模型加大加宽
      - 使用classifier guidance 来引导生成，只用采样25step就能得到好的结果
    - GLIDE : 3.5 B
      - classifier free guidance
    - DALLE2
      - prior 
      - 层级式生成 : 64 + 256 + 1024

    

- 预备知识 : 

  - 引导模型生成

    - guided 
      - 训练时增加引导生成$||\epsilon - \hat{\epsilon}(x_t,t,y(x_t))||^2$
      - 推理时也增加引导推理$p_{\theta}(x_{{t-1}}|x_t) \approx \hat{\epsilon}(x_t,t,y(x_t))$，一般此类方法主要为了提升模型的质量而不是真的做引导，因为模型从训练开始就一直见过这个引导信号，所以可控生成方法比较弱
    - classifier guided 
      - 在训练模型时不变
      - 推理的时候对预测的噪音利用guided score进行引导 :  $\hat{\epsilon} = \epsilon_{\theta}(x_t) - \sqrt{1-\overline{\alpha_t}}\omega \nabla_{x_t}f(y|x_t)$, $\omega $ 为控制引导部分的权重

    > - 具体证明过程是考虑到score function和epsilon的关系 $ s_{\theta}(x_t,t)= -\frac{\hat{\epsilon}(x_t,t)}{\sqrt{1-\overline{\alpha_t}}}$ ，如果score变成了条件概率分布的score，则对应的高斯分布的噪音可以通过$-\sqrt{1-\overline{\alpha_t}} s_{\theta}(x_t,t | y)$ 来生成对应的概率分布的噪音
    >
    > - 缺点：虽然训练不受影响，但是需要一个外界的guidance，比如在**noised ImageNet**上训练一个预测加噪图片的分类器，还是不太方便

    - **classifier-free guidance ** : 训练模型时以一定的概率加入引导生成，推理时全部加入引导生成；这样通过一次diffusion模型训练就可以得到两种不同的概率分布

      - 网络在训练的时候同时见过两种输入：一种是有引导的$\hat{\epsilon}(x_t,t,y)$, 另一种是完全没有引导的 $\hat{\epsilon}(x_t,t,\phi)$, $\phi$ 表示在训练的时候**按照一定的比例drop**掉classifier 信息为空集 (不是mask是drop) 
      - 推理时：给网络两种不同的输入， 对结果进行组合$\overline{\epsilon}_{\theta}(x_t,t,y) = (1+\omega)\epsilon_{\theta}(x_t,t,y) - \omega \epsilon_\theta (x_t,t)$![截屏2023-03-25 21.06.54](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-03-25 21.06.54.png)
      - 缺点：虽然只用训练一个模型，但对训练难度比较高，希望模型具有两种不同的输出而不被confused，对于大公司而言无所谓
      - 注意：这个引导$y$ 可以有很多个向量张量组成

      





- DALLE2 模型

  - Note

    - CLIP多模态信息有时候是ground truth 有时候是 guidance
    - CLIP信息在Decoder中为输入(image embedding guidance), 在prior 中为ground truth标签(image embedding)和输入(text embedding guidance)

  - 训练数据：图片和文本对(x,y)，同时还用训练好的CLIP模型得到$z_i$和$z_t$ ,分别训练以下两个模型

  - 由于CLIP模型是deterministic model 

    - $P(z_i|y) = P(z_i|y,z_t)$
    - $P(x|y) = P(x,z_t|y)$

  - **Decoder ** : $P(x|z_i,y)$  根据CLIP Image embeddding和文字描述生成图片

    - diffusion的对象是原始图片$x$ , $z_i$和$y$表示 模型同时对CLIP的ground truth image embedding和image caption 进行classifier free guidance指导生成
    - 10% CLIP Image embedding为0，50% text caption 为空
    - 该diffusion用的网络结构是spatial cnn 没有用卷积层

  - **Prior **: $P(z_i|y) = P(z_i|y,z_t)$  根据CLIP text embedding和文字生成CLIP Image embedding

    - diffusion的对象是CLIP-ground turth image emdedding $z_i^{(0)}$, 即给一个embedding 向量加噪去噪, 目标函数为$\mathbb{E}_{t\sim [1,2..,T],z_{i}^{t}\sim q(z_i^t | z_i ^{(0)})}[|| f_\theta (z_i^{(t)},t,y)- z_i^{(0)}||^2]$

    - 此时的 classifier free guidance 包括text caption $y$ 和 CLIP-ground turth text emdedding $z_t$ , 在训练时分别以10%进行drop

    - 模型由于要预测一个1维序列，采样**transformer-decoder** 自回归的形式

      > 模型的输入有5部分：
      >
      > - decoder style input : **final embedding whose output from the Transformer** (自回归解码时上一次解码出的结果嵌入)
      > - guidance : **text**
      > - guidance : **CLIP text embedding**
      > - normal diffusion : **time embedding**
      > - normal diffusion : **noised CLIP image embedding** 
      >
      > 模型的输出：预测$z_i^{(0)}$ 即预测原始的样本而不是预测噪音



- DALLE2 灵活使用两个模块进行下游任务

  1. **论文里展示的根据文本$y$生成图片$x$ : two stage model**

  - ![image-20230325213517929](/Users/sirius/Library/Application Support/typora-user-images/image-20230325213517929.png)

  - prior : 给定一个文本$y$ , 以$y$和CLIP-ground truth text embedding为指导从高斯噪音扩散生成image embedding $z_i$

  - decoder : 给定生成的image embedding $z_i$, ， 以$z_i$和文本特征$y$ 为指导从高斯噪音进行扩散生成image $x$ 

    > decoder中可以仅仅以$z_i $为guidance，不需要$y$

  2. 给定图片$x$ 生成图片的变体

  - 通过CLIP image encoder 将$x$ 变为$z_i$ 

  - 以$z_i$ 为guidance 仅通过decoder对$x_T$  ($x$ 利用DDIM进行反向加噪) 进行扩散去噪

    > To do this, we apply the decoder to the bipartite representation ($z_i, x_T$ ) using DDIM with *η >* 0 for sampling. **With *η* = 0, the decoder becomes deterministic and will reconstruct the given image *x***. Larger values of *η* introduce stochasticity into successive sampling steps, resulting in variations that are perceptually “centered” around the original image *x*. As *η* increases, these variations tell us what information was captured in the CLIP image embedding (and thus is preserved across samples), and what was lost (and thus changes across the samples).

  3. 给定图片$x_1,x_2,\theta \in [0,1]$进行插值, 本质是对$x_{T_1}$ 和 $x_{T_2}$ 进行插值 得到$x_T$，对$z_{i_1}$和$z_{i_2}$ 进行插值得到$z_i$

  4. 图片和文本进行生成：输入文本对图片进行修饰 ( **Text Diffs**)

     > 有CLIP text ground truth embedding $z_t$ 和**prior**生成的embedding $z_0$ ,可以得到一个text drift $z_d = norm(z_t -z_0)$ ，接着通过CLIP image ground truth embedding $z_i$ 和$z_d$ 进行插值来进行修饰



- DALLE2 不足和局限性
  - 不能把物体和属性结合起来 : CLIP模型训练的时候只有相似性
  - 直接生成文字做的不好





## Ai4S



### FrameDiff

1. time embedding  = transformer 位置编码  + nn sequential projection

>  DDPM : Diffusion time *t* is specifified by adding the Transformer sinusoidal position embedding into each residual block



RF diffusion code by David Juergens

```python
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    # timesteps : torch.tensor([0,1,2,3...,T]) total T steps noise and 0 for motif no noise
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)

    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class Timestep_emb(nn.Module):

    def __init__(
            self,
            input_size,
            output_size,
            T,
            use_motif_timestep=True
    ):
        super(Timestep_emb, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.T = T

        # get source for timestep embeddings at all t AND zero (for the motif)
        self.source_embeddings = get_timestep_embedding(torch.arange(self.T + 1), self.input_size)
        self.source_embeddings.requires_grad = False

        # Layers to use for projection
        self.node_embedder = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False),
            nn.ReLU(),
            nn.Linear(output_size, output_size, bias=True),
            nn.LayerNorm(output_size),
        )

    def get_init_emb(self, t, L, motif_mask):
        """
        Calculates and stacks a timestep embedding to project

        Parameters:

            t (int, required): Current timestep

            L (int, required): Length of protein

            motif_mask (torch.tensor, required): Boolean mask where True denotes a fixed motif position
        """
        assert t > 0, 't should be 1-indexed and cant have t=0'

        # get the t step embedding
        t_emb = torch.clone(self.source_embeddings[t.squeeze()]).to(motif_mask.device)
        # get the zero steo embedding for motif region
        zero_emb = torch.clone(self.source_embeddings[0]).to(motif_mask.device)

        # timestep embedding for all residues * L
        timestep_embedding = torch.stack([t_emb] * L)

        # slice in motif zero timestep features
        timestep_embedding[motif_mask] = zero_emb

        return timestep_embedding

    def forward(self, L, t, motif_mask):
        """
        Constructs and projects a timestep embedding
        """
        emb_in = self.get_init_emb(t, L, motif_mask)
        emb_out = self.node_embedder(emb_in)
        return emb_out
```









### ESMFold code

MSA transformer 和 ESM-1v用来预测variant effects :

输入一个wild type + 一个mutated type ----- 输出

> - ESM-MSA 1b 使用UR50 +MSA数据训练了一个MSA transformer 模型，可以用来提取MSA embedding信息
>
> - ESM-1v ，与ESM-1b相同的模型架构但是用了UR90数据集 来预测序列突变对功能的影响

ESM-1v 利用masked language model的特点  : 

We score mutations using the log odds ratio at the mutated position, assuming an additive model when multiple mutations T exist in the same sequence:
$$
\sum\limits_{t\in T}\log p(x_{t}=x_t^{mt} | x_{\textbackslash T}) - \log p( x_{t} =x_{t}^{wt}|x_{\textbackslash T} )
$$
T是所有的突变位点, 如果是突变组合则利用log加和性质



1. recycle 

> - recycle the backbone atom coordinates from the structure module and output pair and first row MSA representations from the Evoformer. Af2 original use the pair $C_{\beta}$ distance into 15 bins of equal width from 1.25Å to approximate 20Å. Then project this one-hot distogram bin into the pair representation.
> - slightly  different with af2, this version is more simple
>
> 注意linspace num_bins - 1 , 一共有0-num_bins -1 这num_bins个区间
>
> ```python
> def distogram(coords, min_bin, max_bin, num_bins):
>    # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
>    # min_bin :  1.25
>    # max_bin : 21+3/8
>    # num_bins : 15
>     boundaries = torch.linspace(
>         min_bin,
>         max_bin,
>         num_bins - 1,
>         device=coords.device,
>     )
>     boundaries = boundaries**2
>     N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
>     # Infer CB coordinates.
>     b = CA - N
>     c = C - CA
>     a = b.cross(c, dim=-1)
>     CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
>     dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, 	keepdims=True) #[B,L,L,1]
>     bins = torch.sum(dists > boundaries, dim=-1)  # [..., L, L]
>     return bins
> 

2. add <eos> token and <bos> token

先给整体index加1个padding index ; 再添加开头和结尾两个token

esm中0代表bosi, 1代表padding index, 2代表eosi (第一个padding token为)

```python3
class ESMFold:
  	def __init__(self):
      pass
  
  @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + \
            [d.get_idx(v) for v in residue_constants.restypes_with_x]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0) #[B, L] token shifted right for padding token
        return self.af2_to_esm[aa]

    def _compute_language_model_representations(self, esmaa: torch.Tensor) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0) 
        # esmaa [B,L]

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi) #[B,1] full with bosi
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx) #[B,1] full with padding_idx as we may not know the exact length of each token
        esmaa = torch.cat([bos, esmaa, eos], dim=1) #[B,L+2]
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi #[B,L+2],设置第一个padding token为eos token (padding token都在句子末尾)
 
        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=False,
        )
        esm_s = torch.stack([v for _, v in sorted(
            res["representations"].items())], dim=2)
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        logits = res['logits']
        return esm_s, logits
```





### ProteinMPNN

#### Review : mpnn neural network only for node update

```python
class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(F.relu(self.W2(F.relu(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V
```





### MPNN

- feature

MPNN采样和Ingrahm一样的radia basis function : the fifirst vector is a *distance* encoding  r(·) lifted into a radial basis (We used 16 Gaussian RBFs isotropically spaced from 0 to 20 Angstroms)

```python
# from esm_if
def rbf(values, v_min=2.0, v_max=22.0, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device) #[n_bins]
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1]) #[1,1,1,n_bins]
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1) #[B,L,L,1]
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std #[B,L,L,n_bins]
    return torch.exp(-z ** 2)
```

$r$ enocide into $r_i = exp(\frac{r-\mu_i}{\sigma})^2$ 其中$\mu_i$ 为每个区间的均值，$\sigma$ 为一个固定的方差

- optimization

For optimization we used Adam with beta1 = 0.9, beta2 = 0.98, epsilon= 10−9, and the learning rate schedule described in (22). Models were trained using pytorch (27), batch size of 10k tokens, automatic mixed precision, and gradient checkpointing on a single NVIDIA A100 GPU. Training and validation losses (perplexities) as functions of optimizer steps are shown in Figure S3D. Validation loss converged after about 150k optimizer steps which is about 100 epochs of on-the-fly sampled training data from 23,358 PDB clusters.

- model 

```python
# The following gather functions
def gather_edges(edges, neighbor_idx):
  	"""
    给定一个batch数据的边特征,给每个节点取前K个特征
    这个边特征可以是mask信息,CA之间的距离,N原子之间的距离等等
    """
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    """
    给每一个batch里的node找到包括自身在内的其他node的特征,输出和gather_edge是一样的
    输出能狗找到每个节点邻居k个节点的特征
    """
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    """
    在某一个时刻t,找到一个batch里面的前k个节点
    """
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """
    因为gather_nodes和gather_edges的维度是一样的都是B,L,K,C,将边的信息和点的信息拼接起来
    应用于E_ij边的信息融合全部的V_J信息
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn



class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
				
        # num_in = 2C
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
				# mask_attend [B, N, K,]
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) # [B, N, K, 2C], gather the adjacnet edge E_ij and node V_j info
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1) # [B, N, K, C]
        h_EV = torch.cat([h_V_expand, h_EV], -1) # [B, N, K, 3C], gather the node V_i and the adjacent edge E_ij and node E_j info
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) # [B, N, K, C]
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


#注意encoder的edge已经发生了变换，不再是encoder中的edge，而是encoder的edge信息+sequence的edge信息
#E_ij = Concat[E_ij, S_j] * mask_ij + Concat[E_ij, 0.0*S_j] * (1-mask_ij)


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
				
        # num_in = 3C
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij (exlusive V_j)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1) # [B, L, K, C]
        h_EV = torch.cat([h_V_expand, h_E], -1) # [B, L, K, 2C]

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V 
      
class ProteinMpnn:
  def __init__(self)
    # Encoder layers
    self.encoder_layers = nn.ModuleList([
        EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
        for _ in range(num_encoder_layers)
    ])

    # Decoder layers,Decoder不改变边信息
    self.decoder_layers = nn.ModuleList([
        DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
        for _ in range(num_decoder_layers)
    ])
    .W_out = nn.Linear(hidden_dim, num_letters, bias=True)

    for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
  	pass
  
  def forward(self):
    #xxx
    for layer in self.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
    # Concatenate sequence embeddings for autoregressive decoder
    h_S = self.W_s(S)
    h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
    
		# Build encoder embeddings-constant
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)    
    
    # Decoder uses masked self-attention (slightly different with permuatation mask )
    mask_attend = self._autoregressive_mask(E_idx).unsqueeze(-1) #[B,L,K,1]
    mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1]) #[B, L, 1, 1]
    mask_bw = mask_1D * mask_attend #[B,L,K,1] --- 对于 j < i 的位点，可以利用序列信息S_j, 同时h_j的节点信息可以随着decoder而改变
    mask_fw = mask_1D * (1-mask_attend) #[B,L,K,1] --- 对于j>=i 的位点，序列信息S_j为0，且只能用encoder的h_j节点信息
    
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder
    for layer in self.decoder_layers:
       # Masked positions attend to encoder information, unmasked see. 
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx) # h_j和h_ES
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw # j<i的汇聚信息h_ESV会改变，但是j>=i的汇聚信息h_EXV总是不会改变
        h_V = layer(h_V, h_ESV, mask)

    logits = self.W_out(h_V)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs
```



$r_ij$具有causally consistent manner. Decoder的信息每次分为两部分输入

![image-20230322115332053](/Users/sirius/Library/Application Support/typora-user-images/image-20230322115332053.png)





### EGNN (Equivariance graph neural network)

- 摘要：提出了E(n)-Equivariant Graph Neural Networks(EGNNs)，不需要high-order reprenstation, 同时对于SE(n), reflection , permutation都是等边的

- 介绍：DL的成功很依赖inductive bias ， 比如CNN里面的translation equivariance, GNN里面permutation equivariance。许多依赖于3D旋转平移对称性的任务比如 point cloud、分子建模、N-body particle simulations 需要SE(3) 或 E(3)等变的网络，最近一些框架使用higher-order representations for intermediate network layers来达到这一目的，具有一些缺点 : 

  > - However, the transformations for these higher-order representations require coefficients or approximations that can be expensive to compute (spherical harmonics)
  > - Additionally, in practice for many types of data the inputs and outputs are restricted to scalar values (for instance temperature or energy, referred to as type-0 in literature) and 3d vectors (for instance velocity or momentum, referred to as type-1 in literature).

- EGNN :

  - **考虑一张图G=(V,E) , 注意到对于每一个node V_i有一个feature node embedding $h_i \in \R^{nf}$还有一个坐标向量$x_i \in \R^{n}$** Our model will preserve equivariance to rotations and translations on these set of coordinates xi and it will also preserve equivariance to permutations on the set of nodes V in the same fashion as GNNs.

  - 标准的GNN 

    ​	<img src="/Users/sirius/Library/Application Support/typora-user-images/image-20230322141929239.png" alt="image-20230322141929239" style="zoom: 25%;" />	

  - EGNN

  > 主要差别在汇聚信息(3)和更新node coordinate (4)，其中(3)将两个向量的相对平方误差考虑到边的汇聚中, (4)中$x_i$ 可以通过the weighted sum of all relative differences$(x_i-x_j)_{\forall j}$   来更新, $\phi_x : \R^{nf} \rightarrow \R^1$ , $C=\frac{1}{M-1}$
  >
  > PS : node embedding $h_i^0$天然需要E(n) invariant 表示信息，之后的$h_{i}^{l}$则一定也是invariant

  - <img src="/Users/sirius/Library/Application Support/typora-user-images/截屏2023-03-22 14.19.01.png" alt="截屏2023-03-22 14.19.01" style="zoom: 50%;" />

  >  ```python3
  >  class E_GCL(nn.Module):
  >      """
  >      E(n) Equivariant Convolutional Layer
  >      re
  >      """
  >  
  >      def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
  >          super(E_GCL, self).__init__()
  >          input_edge = input_nf * 2 #hi + hj
  >          self.residual = residual
  >          self.attention = attention
  >          self.normalize = normalize
  >          self.coords_agg = coords_agg
  >          self.tanh = tanh
  >          self.epsilon = 1e-8
  >          edge_coords_nf = 1 #dim ||xi -xj||_2^2 = 1
  >  
  >          # mij = phi_e : hi , hj, ||xi -xj||_2^2, a_ij
  >          self.edge_mlp = nn.Sequential(
  >              nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
  >              act_fn,
  >              nn.Linear(hidden_nf, hidden_nf),
  >              act_fn)
  >          
  >          # h_i = phi_h (hi,mi) 
  >          self.node_mlp = nn.Sequential(
  >              nn.Linear(hidden_nf + input_nf, hidden_nf),
  >              act_fn,
  >              nn.Linear(hidden_nf, output_nf))
  >          
  >          # phi_x (mij) to a scalar
  >          layer = nn.Linear(hidden_nf, 1, bias=False)
  >          torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
  >  
  >          coord_mlp = []
  >          coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
  >          coord_mlp.append(act_fn)
  >          coord_mlp.append(layer)
  >          if self.tanh:
  >              coord_mlp.append(nn.Tanh())
  >          self.coord_mlp = nn.Sequential(*coord_mlp)
  >  
  >          if self.attention:
  >              self.att_mlp = nn.Sequential(
  >                  nn.Linear(hidden_nf, 1),
  >                  nn.Sigmoid())
  >  
  >      def edge_model(self, source, target, radial, edge_attr):
  >          if edge_attr is None:  # Unused.
  >              out = torch.cat([source, target, radial], dim=1)
  >          else:
  >              out = torch.cat([source, target, radial, edge_attr], dim=1)
  >          out = self.edge_mlp(out)
  >          if self.attention:
  >              att_val = self.att_mlp(out)
  >              out = out * att_val
  >          return out
  >  
  >      def node_model(self, x, edge_index, edge_attr, node_attr):
  >          row, col = edge_index
  >          agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
  >          if node_attr is not None:
  >              agg = torch.cat([x, agg, node_attr], dim=1)
  >          else:
  >              agg = torch.cat([x, agg], dim=1)
  >          out = self.node_mlp(agg)
  >          if self.residual:
  >              out = x + out
  >          return out, agg
  >  
  >      def coord_model(self, coord, edge_index, coord_diff, edge_feat):
  >          row, col = edge_index
  >          trans = coord_diff * self.coord_mlp(edge_feat)
  >          if self.coords_agg == 'sum':
  >              agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
  >          elif self.coords_agg == 'mean':
  >              agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
  >          else:
  >              raise Exception('Wrong coords_agg parameter' % self.coords_agg)
  >          coord = coord + agg
  >          return coord
  >  
  >      def coord2radial(self, edge_index, coord):
  >          row, col = edge_index
  >          coord_diff = coord[row] - coord[col]
  >          radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
  >  
  >          if self.normalize:
  >              norm = torch.sqrt(radial).detach() + self.epsilon
  >              coord_diff = coord_diff / norm
  >  
  >          return radial, coord_diff
  >  
  >      def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
  >          row, col = edge_index
  >          radial, coord_diff = self.coord2radial(edge_index, coord)
  >  
  >          edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
  >          coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
  >          h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
  >  
  >          return h, coord, edge_attr
  >  ```
  >
  > 

  - include momentum

  <img src="/Users/sirius/Library/Application Support/typora-user-images/截屏2023-03-22 14.34.37.png" alt="截屏2023-03-22 14.34.37" style="zoom:33%;" />

  - 相关工作 : **Radiall Field 和EGNN接近, TFN加attention即变成SE3-transformer (spherical hamornics)**

![截屏2023-03-22 14.43.06](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-03-22 14.43.06.png)

Radial Field $r_{ij} = x_i- x_j$, 

TFN instead propagate the node embeddings $h$ but is uses spherical harmonics to compute its learnable weight kernel $W^{lk} : \R ^3 \rightarrow \R^{()(2l+1))(2k+1)}$





- **总结 : EGNN首先定义node embedding $h_i$和坐标$x_i$ ，通过radial field 的思想来保证更新过后$x_i'$ equivariant 同时$h_i'$ invariant, 注意$h_i$和$x_i$的信息本质是两个不同的任务 , 生成更关心$x_i$坐标本身的变化，而预测则关心$h_i$的信息**  



### GVP 

GVP 是在保证每个节点n个 3*1矢量特征等变$\R^{v \times3}$

EGNN是在保证每个节点只有1个矢量特征等变，但这个矢量特征可以是$\R^{n}$









### SE3-transformer

PS :  AlphaFold IPA 更适合做decoder的任务来搭配single representaion + pair representation + FAPE Loss来生成结构，因为本身IPA输入的frame不会包含刚体内部键长平面角的信息 





### Grpaph based protein structure prediction

- 计算二面角：通过四个原子来计算二面角

> 第一个氨基酸没有phi角，最后一个氨基酸没有 psi + omega

![image-20230324151400701](/Users/sirius/Library/Application Support/typora-user-images/image-20230324151400701.png)

```python
   def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) # 堆积每个蛋白所有的atom

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)

        # print(cosD.cpu().data.numpy().flatten())
        # print(omega.sum().cpu().data.numpy().flatten())

        # Bond angle calculation
        # A = torch.acos(-(u_1 * u_0).sum(-1))

        # DEBUG: Ramachandran plot
        # x = phi.cpu().data.numpy().flatten()
        # y = psi.cpu().data.numpy().flatten()
        # plt.scatter(x * 180 / np.pi, y * 180 / np.pi, s=1, marker='.')
        # plt.xlabel('phi')
        # plt.ylabel('psi')
        # plt.axis('square')
        # plt.grid()
        # plt.axis([-180,180,-180,180])
        # plt.show()

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features
```



## Python basic

### numpy indexing

numpy 使用了标准的python 索引语法  : x[obj] ; x是array, obj表示选择

x[(exp1,exp2,...,expN)] 等价于 x[epx1,exp2,exp3...,expN] 后者只是一种语法糖

(exp1, exp2, ... , expN) 被称为选择元组 selection tuple

#### Basic Indexing

- single element indexing

  - 0 based 索引
  - 接受负数索引
  - 如果对于一个多维数组的索引小于dimension, 则会返回一个子数组(subdimensional array)。返回的子数组是一个视图, 所以我们通常可以通过对子数组的指定来改变 原数组 e.g.  对于Transformer position encoding

  ```python
  pe[:,0::2] = torch.sin(position * div_term) # broadcast : [max_len, 1]* [d_model/2] -> [max_len, d_model/2] 
  pe[:,1::2] = torch.cos(position * div_term)
  ```

  - 对于一个多维度的数组，**没有必要**每个维度index都加一个括号比如 `x[0][2]`, 加一个括号就行了`x[0,2]`  后者是更有效的方式

  > Numpy 使用了C语言风格的按行索引 (Row-major )，或者说最后一个维度索引通常对应了变化最快的内存位置. 这与Fortran 和 MATLAB按列索引不相同 (Column-Major)

- Slicing and striding

将python对于array的切片扩展到了N维度上. 基本的索引语法 `start:stop:step`  对应 `i:j:k` , start 可以选到，stop无法选到

slicing 同样支持负数索引

如果`i` 没有给出，则为`:j:k` 表示i=0 if k > 0 or i= n-1 if k < 0 

如果`j` 没有给出，则为`i::k` 表示j = n if k >0 

如果`k` 没有给出默认为1

- Dimensional indexing tools

  - `:`表示某一个维度的所有值
  - `...` ellipsis 表示扩展selection tuple 到 x.ndim
  - `np.newaxis` `None` 主动在某个维度添加一个轴

  最明显的作用：

```python
mask_1D = np.array([1,1,1,0,0])
mask_2D = mask_1D[...,None] * mask_1D[...,None,:]
```



#### Advanced indexing

当selection object不再是一个tuple的时候，是一个ndarray (data type integar or bool) , 或者是一个tuple object包含ndarray. 有两种高级索引的方式: integar and Boolean



高级索引返回的是原有数据的一个拷贝而像basic slicing 一样返回一个视图.



> x[(1,2,3), ]将触发高级索引;
>
> x[(1,2,3)]表示basic indexing 等价于x[1,2,3]



- **Integer array indexing**

假设现在有一个array A 和 index array B  和假定最终的index结果array C

C的形状和B形状相同, C的每一个值都是 拿B的值去当索引 去A里面取



- Boolean array indexing

#### Field access













## 词汇

Monomer：单聚体 Oligomer：寡聚体 Polymer：多聚体

Protomer

homomer : 同聚物 protein complexes that are **formed by the assembly of multiple copies of a single type of polypeptide chain**. 

Heterimer ：异聚体  heteromeric protein complexes are  **formed from at least two different polypeptide subunits**, usually encoded by different genes 

globin : 球蛋白 hemoglobin：血红素  insulin  : 胰岛素

transmembrane protein 跨膜蛋白

Carbohydrate  n. 糖类

bilayer : 双分子层

antibody n.抗体

epitope n.抗原决定簇；表位

antigen n.抗原

viral adj.病毒性的

viral receptor



fluorescent protein 荧光蛋白

cell proliferation, survival, migration, and differentiation     细胞增殖、生存、迁移、分化

lyciferase荧光酶

nucleophile    electrophile 	亲核试剂、亲电试剂

protein threading  蛋白质穿针法 ,核心的想法是说对于非同源的序列，由于进化的保守性，也有可能存在相同的折叠模式







**radius of gyration, Rg 回旋半径**

> For a macromolecule composed of n mass elements, of masses $m_i$ , located at fixed distances $s_i$ from the centre of mass, the radius of gyration is the square-root of the mass average of  $s_i^2$ over all mass elements
>
> $s = \left(\sum\limits_{i=1}^nm_i s_i ^2/\sum\limits_{i=1}^nm_i\right)$

Rg用来反映蛋白质的紧密程度compactness，Rg越小表示蛋白质越紧，Rg越大表示越膨胀



**conformational ensemble 构象集合**

> **Conformational ensembles**, also known as *structural ensembles* are experimentally constrained computational models describing the structure of intrinsically disordered proteins.Such proteins are flexible in nature, lacking a stable tertiary structure, and therefore cannot be described with a single structural representation



allosteric 变构 蛋白激酶

eg. protein kinase 



equilibrium

eg. Nash equilibrium

 



PSSM (**position-specific weight(scoring) matrix** )  = PWM : position weight matrix-----保守信息，序列越保守越可能有功能



- **is a commonly used representation of motifs (patterns) in biological sequences  **

- PWMs are often derived from a set of aligned sequences that are thought to be functionally related and have become an important part of many software tools for computational motif discovery.

- Both PPMs and PWMs assume statistical independence between positions in the pattern

- 如何计算位置权重矩阵 PWM：----针对一个包含$N$个对齐的长度为$L$的序列集合,每个序列元素来自长度为$K$的一个字母表

  - 先构建PFM $\in \R^{K\times L}$,每个位置$M_{k,j} = \frac{1}{N} \sum\limits_{i=1}^NI(x_{i,j}=k)$,其中k代表每个字母表的元素,j代表每个位置；注意是每一列的概率和为$1$

  - 再将PFM通过log likelihood的方式转化为PSSM（PWM）：$M_{k,j}=\log_2\frac{M_{k,j}}{b_k}$,最后单位为bit

    $b_k$代表background model,如果$b_k=\frac{1}{k}$则可以用来表示该序列集和随机序列的关系，如果该序列集是随机生成的那基本上所有的entry都是0

- 矩阵的每一个实体可以称作 entry





设计出的蛋白质能够表达已经是一件很困难的事情了，如果从基因合成到质粒转导到表达纯化全部交给公司做比较贵，20条5w

1. 对于蛋白质结构预测而已，精确的2D distance map到 3D 结构有封闭解，但是生成2D distance map往往伴随着神经网络生成的系统噪音，因此从distance map到3D结构通常需要另一个步骤（另一个神经网络，能量函数梯度下降）

2. 蛋白质看成图网络的最大缺点是不够精细，没办法反应具体的3D结构，毕竟distance map是可能的3D结构的父集合；3D CNN倒是可以直接输入蛋白质3D结构，缺点就是太大了，每次batch只能=1，2，如果你用3D CNN把体素颗粒调大，虽然模型不大了，但又反应不了3D特征了；同时我们不希望输入网络的是一个blurry 结构信息，让网络对于同一个蛋白质结构输出更不一样的蛋白质序列信息

   PS：residue gas的添加就是对这个特点的一个缓解，

3. MPNN的方法是添加多个distance map的，将3D信息更好的利用2D distance map进行约束；添加多个distance map同时也相当于对于orentation和rotation的一种表示方法





Hydrophobic interaction 

非极性的物质在水溶液中通常会聚集到一起来降低和水分子的接触面积

> Note : Water is a polar covalent molecule as the non equal sharing of electrons.
>
> 1. Def : Hydrophobic interactions describe the relations between water and hydrophobes (low water-soluble molecules). Hydrophobes are **nonpolar molecules**.
>
> 2. Causes (Hydrophobic effect) : American chemist **Walter Kauzmann** discovered that nonpolar substances like fat molecules tend to clump up together rather than distributing itself in a water medium, because this allow the fat molecules to have minimal contact with water. 
>
> <img src="/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-20 上午11.14.58.png" alt="截屏2022-06-20 上午11.14.58" style="zoom:50%;" />
>
> The image above indicates that when the hydrophobes come together, they will have less contact with water. They interact with a total of 16 water molecules before they come together and only 10 atoms after they interact.
>
> 3. Strength of hydrophobic interactions : Hydrophobic interactions are relatively stronger than other weak intermolecular forces (i.e., Van der Waals interactions or Hydrogen bonds).
> 4. Hydrophobic Interactions are important for the folding of proteins. This is important in keeping a protein stable and biologically active, because it allow to the protein to decrease in surface are and reduce the undesirable interactions with water.
> 5. $\Delta G = \Delta H- T \Delta S$ , as the entropy is negative and hence hydrophobic interactions are spontaneous.



Van Der Waals Forces

> 1. heisenberg’s uncertainty principle : $\Delta x \Delta p \ge\frac{h}{4\pi}$
> 2. The Heisenberg’s Uncertainty Principle proposes that the energy of the electron is never zero; therefore, it is constantly moving around its orbital.
>
> These two important aspects of Quantum Mechanics strongly suggest that the electrons are constantly are moving in an atom, so dipoles are probable of occurring. A dipole is defined as molecules or atoms with equal and opposite electrical charges separated by a small distance.
>
> It is probable to find the electrons in this state:
>
> ![截屏2022-06-20 上午11.33.40](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-20 上午11.33.40.png)
>
> ![截屏2022-06-20 上午11.35.33](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-20 上午11.35.33.png)
>
> ![截屏2022-06-20 上午11.35.47](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-20 上午11.35.47.png)
>
> ![image-20220620114029303](/Users/sirius/Library/Application Support/typora-user-images/image-20220620114029303.png)



Hydrogen bond is a primarily electrostatic force of attraction between a hydrogen (H) atom which is covalently bound to a more electronegative donor atom or group, and anotherelectronegative atom bearing a lone parif of electrons-the hydrogen bond acceptor (Ac).  `Dn-H···Ac` , where the solid line denotes a polar covalent bond and the dotted or dashed line indicates the hydrogen bond.

![截屏2022-06-20 上午11.45.41](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-20 上午11.45.41.png)







**蛋白质三级结构**的形成**驱动力**通常是疏水残基的包埋，但其他相互作用，如氢键、离子键和二硫键等同样也可以稳定**三级结构**。 **蛋白质三级结构**包括所有的非共价相互作用（不包括二**级结构**），并定义了**蛋白质**的整体折叠，对于**蛋白质**功能来说是至关重要的。



两种在蛋白质折叠过程中非常重要的作用

- Hydrophobic effect

- Salt bridge

  > a combination of two non-covalent interactions : hydrogen bonding and ionic bonding.![截屏2022-06-20 下午12.00.08](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-20 下午12.00.08.png)



## MD

### The principles of Molecular Dynamics Simulation

- 每个原子当成小球，通过弹簧连接
- 只考虑宏观状态下的牛顿运动
  - 不考虑原子周围电子
  - 模拟过程中不会有化学键的断裂和形成
- 将时间划分为离散的time steps, 原子坐标依赖每一个equal length time step 而改变, 每一个时间步骤不超过 1 fs (1 femtosecond = $10^{-15}$ s)
- 能量一般可以分为
  - $E_{total}=E_{bonded}+{E_{non-bonded}}$
    - $E_{bonded} = U_{bond} + U_{angle} + U_{dihedral} + U_{improper}$
      - 非正常二面角应该专指$\omega$ 角
    - $E_{non-bonded}=U_{VDW}+U_{elec}$
  - 总能量(porential energy + kinetics energy) 保持不变
- 流程如下
  - 手写代码输入信息：atomic coordinates, nonds etc. of a system
    - coordinates .pdb or .xyz
    - 结构文件 .gro or .psf
    - 参数文件 .prm
  - 使用势能函数计算atomic energies 
  - 计算每个原子的力
    - 势能$U(x,y,z)$ 对3D坐标求导:$F(x,y,z) = -(\frac{\partial U}{\partial x}\hat{x} + \frac{\partial U}{\partial y}\hat{y}+\frac{\partial U}{\partial z}\hat{z}) = -\nabla U(x,y,z)$
  - 计算加速度 
    - $\frac{F}{m}=a$ 
    - 使用timestep (e.g. 1fs)和加速度来计算出速度和坐标

<img src="/Users/sirius/Library/Application Support/typora-user-images/image-20230222164202942.png" alt="image-20230222164202942" style="zoom: 25%;" />

- 主要的几个文件：
  - PDB file
  - parameters file
  - PSF file

<img src="/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-24 20.29.51.png" alt="截屏2023-02-24 20.29.51" style="zoom: 33%;" />



- Solvent 
  - vacuum (ignore the solvent)
  - implicit (mathematical model to approximate average effects of solvent, less accurate but faster)
  - Explicit  (high  computational expense but more accurate)

<img src="/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-24 20.37.38.png" alt="截屏2023-02-24 20.37.38" style="zoom:50%;" />

- Boundary conditions : 

  Boundary conditions describe the way in which your atoms will interact with the edges of the simulation box. This is important because it signifies the type of environment that the atoms are in.

  - solid wall: the atoms are repelled or deflected when they come in contact with the boundary.
  - periodic; **despite being more computationally intensive, this is the preferred and most common method.** If an atom goes into one boundary, it comes out on the other end -effectively, your system is composed of a repeating unit cell.
  - void ; atoms get deleted when yhey leave the cell. A seldom used method.
  - Non-cubic : sometimes used , can be implemented in NAMD. 

<img src="/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-24 20.37.52.png" alt="截屏2023-02-24 20.37.52" style="zoom:33%;" />



- What is kept constant?
  - **Microcanonical ensemble (NVE)**; a classical MD simulation ,where the amount of substance (N), volume (V) and energy (E) are conserved.
  - **Canonical ensemble (NVT)**. It is also sometimes called constant temperature molecular dynamics (CTMD). IN NVT, the energy of endothermic and exothermic processes is exchanged with a thermostat.
  - Isothermal-isobaric (NPT) ensemble; a type of MD simulation where N, P ,T are conserved. In addition to a thermostat, a barostat is needed.



- Minimization and Equilibration
  - **Equilibration** is a pre-simulation process where you equilibrate the kinetic and potential energies, usually in order to sort out any discrepancies that arose during the heating process (raising temperature of the system from 0kelvin to your set temperature)
  - **Minimization** is an important process that you carry out before your main MD simulation. It is designed to "relax" the system and distribute the energy equally, as weel as place atoms in lower energy positions and let them escape local minima.









势能 : 由参数确定

自由能 : 物理化学有个概念，一般说的是pMF





















## Protein

## amino acids

- Glycine 甘氨酸
- Alanine 丙氨酸
- Valine   缬氨酸
- Leucine 亮氨酸
- Isoleucine 异亮氨酸
- Serine 丝氨酸
- Threonine 苏氨酸
- Phenylalanine 苯丙氨酸  F
- Tyrosine 酪氨酸 Y
- Tryptophan 色氨酸 W (吲哚)

- Aspartate 天冬氨酸 D
- Glutamate 谷氨酸 E

- Asparagine 天冬酰胺 N
- Glutamine 谷氨酰胺 Q

- Cysteine 半胱氨酸 C

- Methionine 甲硫氨酸 M

- Lysine 赖氨酸 K
- Arginine 精氨酸 R
- Histidine 组氨酸 H
- Proline 脯氨酸 P



平衡态

1. Hydrophobic  

aliphatic : G,A,P,V,L,I,M 

Arom :   W,F 

![image-20220405114105881](/Users/sirius/Library/Application Support/typora-user-images/image-20220405114105881.png)

![image-20220405114145406](/Users/sirius/Library/Application Support/typora-user-images/image-20220405114145406.png)



2. Polar amino acid

- polar but non charged

  **hydroxyl group**(羟基) : S,T,Y ;   **carboxyamide**（酰胺） : N,Q.   巯基 Cys

  ![image-20220405114816060](/Users/sirius/Library/Application Support/typora-user-images/image-20220405114816060.png)



![image-20220405114924692](/Users/sirius/Library/Application Support/typora-user-images/image-20220405114924692.png)



- positively charged (base)

  L R H

  ![image-20220405115109912](/Users/sirius/Library/Application Support/typora-user-images/image-20220405115109912.png)

- Negatively charged  (acid)

  从Asn N ->Asp D 脱氢, 从 Gln Q-> Glu E 脱氢 

![image-20220405115339889](/Users/sirius/Library/Application Support/typora-user-images/image-20220405115339889.png)





![image-20220405115404309](/Users/sirius/Library/Application Support/typora-user-images/image-20220405115404309.png)



### structure

- Some basic concepts : 

二级结构：CO和NH的氢键是主要作用力  : alpha-helix beta-sheet turn loop

三级结构：长程的相互作用 long-range contact ；主要依靠疏水相互作用 

peptide bond是平面的





- [Difference between motif and domain](https://pediaa.com/what-is-the-difference-between-motif-and-domain-in-protein-structure/#Motif%20in%20Protein%20Structure)

Secondary structure is the first structure evolving 3D structure which formed to neutralize the the natural polarity of different amino acids in the primary protein structure. Typically, this neutralization occurs through the formation of **hydrogen bonds**. Further, these secondary structures combine with each other to form these motifs. The combining occurs through small **loops**. 

- **Motif** : super secondary structure ![截屏2022-02-28 下午2.31.54](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-02-28 下午2.31.54.png)
- **Domain** : tertiary structure 
  - Protein domain evolves, functions, and exists independently of the rest of the protein chain. The main type of bond formed is the disulfide bridge. They are the most stable interactions as well. Moreover, ionic bonds or salt bridges can also form between the positively and negatively charged amino acids in the secondary structures. Additionally, hydrogen bonds can form to stabilize the tertiary structure.![截屏2022-02-28 下午2.32.27](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-02-28 下午2.32.27.png)

## 序列比对

两序列比对：

- DP
- BLAST ： 启发式的算法

多序列比对：

通过一个reference sequence 

通过 HHblits 迭代的方式得到MSA，MSA的序列要根据reference sequence对齐

- Progressive model

  - Clustaw

  - Muscle

- Profile HMM
  - HHlibts
  - JackHHMer



### 蛋白质结构预测

- 基于模版
- 不基于模版
  - 能量打分函数
  - 共进化分析



1. 对于蛋白质主链而言，每个氨基酸只需要两个角度psi和phi就能确定构象，因此backbone的自由度为2n,如果主链有300个aa，backbone design的自由度为600

2. 共进化分析来预测距离矩阵

   - 全局统计思想来做**共进化分析**
     - 共进化分析主要的思想是 corrrelated mutation-----突变通常伴随着局部结构其他residue的联合突变,以维持蛋白质的整体结构.---1994年提出
     - 早期用的不多，主要是因为：
       - MSA中的序列不多
       - 系统发生树构建的偏差
       - indirect couplings mixed with direct couplings
     - 近十年来为什么用的多了？
       - 测序技术多发展，序列多了（模版仍然有限）
       - direct coupling analysis （DCA，本质可以理解为概率图中的马尔科夫场）的方法已经能够解耦合推断出因果关系或相关的问题
         - Message-passing DCA
         - Mean-field DCA
         - Psudo-likelihood-based optimization DCA----SOTA
   - ResNet预测 contact map
   - Transformer

3. ![截屏2022-03-19 下午10.48.57](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-03-19 下午10.48.57.png)

   从距离矩阵到接触矩阵的cutoff是8A

   大部分人一直在预测接触矩阵

   

4. 膜蛋白数据比较少，结构比较难解







### 蛋白质 residue-resiude contact prediction

问题：为什么评估不直接预测整个接触矩阵的正确率而要选择长程接触的precision

A : 统计的是长程接触的精确度 precision = $\frac{TP_C}{TP_c+FP_c}$，因为对于模型而言，长程接触更加难以判断



问题：TOP L/2,L/5 long-range是什么意思？

A : 预测出的contact可能很多，先根据long-range进行筛选满足长程接触的contact, 根据预测出来的置信度保留前5L，2L，L，L/2，L/5个contact算precision=（真正距离小于8A的个数）/ contact的总数. 明显保留的contact越少，置信度越高,precision越高



评估contact的组合：

- TOP L，TOP L/2，TOP L/5

- Long-range : at least seperate 24 residues

medium-range contacts are those separated by 12–23 residues

Short-range contacts are those separated by 6–11 residues in the sequence



对于许多蛋白质，只有 8% 的天然接触就足以重建蛋白质的折叠;此外，并非所有蛋白质的接触次数都与序列长度成正比。因此，通常使用精度评估前 L/2 或仅前 L/5 预测接触，其中 L 是蛋白质的序列长度;



对于short-range和medium-range的接触非常容易预测（尤其是对具有beta折叠的蛋白质而言），CASP的重点是评估 long-range contact .

top L/5的contact只包含contact的一部分，覆盖率比较低，需要根据具体的预测问题来选择所需要接触的数量

[Protein Residue Contacts and Prediction Methods](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4894841/)





### 如何评估两个序列的不相似性？

- 海明距离 ：

  **两个序列不相同位置的个数**

- 编辑距离

- 打分矩阵



PS：相似性, the identity of two sequences



### 膜蛋白

一般蛋白内部核心是hydrophobic ，而外部是亲水的

hydrophobic effect和salt bridge是蛋白质折叠的驱动力

但膜蛋白不仅核心是hydrophobic，外部因为跨膜的原因，也是疏水的，是lipid soluable

1. 膜蛋白的定义：“**Membrane proteins** are common [proteins](https://en.wikipedia.org/wiki/Protein) that are part of, or interact with, [biological membranes](https://en.wikipedia.org/wiki/Biological_membrane). Membrane proteins fall into several broad categories depending on their location ”



2. 膜蛋白的重要性：大约$\frac{1}{3}$的人类蛋白质蛋白是膜蛋白，是一半以上药物的靶点；但是实验确定膜蛋白的结构仍然是一个挑战，主要是因为难以建立能够让膜蛋白保持正确构象的实验条件，因此通过计算的手段如md来研究膜蛋白更为通用.



3. 膜蛋白大致分为以下几类：

- **integral membrane protein** : permanently anchored or part of the membrane
  - Transmembrane protein  :
    - Helix bundle proteins which are present in all types of biological membranes
    - Beta barrel proteins,  which are found only in [outer membranes](https://en.wikipedia.org/wiki/Bacterial_outer_membrane) of [Gram-negative bacteria](https://en.wikipedia.org/wiki/Gram-negative_bacteria), and outer membranes of [mitochondria](https://en.wikipedia.org/wiki/Mitochondria) and [chloroplasts](https://en.wikipedia.org/wiki/Chloroplasts)
  - Integral monotopic proteins : attached to only one side of the membrane and do not span the whole way across.
- **peripheral membrane protein** : only temporarily attached to the lipid bilayer or to other integral proteins
- **lipid-anchored proteins** : 



4. 膜蛋白的作用：
   - membrane receptors
   - ion channes
   - GPCR (G protein-coupled receptors) 
   - transport proteins



GCPR 膜蛋白







## 接触矩阵中进化信号的解偶 DCA

1. MSA的特性

   保守序列分析 PSSM

   最简单的序列设计 argmax(PSSM)---**consensus design**

   MSA之间的聚类 --- **系统发育树**

   

2. 氨基酸之间有相关性--共进化



我们可以从氨基酸学到什么

- 保守性
- 共进化
- 系统发育



现有模型介绍：

1. 马尔可夫随机场？？

   1+2预测3，1+3预测2，2+3预测1的简单神经网络----- MLPDCA等



2. APC (Average Product Correction)



![截屏2022-03-23 下午7.45.03](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-03-23 下午7.45.03.png)





## 基因测序相关知识

测序深度：总测序的碱基数除以基因组的长度 or 基因组中每个碱基被测序到的平均次数







## 比较蛋白质结构差异性的标准

**蛋白质全局相似度的表示**

1. RMSD : 具有单位,一般为Å

- Full chain C$\alpha$原子的RMSD



2. TM-score : (0,1]

- 全称：template modeling score
- 一般 TM-score < 0.2 代表两个蛋白质序列是随机选择的 ; TM-score > 0.5 代表两个蛋白质具有相同的折叠
- TM-score对**蛋白质序列的长度无关**，如果长度一样则直接进行从N端到C端的每个原子的计算，长度不一样需要先利用结构比对，找到commom residue进行计算

$$
TM-score = \max_i[\frac{1}{L_{largest}}\sum_i^{common}\frac{1}{1+(\frac{d_i}{d_0(largest)})}]
$$

**TMalign** : 一种基于蛋白质结构进行比对的方法，能够基于结构生成残基到残集的比对，与蛋白质序列是无关的，只和backbone有关，因此AF Design即可以用TM score进行固定主链的蛋白质设计！



3. GDT : Global Distance Test

- CASP的评价标准, 高于90分可以认为达到了实验验证的精度

  

  





**局部相似度的表示**

4. lDDT : local Distance Difference Test 

- RMSD是基于superpostion的一种表示,对于domain motion比较敏感,  lDDT 可以对每个残基计算出来一个分数,范围在0-100
- plDDT : 对每个残基预测的lDDT



## 序列比对的方式

AlphaFold 序列比对 : HHblits (基于profile hidden Markov models )，比PSI-BLAST快50%

Clustal Omega 

CD-hit

MMseqs2



## DataBase 

### 几种蛋白质的存储格式

序列：Fasta

结构：PDB; mmCIF

- CIF : Crystallographic Information File,developed for archiving small molecule crystallographic experiments 
- mmCIF : macromolecular Crystallographic Information File 大分子学晶体学文件

### CATH

PDB格式很有历史局限性，比如有些结构如两个beta-sheet之间的loop序列可能没有结晶出来，本身的序列就会有gap，



从PDB去除冗余切割得到





4 hierarchy  

- C : class : domains are assigned according to their secondary structure content

![image-20220908103355755](/Users/sirius/Library/Application Support/typora-user-images/image-20220908103355755.png)

- A : Architecture

  ![image-20220908103725598](/Users/sirius/Library/Application Support/typora-user-images/image-20220908103725598.png)

- T : Topology (Fold)

![image-20220908103818303](/Users/sirius/Library/Application Support/typora-user-images/image-20220908103818303.png)

- H : homologous superfamily

![image-20220908103839237](/Users/sirius/Library/Application Support/typora-user-images/image-20220908103839237.png)



### 序列数据库

**UniProt** :

- **UniProt KB** (Knowledgebase)

  - **Swiss-Prot**

    >  **Reviewed,Manually annotated, Records with information extracted from literature and curator-evaluated computational analysis.**

  - **TrEMBL**

    > **Unreviewed, Records that await full manual annotation.**

- UniRef (UniProt reference cluster)

  > The UniProt Reference Clusters (UniRef) provide clustered sets of sequences from the UniProt Knowledgebase (including isoforms) and selected UniParc records. This hides redundant sequences and obtains complete coverage of the sequence space at three resolutions:

  - UniRef100

    > **UniRef100** combines identical sequences and sub-fragments with 11 or more residues from any organism into a single UniRef entry.

  - UniRef90

    > **UniRef90** is built by clustering UniRef100 sequences such that each cluster is composed of sequences that have at least 90% sequence identity to, and 80% overlap with, the longest sequence (a.k.a. seed sequence).

  - UniRef50

    > **UniRef50** is built by clustering UniRef90 seed sequences that have at least 50% sequence identity to, and 80% overlap with, the longest sequence in the cluster.

- UniParc (UniProt Archive)

  > The UniProt Archive (UniParc) is a comprehensive and **non-redundant database** that contains most of the publicly available protein sequences in the world. Proteins may exist in different source databases and in multiple copies in the same database. UniParc avoided such redundancy by storing each unique sequence only once and giving it a stable and unique identifier (UPI) making it possible to identify the same protein from different source databases. A UPI is never removed, changed or reassigned.



注意到三个数据库都是非冗余的 non-redundant 

> - UniProtKB/TrEMBL: one record for 100% identical full-length sequences in one species
> - UniProtKB/Swiss-Prot: one record per gene in one species;





## AlphaFold架构

### 意义(施一公的评价)

- PDB 18w个非常重复的结构，比如溶菌酶有600个结构；不重复的蛋白结构不超过5w个，对于全长蛋白质不超过2,3W个结构

- AF出来之后，DB至少增加10倍，几乎预测了全部的人类蛋白，而且是全长的蛋白质
- 施一公说：AF对于只要有核心区域的蛋白质的预测绝对非常精准；对于flexible sequence和surface domain预测不准，但是对于这些蛋白质哪怕是NMR和冷冻电镜也是一种high temperature factor的动态结构， 也不是很准；**AF达到了实验科学的最高水平**。我们可以踩在巨人的肩膀上，做AF不能做的，比如基于AF结构预测来预测功能。最基本的功能包括丝氨酸、苏氨酸和酪氨酸的磷酸化
- 我自己的评价，AF的出现会比较好的解放生产力，让生物学家可以更关注于一些结构背后的功能等问题而不在关注如何获得蛋白质这个问题

### Architecture

如何得到一个在某个区间的连续型概率分布？ 

- 将该区间bin化，得到离散型概率分布



离散型概率分布 $H(X)=E[-\log p(x)]=\sum-p(x)\log p(x)\ge 0$

- 最大熵---均匀分布，得到最大熵为$\log \frac{1}{k}$
- 最小熵---一个为1,其余为0





### Structure Module

难点：

- IPA
- 如何构建全原子坐标：预测backbone Frame + dihedral(7个二面角)

1. Rigid bb representation，每个氨基酸是一个刚体(rigid body)

- $T=(R_i,t_i)$对每个真实位置的三角形做旋转平移操作，得到真实位置的三角形
- T全部指代空间变换



S的更新保证invariance, 之后的更新保证equivariance.



Invariant Point Attention:

主要包含3项，pair representation, **single representation**, frame  



























AlphaFold评估其预测结果的置信度

- pLDDT : 每个residue 预测出的lDDT,范围在0,100---- 用来评价每个位置的置信度，不需要label能直接得出，且AF中对应有confidence的自监督loss函数

- PAE (Predicted Aligned Error)  : 每个residue的误差，需要根据label来得出,$e_{(i,j)} = ||T_j^{-1} x_i  - (T_j^{True})^{-1}x_i^{True}|| $ 
  - for (x,y) aligned on y,  error in x

<img src="/Users/sirius/Library/Application Support/typora-user-images/image-20220420132749499.png" alt="image-20220420132749499" style="zoom:33%;" />

- 一些其他隐式的建模 : 核心是得到对PAE的预测之后$e_{ij}$ 再计算其他值

  - $z_{ij}$通过线性投影到64bins后, softmax得到最终每个区间的概率值；每个区间的64个bins求期望就可得到最终的$e_{ij}$

  - pTM建模

    1. 预测方法：![image-20220420160814933](/Users/sirius/Library/Application Support/typora-user-images/image-20220420160814933.png)

    2. 如果只对部分domaim求pTM

       ![截屏2022-04-20 下午4.08.58](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-04-20 下午4.08.58.png)

    3. 与FAPE不相同，FAPE考虑所有的align方式，pTM考虑每一行最好的align比对结果

  - 还可以对GDT,FAPE,RMSD进行预测



AF会输出3D结构，3D结构与真实的结果有一个比较的差值FAPE当作主损失函数；AF也会对这个FAPE做一个预测，真实的FAPE和预测的FAPE再得到一个差值，继续当作损失函数。第一步是输出预测的结构得到损失函数，第二步是输出预测的损失函数得到辅助损失函数。



总结AF中最常用的两个指标：

- PLDDT 来衡量局部的特征
- PAE 来衡量不同domain之间整体的比较



AlphaFold hallucination的问题在于：生成的序列往往是adversarial sequence，但是backbone通常是有用的



### Recycle

1. 把通过Evofmer的m1i直接加过去，zij要和最后输出的结果做一个信息的交融再加回去

<img src="/Users/sirius/Library/Application Support/typora-user-images/image-20220420185625209.png" alt="image-20220420185625209" style="zoom:33%;" />

2. 从0-N均匀采样N'，算N'次forward，最后第N'次forward后做backward



### Loss

1. FAPE loss

- 存在的工作--- 

  - superposition-based method （基于重叠）---- 整体的指标,不可导，但都可以找到chirality

    给定的两个蛋白质结构并不能直接计算坐标之间的差异，需要先寻找到一定的旋转平移使得两个结构align到一起，注意这些方法都是不可导的 ，**寻找旋转平移的过程中并不可导，只能用一些启发式的方法**，不能用到AF end-to-end

    - RMSD （数值可导，但寻找最合适旋转平移的过程不可导）
      
      - 序列严格一一对应可以有封闭解，直接找到旋转平移解
      - 序列只是同源没办法找到旋转平移
      
      $$
      S(\{\vec{x}_{i}\},\{\vec{y_i}\}) = \min_{T^{align}\in SE(3)}\frac{1}{N_{point}}||\vec{x}_{j}-T^{align} \circ \vec{y_j}\}||
      $$
      
    - TM-score : **基于结构比对**，完全不需要序列信息，甚至可以长度不等
    
    - GDT：
    
      为了避免RMSD导致的长距离不匹配信息带来的打分函数失真，GDT score对长距离的残基做一定的筛选。**计算距离小于阈值的$C_{\alpha}$原子的百分比**，CASP用1,2,4,8Å 四个值取平均后当作平均GDT.
    
  - Superposition-free method ---内部之间的距离,手性分子之间是没有差别的！
  
    - dRMSD ： 得到两个结构各自的distance map，再算两个distance map之间的距离 
    - lDDT
      - 规定不同residue的两个原子间距离小于15Å则称这两个原子有相互作用
      - AF中利用的是lDDT-$C_{\alpha}$，计算局部的一个相似性
        - 对每个residue，根据目标结构(reference)找到与其相互作用的$C_{\alpha}$原子
        - 计算**预测结构**中相互作用的$C_{\alpha}$原子距离和真实结构中距离的差值小于cut off 0.5Å,1Å,2Å,4Å的比例，取平均值得到该residue的平均lDDT
  
- DeepMind设计的方法 -----FAPE 

  PS：$T^i$是一种将local frame 向 global reference frame的旋转矩阵，因此转化为局部坐标系时应当用逆变换

  PPS：主FAPE是针对所有的rigid body frame，计算全原子的Loss;  cheaper version (scoring only all *Cα* atoms in all backbone frames) is used as an auxiliary loss in every layer of the Structure Module

  设计思路：

  - 本质还是根据uperposition: 找到**手性分子之间的差别** 且 **可导** 

  - 对于预测结构和真实结构：对于每一个原子j的绝对坐标$\vec{x}_{j},\vec{x}_{j}^{true}$，遍历每个frame坐标系$T_{i},T_{i}^{true}$进行坐标转换 : 得到同一个坐标系下的相对坐标 $\vec{x}_{ij},\vec{x}_{ij}^{true}$ 进行l2 norm 之后取一个平均即可得到Loss FAPE。注意到还有一个参数d clamp来控制只考虑距离小于d_clamp(10Å) 之间的距离！
  
    PS：FAPE和RMSD的区别是虽然都将原始的坐标转化为相对坐标，但是RMSD找到的是最好的那个旋转矩阵align后求l2 norm，而FAPE是直接考虑Nres个旋转矩阵，相当于考虑了N个 frame(N个相对坐标系下)N次align后的平均l2 norm
    $$
    \begin{aligned}
    FAPE(X,\tilde X) &= \frac{1}{N_{frame}N_{atoms}}\sum\limits_{i=1}^{N_{res}}\sum\limits_{j=1}^{N_{atoms}}||T_{i}^{-1}\circ \bold {\vec x_{j}} - \tilde T_{i}^{-1}\circ \bold {\vec{\tilde x_{j}}}||\\
    &=\frac{1}{N_{frame}}\sum\limits_{i=1}^{N_{res}}\left(\frac{1}{N_{atoms}}\sum\limits_{j=1}^{N_{atoms}}||T_{i}^{-1}\circ \bold {\vec x_{j}} - \tilde T_{i}^{-1}\circ \bold {\vec{\tilde x_{j}}}||\right)\\
    & \ge \frac{1}{N_{frames}}\sum_{i} S(\{\vec{x}_{i}\},\{\vec{y_i}\})
    \end{aligned}
    $$
    关于损失函数而言，保证unitless，而且只考虑旋转后距离不超过10Å的l2 norm ,
    $$
    Loss_{FAPE} = \frac{1}{Z}\frac{1}{N_{frame}N_{atoms}}\sum\limits_{i=1}^{N_{res}}\sum\limits_{j=1}^{N_{atoms}}\min\left(d_{clamp} ,\sqrt{||T_{i}^{-1}\circ \bold {\vec x_{j}} - \tilde T_{i}^{-1}\circ \bold {\vec{\tilde x_{j}}}||^2+\epsilon}\right)
    $$

  - FAPE能够找到手性：
  
    1. 对于手性分子$x_i$和$-x_i$，二者关于FAPE的loss不是0 ！
       $$
       FAPE(X,X_{relfection}) = \frac{1}{N_{frame}N_{atoms}}\sum\limits_{ij}||T_i^{-1}\circ\vec{x}_j-(T_i^{reflect})^{-1}\circ\vec{x}^{relfect}_j ||\\
       \because \vec{x}^{relfect}_j = -\vec{x}_j\\T_i^{reflect}=\left(\begin{pmatrix}
       -1 & 0 & 0\\
       0 & -1 & 0\\
       0 &0 &1
       
       
       \end{pmatrix} R_{i}^{-1}, -\bold{\vec{t}_i} \right)
       $$
       ![截屏2022-04-16 下午8.40.31](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-04-16 下午8.40.31.png)

    ![截屏2022-04-16 下午8.41.12](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-04-16 下午8.41.12.png)

    2. 如果不用能找到手性分子的FAPE loss,而是直接用dRMSD,预测出的结果是不包含手型的![截屏2022-04-16 下午8.42.14](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-04-16 下午8.42.14.png)

  - FAPE是一种合理的度量metric :
  
    ![image-20220416204410727](/Users/sirius/Library/Application Support/typora-user-images/image-20220416204410727.png)



2. Auxiliary loss:

   - average FAPE loss

     - $C_{\alpha}$的FAPE

   - torsionAngleLoss

     - 得到二维的扭角：

       ![截屏2022-04-16 下午9.32.25](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-04-16 下午9.32.25.png)

     - Loss 的形式：

       输入：预测的扭转角、真实的扭转角、alternative 真实的扭转角

       输出：Loss torsion表示预测的扭角和真实扭角的差值；Loss anglenorm表示希望神经网络输出的二维坐标更接近一个单位圆，防止梯度过小

       PS：    一些side chain 180-旋转角度-对称，即x和x+π有相同的物理结构，因此我们$\vec{\alpha}_i^{alt truth,f}=\vec{\alpha}_i^{truth,f}+\pi$。对于所有的非对称的情况，有$\vec{\alpha}_i^{alt truth,f}=\vec{\alpha}_i^{truth,f}$

       ![截屏2022-04-16 下午9.34.11](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-04-16 下午9.34.11.png)

       ![image-20220416214732661](/Users/sirius/Library/Application Support/typora-user-images/image-20220416214732661.png)

3. distogram loss

   distogram : NumPy array of shape [N_res, N_res, N_bins]   ---- 一般是64个bins, 每个点ij对应着residue i 和 residue j 距离之间的概率分布

   - AF只预测一个distogram, trRosetta等一系列的工作预测两两残疾对之间的 6D tensor信息！
   - Distogram loss 为概率分布与真实 one-hot binned  ground-truth data之间的概率分布

4. MSA mask loss $p_{ij}^b$

5. confidence loss

由于模型可以直接根据${s_i}$预测$p_{i}^{pLDDT}$的50binned分布，可以得到每一个residue的置信度得分$r_i^{pLDDT}$、整个序列的confidence、confidence的loss



### 蛋白质复合物方面的应用



## RF design

1. trRosetta - 2019年发布
   - 利用MSA提取出的保守性和共进化信息通过64层ResNet预测残基之间的4个几何特征，再将几何特征转换为约束的势能，在Rosetta最小化能量

2. trRosetta可以预测de-novo designed 序列，能否将其用于蛋白质设计？

   - 2020 年 trDesign-fixed backbone：GD进行序列更新

   - 2020 年 没有backbone直接做hallucination：MCMC进行序列更新

   - 2021年 3月 bake lab 结合二者，设计功能性(motif)的蛋白质：hallucination scaffold+ motif fix-bb design 

     > - 给轮子(motif)找一个车(scaffold)让他跑起来
     >
     > - 做法：将loss分为两部分：
     >
     >   Loss= Min CE + Max KL div
     >
     > - 难点：motif应该插入到scaffold的哪个区域？
     >   - 随机插入motif
   
3. 2021年 RF与AF诞生之后

   - 2021年RFDesign : 网络换成RoseTTAFold + Loss更新（Motif Loss + hallucination Loss + end-to-end coordinate loss ）
     - Motif Loss : 
     
     - Hallucination Loss：想让预测的结构、contact有比较高的可信度
       - AlphaFold 有 pLDDT可以评判结构的可信度
       
       - RoseTTAFold 最小化预测结果的熵（最大熵对应着均匀分布，熵最小对应着分布更加尖锐，contact更加可信）---这样能够生成更多的螺旋，蛋白质折叠的稳定性更高![截屏2022-04-17 下午1.34.12](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-04-17 下午1.34.12.png)
         $$
         \begin{aligned}
         L_{H} &= \sum_{i=1}^L\sum_{j\ne i}^{L}H(p(y_{ij}))\\
         H(p(y_{ij})) &= -\sum_{b=1}^{64}p(y_{ij}^b)\log (p(y_{ij}^b))
         \end{aligned}
         $$
         
       - RoseTTAFold 还可以利用trDesign的思想，即最大化与背景分布的KL散度
       
         ![image-20220417152048549](/Users/sirius/Library/Application Support/typora-user-images/image-20220417152048549.png)
       
     - Auxiliary Loss (end-to-end)：
     
       1. 对蛋白质幻想起到重要作用，提升计算效率
     
       - **motivation**：比如在设计scaffold的时候想要scaffold+motif和binder结合；如果在hallucination的时候不考虑binding信息，则生成的scaffold极有可能与binder产生clash，设计蛋白质的效率比较低
       - **Solution**：增加 repulsive loss （排斥损失函数），计算scaffold和binder的距离，利用一个范德华的能量函数来计算能量，最小化能量来使得二者远离
     
       2. 其他loss :
          - Attractive ----吸引势能
          - Radius of gyration ---利用回旋半径控制一个蛋白质更像一个球
          - RMSD：scaffold更像PDB里面的RMSD
          - 氨基酸组成的loss，表面上疏水氨基酸比较少
     
   - RFD如何确定motif-placement ：随机插入motif还是最可能的
   
   - 更新序列的时候到底用MCMC还是gradient descent ----Gradient Descent 500-600steps,再去接2000步的MCMC
   
   RFDesign 更适合做一些小蛋白binder的hallucination；分为两个阶段：第一阶段幻想是不带着受体的，但是有一些辅助loss帮助控制scaffold的朝向；第二个阶段可以将binder一起带入



## Conditional Modeling

条件语言模型用于表示在一定的条件c下一系列词语的概率分布，用chain rule分解为
$$
p(x|c) = \Pi_{i=1}^np(x_i|x_{<i},c)
$$
条件：单个序列、MSA、结构

1. 蛋白质：基于序列的条件模型

**ProGen**：没有运用任何的结构信息、MSA信息，完全利用单序列的蛋白质语言模型的预训练，生成自然界中不存在的蛋白质空间 2021 AI-----核心是Conditional-Transformer, 注释来自GO和NCBI

`Deep neural language modeling enables functional protein generation across families`



**ProteoGAN**：Conditional-GAN，大量的生带注释的数据，生成一些基于GO导向的数据 



2. 蛋白质：基于结构的模型

Graph based Transformer



3. 分子生成：基于SMILES编码，通过cRNN进行生成



生成模型如何评判最后的结果好还是不好？

算法端如何融入实验端 ？ 生成的100w条序列肯定没办法全部融入到实验室



## MD、QMMM

### MD、Force field、Poteintial energy

MD： **giving a view of the dynamic "evolution" of the system**.

区分：力场形式都一样，参数来自于QM或实验（不同力场不太一样），但并不等于MD

计算化学

1. 蛋白质对接  包括小分子和蛋白质，蛋白质和蛋白质等等

- 蛋白质对接的基本步骤：

  - 对三维空间进行sampling，生成一系列对接结构 decoys

  - 利用打分软件score functions对decoys结构打分，选取合理的对接decoys

    PS：Docking不预测蛋白质是否结合，只提供蛋白质相互作用的结构细节

  - 结合之后可以计算相关的自由能等

2. MD：

输入：原子细节的蛋白结构；Force Field；模拟时间；溶剂；PH；温度

输出：MD trajectory；某个时间节点的蛋白质结构

值得注意的是，MD在某个时间输出的结构RMSD是与自身的初始结构相比较的！



3. MD在蛋白质相互作用的分析：

docking之后，MD来改进对接结果，同时观察动态的相互作用

![image-20220511122518763](/Users/sirius/Library/Application Support/typora-user-images/image-20220511122518763.png)



5. 力场的基础知识

- 力场包含：

  - 势函数
  - 势函数的参数

- 原子类型：

  - 参数化与普适性

    ![截屏2022-05-27 下午7.31.59](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-05-27 下午7.31.59.png)

    



4. Amber力场的具体形式：

![截屏2022-05-15 上午11.33.54](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-05-15 上午11.33.54.png)

> - First term (summing over bonds): represents the energy between covalently bonded atoms. This harmonic (ideal spring) force is a good approximation near the equilibrium bond length, but becomes increasingly poor as atoms separate.
>- Second term (summing over angles): represents the energy due to the geometry of electron orbitals involved in covalent bonding.
> - Third term (summing over torsions): represents the energy for twisting a bond due to bond order
> - Fourth term (double summation over i{\displaystyle i}![i](https://wikimedia.org/api/rest_v1/media/math/render/svg/add78d8608ad86e54951b8c8bd6c8d8416533d20) and j{\displaystyle j}![j](https://wikimedia.org/api/rest_v1/media/math/render/svg/2f461e54f5c093e92a55547b9764291390f0b5d0)): represents the non-bonded energy between all atom pairs, which can be decomposed into [van der Waals](https://en.wikipedia.org/wiki/Van_der_Waals_force) (first term of summation) and [electrostatic](https://en.wikipedia.org/wiki/Electrostatics) (second term of summation) energie

PS：化学相互作用分类

- 基本化学键（原子和原子或离子与离子之间的相互作用）

  - 共价键：原子共享轨道之后，原子周围的电子活动区域变大，能量变低
  - 离子键：库仑作用力相互吸引

- 分子之间的相互作用力

  - 色散力

  - 取向力

  - 诱导力

- 其他介于范德华力和原子之间相互作用力的化学键：
  - 氢键
  - 化学配位键



参数怎么来的？

DFT



5. 蛋白模拟力场

- Amber :
  - ff14SB
  - ff19SB : 加入CMAP ($\phi,\psi$ 两个自由度耦合起来), OPC water (4 point water)
- Charmm
  - 2004 Charmm27 CAMP
  - 2012 Charmm 36 重新拟合二面角
  - 2017 Charmm36m 改进IDP 
- OPLS-AA

都经历过联合原子力场到全原子发展的过程，从1970-1980 



蛋白模拟力场---已经参数化了吗 ？

- 蛋白
- 磷脂膜
- 糖蛋白
- 糖磷脂
- DNA- RNA
- 小分子
- 水
- K+,Cl-,Na+



### QM MM (quantum mechanics/ molecular mechanics)

QM : 描述化学键

MM：描述每个原子之间的动态轨迹

2013 Nobel Prize in Chemistry  

- Prize motivation: “for the development of multiscale models for complex chemical systems”



## 旋转和四元数

1. 学习路径
   - 定义复数 y = a + bi的向量形式和矩阵形式 表示
   - 将2D平面的向量和 复数联系到一起
   - 表示3D空间的旋转 --Rodrigues' Rotation Formula
   - 定义四元数作为一个群
     - 四元数不具备交换律
     - 定义纯四元数和逆、共轭
     - 将3D旋转写为4元数形式 ----- 任何一个3D空间中的旋转都可以通过4元数来表示
     - 将4元数旋转写成矩阵的形式，将
     - 4元数和旋转矩阵的转换
   - 单位四元数作为 SU2 和旋转矩阵SO3是满射关系，同时是 2-1 满射同态
   - 四元数插值 
     - 两个单位四元数的夹角是对应旋转变化量的一半
     - 插值法 Leap、NLerp、Slerp 本质都是$\alpha_t q_{0}+ \beta_t q_{1}$ 且${\alpha_t + \beta_t = 1}$





## 讲座

### Chunfu Xu计算蛋白质设计 helix bundle

> 穿膜蛋白孔和荧光蛋白的设计
>
> $\alpha$螺旋的参数化设计--Coiled-coil equation--用方程来产生骨架



含孔蛋白的设计

- Ion channel
  - Gating mechanism
- Small molecule or meta binders
- Nanopores (leaky)
  - DNA sequencing
- Nanoparticle binders



![截屏2022-06-01 下午7.36.58](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-01 下午7.36.58.png)

![截屏2022-06-01 下午7.43.55](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-01 下午7.43.55.png)



### Scaffolding protein functional sites using deep learning

> Making functional proteins by scaffolding a motif
>
> 1. Motif identification
> 2. Designed scaffold (inpainting or hallucination)

- 目前只能从自然界中挑选某些具有功能的蛋白质，结合这些功能，并不能完全设计一个新的功能  ；我们做的是识别已有的motif并为之设计新的scaffold，原因是自然界的motif常常attach在不稳定的蛋白质上因此并不适合生物学或合成生物学的应用
- 应用：
  - **epitope presentation** for vaccine development
  - **viral receptot traps**
  - **Metall binding proteins and Active Sites**
  - **Protein-Protein Interactions**
- 方法
  - **Hallucination** : Iterative refinement
  - **Inpainting**：Inpainting Filling missing information

### Hallucination method

RoseTTAFold Design,利用两个氨基酸之间的六个值，Backbone Loss函数分解为两个指标

- motif region : cross entropy
- other region: minmize entropy 

![image-20220701131656212](/Users/sirius/Library/Application Support/typora-user-images/image-20220701131656212.png)

Filtering

E.g. AlphaFold is a quasi orthogonal metric.

- pLDDT > 80
- Motif RMSD < 1Å ( sub-angstrom accuracy)



### Inpatinting method

  

![截屏2022-06-28 下午3.54.29](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-06-28 下午3.54.29.png)

优势：更快；需要重新训练网络



### 数学物理原理在生物大数据中的应用



### 结构生物学讲座

1. 如何看蛋白质结构->X射线结晶

- 由于蛋白质的大小在纳米级别，我们需要分辨率更小的波：X射线0.1nm  
- 结晶 -> X-ray diffraction -> 傅里叶逆变换 -> 电子密度地图
- 1980年 X射线结晶标志着结构生物学进入了黄金时代



2003 诺贝尔化学奖 : Potassium channel 钾离子通道



2006 诺贝尔化学奖 ：RNA Polymerase II

2009 诺贝尔化学奖 ： Ribosome

2012 诺贝尔化学奖 GPCR



eg. Potassium ion; ionic; proton; iron; metal



X射线结晶的先天缺陷

- **结晶化**需要大量的蛋白质试错
  - 大分子量的蛋白难以结晶
  - 膜蛋白难以结晶
- 只能看到蛋白质的结晶状态



2. 有没有其他方法，能看到溶液条件下的搞分辨率蛋白质
   - 电子显微镜--虽然是溶液方向的，但是分辨率上不去
     - 样品损伤 （2013年之后分辨率革命）
3. 2017 年冷冻电镜 cryo-EM 横空出世
   - Single Particle Cryo-EM
   - Average : 3-4Å



4. Rational drug design





### 蛋白质边角预测

安芬森提出蛋白质构象是自由能状态最低的一个构象，且序列完全决定其构象。在过去人们 对蛋白质结构预测发现的第一个除了非模版方向的方法是从头蛋白质结构预测，假定一个能量函数然后优化。

- **能量函数不准**。原子之间的相互作用能量项理论应该由量子化学来确定，但是蛋白质原子太多还要考虑水溶剂的话现在的超级计算机也无法做到，目前的能量函数都是用经典力学下的相互作用来计算的。
- **采样空间太大，优化时间太慢**

长期以来对经典能量函数的改进：

- 只适用于非常小的蛋白质或多肽
- 也可能用一些统计的势能函数，比基于物理的能量函数稍微强一点。

在过去，从头预测不准，还是基于模版进行预测。但是大部分的蛋白质是没有模版的。



后来发展出基于结构碎片的方法，也算是一种de no，比如ROSETTA。从预测的二级结构到三级结构。



再后来发展为根据共进化氨基酸接触图来预测结构。



预测蛋白质的二级结构主要为预测三种形态，strand,helix,loop，主要的问题是粗粒化，没有理想的三种二级结构，且一旦分到了loop区域，这个区域的结构你是一无所知的。

因此DSSP直接定义了八种二级结构，但是问题并没有解决，同样是粗粒化分类，没有理想的八态结构，并且定义的类太多了之后精准度一定会下降。

目前二级结构都是预测三种类别。



预测蛋白质二级结构小于预测主链的二面角



预测主链的二面角一般都为分类问题(bin classfication)而不是回归问题

- 分类分的越多，预测越不准确
- 分类可以使角度落在非预测区域范围



回归问题由一些优点

- 不需要人为划分角度
- 似乎可以直接建立高精度主链结构



SPINE对真实主链二面角预测平均预测$\psi$角度误差=54度![截屏2022-08-02 下午9.49.27](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-08-02 下午9.49.27.png)



SPINE2 对$\psi$和$\phi$同时建模，希望能够提高精确度![截屏2022-08-02 下午9.50.23](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-08-02 下午9.54.26.png)





真实主链的二面角目前可以误差在23度以下，已经可以用来构建可靠的主链结构





### MPNN讲座

- 数据来源：
  - Single chaiin PDB sequence were clustered at 30% sequence identity
  - 3.5Å resolution cutoff
  - Up to 10,000 residue long complexes
  - **Check if the seuqnce similarity is above 70% over the residue pairs aligned using TM-align** 

- 参数：模型一共有1.7 million参数 ： 3 encoder + 3 decoder,hidden dim =128 (AlphaFold 100 million)

- 解码方式
  - 和顺序无关的自回归
  - 训练的时候采用和transformer一样的teacher forcing训练方法，给他正确答案来训练

- 特征提取
  - 25个距离，每个距离采用RBF编码---- 不同中心的RBF是一种不同维度normalized距离的方法，如果距离特别大，RBF就很接近0；如果距离特别小，RBF也为0，只有和；对于中心为$D_i$ 的$RBF = \exp[\frac{-(d-D_i)^2}{2\sigma ^2}]$![截屏2022-10-01 下午6.01.35](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-10-01 下午6.01.35.png)

- 训练时一个重要的点：

  - 训练时给结构加入高斯噪音
    - 因为PDB晶体结构是人为产生的
    - 具体应用时人们刚开始的backbone 可能并不完美
  - 推理的时候不需要噪音

- 温度的影响：MPNN即使T=0表示采取argmax进行采样而不是采取torch.multinomial，推理生成的序列仍然不一样，因为模型inference时会采取random decoding order（并不是噪音的原因）

- 生成序列的bias ：

  - MPNN本身倾向于给surface带负电的glutamate和带正电的lysine；MPNN并不喜欢生成polar aa 

  > ProteinMPNN generates more charged amino acids in expense of the polar ones at low temperatures which likely leads to highly thermo-stable proteins.
  >
  > - Rosetta本身同样也会有bias : Core区域的Alanine过多，表面Tryptophans过多

- MPNN对backbone很敏感 (resolution + pLDDT)

  ![截屏2022-10-01 下午8.20.08](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-10-01 下午8.20.08.png)

- 预测不确定性---在真实design的情况下怎么判断模型设计出序列的好坏

  > score = NLL
  >
  > MPNN在实验上给出了score和recovery的相关性是比较好的

- Self-consistency check : 设计出蛋白质用AF折叠一下来看一下设计的蛋白质和预测的蛋白质是不是一样的



对于soluble protein而言，core的revocery高不仅仅是疏水性，更应该是它的环境信息非常丰富；对于膜蛋白而言，整体recoery应该更高，但是如何利用膜的信息?



Review:

疏水氨基酸:

G, A ,V, P, L ,I ,M, W, F



### 刘海燕讲座

97年 Mayo 第一次提出根据能量来设计序列 (inverse folding) ; 2003年第一次提出RosettaDesign 骨架+序列

![截屏2022-11-25 09.50.55](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-11-25 09.50.55.png)

![image-20221125095846415](/Users/sirius/Library/Application Support/typora-user-images/image-20221125095846415.png)

SCUBAS的关键：如何设计一个和侧链无关或和侧链独立且具有可设计性的主链

- 可以从完全随机的initial backbone来生成novel topology
- 从某些已有的满足性质的拓扑结构出发，设计binding site等等



原来蛋白纯化不出来，现在用MBP fused  (麦芽糖结合蛋白)  就可以表达看结构

> Maltose-binding protein (MBP) is one of the most popular fusion partners being used for producing recombinant proteins in bacterial cells. MBP allows one to use a simple capture affinity step on amylose-agarose columns, resulting in a protein that is often 70-90% pure. In addition to protein-isolation applications, MBP provides a high degree of translation and facilitates the proper folding and solubility of the target protein. This chapter describes efficient procedures for isolating highly purified MBP-target proteins. Special attention is given to considerations for downstream applications such as structural determination studies, protein activity assays, and assessing the chemical characteristics of the target protein.



### 卜东波老师

- ProFOLD

  ![截屏2022-11-25 10.29.54](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-11-25 10.29.54.png)

  ![image-20221125103454159](/Users/sirius/Library/Application Support/typora-user-images/image-20221125103454159.png)

Rosetta 用线性加权来平衡不同的能量函数，每个人加一项所以最后Rosetta就很复杂

trRosetta 用学习到的Distance, oriteation 来进行 能量函数来约束

ProFOLD : 本身还是AlphaFold1的非end-to-end的结果

- ProDESIGN
- ProAffinity



### 马剑竹 - - OmegaFold 

> 博士导师Jinbo Xu
>
> 清华大学电子系副教授-

Anfisen的原则到底是不是普适的规则？

- 强烈认可Anfisen的原则是OmegaFold的背景

- RaptorX 曾经有很多人在用，现在没了；有了基因组之后，利用Web Server来预测结构，发了相关的paper

![截屏2022-11-25 11.16.48](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-11-25 11.16.48.png)



- AlphaFold在structure module中对每个residue更新一个backbone frame tuple  $T_{i} \coloneqq (R_i,\vec {t_i})$ 

  - 这个tuple表示一个**从局部坐标系(local frame)到全局坐标系(global refrence frame)的欧式变换**，也就是将我们预测出来的局部坐标系下的三维坐标$\vec {x}_{local} \in \R ^ 3$ 变换到全局的坐标 $\vec {x}_{global} \in \R ^{3}$ , 公式如下:
    $$
    \begin{aligned}
    \vec {x} _{global} &= T_i \circ \vec {x}_{local} \\
    &= R_i \vec {x}_{local} + \vec t_i
    \end{aligned}
    $$

    > 预测出所有的三维坐标，一定要有 frame来将其转换到整体的坐标系下
    >
    > - 注意到对于backbone frame而言，其实空间变换等价于坐标；文章中全部指代为initilizae 之后的空间变换
    > - 但似乎torsion angle仍然需要侧链rigid body frame 变换后才能确定坐标

  - IPA module的内容 ：

  - FAPE Loss : 对于所有backbone frame和side chain frame，分别align到这个这个局部坐标系下，进行RMSD的计算

  

- TPU通信代价非常小
- GPU结点的通信代价非常大
- ![截屏2022-11-25 11.23.11](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-11-25 11.23.11.png)
- OmegaFold提取的也是公进化信息，只不过是从具体的系列变成弱一点抽象的信息
- mask氨基酸的时候采取策略
  - mask就是一段，防止model看见相邻的氨基酸偷懒，比如看见S就预测T



- 训练模型

  - **Truncated to fit GPU memory** 

    - 直接一截一截的截断   -----长程约束捕捉不对

    - 3D空间上截断truncate a protein in the 3D space -----序列上比较碎，简单的近程约束做不对

    - Solution :

      - 当成一个优化问题：让这些氨基酸空间离的比较近，而且也不会序列上太碎

      ![截屏2022-11-25 11.38.09](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-11-25 11.38.09.png)

  - **希望模型focus on hard targets (hard patterns) , 计算机很难判断这到底是噪音还是一个困难的pattern，利用AlphaFold plddt来指导**

    - ![截屏2022-11-25 11.39.52](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-11-25 11.39.52.png)
    - **AlphaFold2 和 MSA的数量有极大的bias，造一大堆假的MSA plddt也会很高**
    - **和ESMFold的区别**
      - ESMFold 在12M的AlphaFold的自蒸馏数据集，难免会带来AlphaFold DB的bias

  - OmegaFold抗体预测

    - AlphaFold2 对 CDR of antibodies的建模不太好是因为没有共进化信息，这也是所有复合物的AlphaFold2预测的难点

    - 只预测native 的单独抗体，且和抗原结合后形状不会发生改变的那些抗体

      ![image-20221125114608648](/Users/sirius/Library/Application Support/typora-user-images/image-20221125114608648.png)

    - OmegaFold不太需要recycle

  ![image-20221125120423711](/Users/sirius/Library/Application Support/typora-user-images/image-20221125120423711.png)

1. TMalign有封闭的transformation解吗？ --- 无
2. RMSD 似乎有封闭的transformation  ?   ---- 有 (kabsch RMSD)
3. AlphaFold 预测坐标系的问题



补充阅读  TM align 原文

1. 最初是先有了TM-score （一种打分方式），主要是为了对标RMSD scores all the atoms equally的问题，align仍然是基于prior equivalency利用iterative的形式没有什么创新点；之后2005年利用TMscore推出了TMalign ，不基于residue quivalency的align方法；在web中，直接算TMscore和TMalign的TMscore是不一样的，因为align的方式不一样

   > - 

2. 有两种做protein comparison的方法 

   - The first is to compare protein structures/ models **with an a priori specified equivalence between pairs of residues** (such an equivalence can be provided by sequence or threading algorithms, for example). 

     - Kabsch RMSD 
     - Protein threading
     - TM-score

     > Kabsch有最佳比对的RMSD alignment封闭解；单纯算TM-score仍然用kabsch alignment做为初始化，不断迭代优化
     >
     > RMSD不是一个非常好的评判标准 : Since the RMSD weights the distances between all residue pairs equally, a small number of local structural deviations could result in a high RMSD, even when the global topologies of the compared structures are similar. Furthermore, the average RMSD of randomly related proteins depends on the length of compared structures, which renders the absolute magnitude of RMSD meaningless

   - The second type of structure comparison compares a pair of structures where the alignment between equivalent residues is not a priori given. Therefore, **an optimal alignment needs to be identified**, which is in principle an NP-hard problem with no exact solution

     - TMalign就是这种不给定prior equivalency，通过三种初始化方法（二级结构DP，threading 初始化等）利用TM-score进行启发式迭代

       

### John Jumper   Kendrew Lecture 2021

condensed matter theory 凝聚态理论

I have to **confess** i spent some ti me doing the same thing  and eventually de parted in the same way



Uniprot grew three thousand times faster than  pdb in terms of number depositions.

Genomics revolution had given us really sequence abundance, and alphafold is a tool to turn  that into structural abundance. 



AlphaFold one was really an off-the-shelf system from computer vision. We want to build the protein structural knowledge around the deep learning system.



Each pdb file represents a phd's worth of effort.



Add the physical and biology knowledge into the network

- Vignette 1 : Triangular Attention
  - Take 3 points A,B,C
    - If Distance AB and distance BC known,strong constraint on AC (triangle inequality)
  - Pair Embedding encodes relations
    - Update for pair AC should depend on BC, AB
    - All about who communicates in the network not what is computed

![截屏2022-12-16 23.10.28](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-12-16 23.10.28.png)

- Vignette2 : Structure module

  ![截屏2022-12-16 23.12.19](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-12-16 23.12.19.png)

In this sense, we are trading proper biophysics for computational expedience. 





- AlphaFold can understand biological context



- Impact

  - AlphaFold increase in coverage of the human proteome, such as the membrane proteins.

    ![截屏2022-12-16 23.27.38](/Users/sirius/Library/Application Support/typora-user-images/截屏2022-12-16 23.27.38.png)



- Disorder region --- AlphaFold is a good predictor 





Q : How AlphaFold overcome the limitations of basing evolutionary and sequencing data ?

A : bert mask loss



AlphaFold 有三部分输入

- primary sequence
- msa
- template

**大部分蛋白：msa信息很充分，预测很好**

**一小部分蛋白没有msa，但是有40-50 identity的template，template信息很充分**

**design protein, primary sequence is strong of encoding structure.**



Batch_size太大

- 优点
  - 收敛速度比较快，例如batch如果是整个数据集的话，能够忽略数据集中的大部分噪音，整个数据集优化更容易优化到一个**极小值**
  - 速度比较快，利用了GPU的加速

- 缺点
  - 显存吃不消
  - 虽然更容易收敛到一个极小值，但是收敛到的这个值可能和最小值之间偏离很远，同时因为忽略了噪音，更容易过拟合 ---- **收敛的结果容易有问题**

Batch_size太小

- 缺点：
  - 极端情况Batch_Size = 1，**在线学习**Online Learning， 每次修正方向以各自样本的[梯度方向](https://www.zhihu.com/search?q=梯度方向&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A71137399})修正，横冲直撞各自为政，**难以达到收敛**。





### CASP15-郑伟-张阳组

UM-TBM server

一个问题，如何评价得到的MSA？

- Ranking MSAs by the predicted plddt (用AF跑每个MSA序列得到一个plddt打分)

联想ESM-IF1对1200w个蛋白质序列的筛选，用MSA transformer来自动判断 AlphaFold2对这些序列是否能折叠的比较好，每一个序列都有一个打分





### 基于集合变量到增强采样算法

为什么做增强采样？---某些统计量趋于稳定是否代表着模拟时间足够反应是否平衡了？

常见的生命活动时间尺度

- 水分子在溶液中的氢键形成和解离 $1ps = 10^{-12} s$
- 蛋白和配体结合 $1 nm = 10^{-9}$s
- 小肽的折叠 $1 \mu s=10^{-6}s$ 

![截屏2023-02-20 14.05.58](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-20 14.05.58.png)







增强采样算法

- 不依赖集合变量的增强采样算法 

  - Accelerated MD (AMD) : 修改势能面，降低反应能垒

    PS : 只加速了由能量驱动的采样过程

  - Replica-exchange MD (REMD)

    PS : 模拟不可控，操作比较简单，只需要修改config配置文件

![截屏2023-02-20 14.11.35](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-20 14.11.35.png)

![截屏2023-02-20 14.13.11](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-20 14.13.11.png)

- 依赖集合变量的增强采样

  - 定义

    - 反映坐标：描述感兴趣过程中的变量，一般未知且维度很高
    - 反映坐标模型：模拟中“假定”的反应坐标，由一个或多个集合变量构成
    - 集合变量
    - 自由能面：沿着反应坐标模型的自由能变化

  - 增强采样方法 :

    原理：沿着反应坐标方向施加偏执力，加速沿某方向的运动

    - Umbrella Sampling (US) 伞状采样
    - Metadynamics (MtD)
    - Adaptive Biasing Force (ABF)

![截屏2023-02-20 14.22.46](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-20 14.22.46.png)





- 正交空间与分窗口策略
  - 正交空间：与感兴趣的过程相关，却未被包括在反应坐标模型中的自由度



当反应坐标模型构建不是很合适时，会出现能垒偏高，缺少本应有的极小值等现象



MD希望得到一个完整的系综势能面，关注正则系综的平均自由能，而不是说仅仅寻找一个最低能量





### RF diffusion

Q

0. diffusion - basic , RF diffusion is not DDPM while diffusion ?
   1. CA DDPM

   2. Rotation needs IGSO3 or Diffusion in Riemann manifold

1. self-conditioning
   1. $\hat{x_0}^t$ 预测不仅仅依赖于$x^{t+1}$还依赖于$\hat{x_0}^{t+1}$ , 预测的是

2. SO3-noise  brown motion 
3. adjacency block 和 二级结构是怎么定义的？如果和Anand一样不是residue level的话怎么输入给网络？
4. training details for different models (binder, motif-scaffolding, symmetric)
   1. 用guidence 加一个score




----



Why  : Nature has explored only a tiny subset of the possible protein landscape. Evolution does not necessarily select for protein attributes that are desirable from a pharmaceutical/biotechnological perspective (in virto solubility, stability, ease of production, low immunogenicity etc). De novo protien design allows us to derive new proteins with new functions and desirable attributes.



Workflow:

<img src="/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-26 15.05.04.png" alt="截屏2023-02-26 15.05.04" style="zoom:50%;" />



Advances:

with the advent of ML and its application to protein design, there's really been sort of enormous advance in the latter three parts.  We have really extensive evidence now showing if you can get really good recapitulation by for instance AlhpaFold, then your chance of experimental success on a whole range of different problems is really high.

![截屏2023-02-26 15.11.47](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-26 15.11.47.png)



Diffusion models as an attractive framework for protein design

- can generate endlessly diverse outputs (Broadly applicable).
- can operate directly on amino acid coordinates.
- can condition on a wide range of inputs, and can be guided with auxiliary potentials.



How can we learn on protein structures?

- Challenges of Proteins vs Images
  - Strong geometric constraints (4 backbone heavy atoms, 3covalent bonds, continuous chain)
  - Must also have a sequence that can encode it. (Not all backbones are encodable by an amino acid sequence)
- Frame-based representation
  - It takes advantage of the fact that the geometry of the NCA bond and CAC bond in protein backbones is highly constrained and fixed
  - Noise on translation and rotation 
    - noise CA coordinates with 3D gaussian noise
    - noise rotations with Brownian motion on SO(3)



![截屏2023-02-26 15.32.23](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-26 15.32.23.png)



![截屏2023-02-26 15.34.25](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-26 15.34.25.png)



diffusion attractive 

- denoising 过程每次预测$\hat{x_0}$ ，充分利用RF 的inductive bias (pre-training 很重要)



Training Summary



- Dataset : PDB (< 384 aa; i.e. no cropping), clustered by sequence similarity
- 200 timesteps (t=0: true structures, t= 200 : 3D Gaussian Ca coordinates, uniform frame rotations)
- 8x Nvidia A100s, ~4 days 
  - RosettaFold has 80 million parameters 
- **Self-conditioning** of predicted X0 structures between timesteps
- **Fine-tunning** from RoseTAFold structure prediction weights
  - 利用RosettaFold Inductive bias来fine-tuning RF diffusion



如果不用self-conditioing, 生成的结构 are not well packed and they are not particularly  diverse





Experiment

- Unconditional (与hallucination benchmark )

  - 通过self-consistent RMSD 来评价不同长度的蛋白质设计
  - TM-score 来评价设计的diversity
  - RF diffusion captures the ideality of de novo designed proteins (desirable de novo protein propertis)
    - expressing well
    - being vert thermostable







- RF diffuison can generate specific protein folds

  - Protein folds can be coarsely described by secondary structure (residue level) & "block adjacency" (residue level)
    - Train a new model to condition on these 2 things
    - 二级结构由DSSP训练得到 [L, 4] 表示 (helix, sheet ,loop, mask)
    - block adjacency  [L, L] one-hot matrix  注意和Anand 的block并不一样，RF diffusion里面的block就是氨基酸level ，$m_{ij} = 1$ 仅当：
      - i,j 氨基酸的二级结构都不是loop
      - i,j 氨基酸CA原子距离小于8Å
  - 根据二级结构信息来采样一整个protein family
    - e.g. TIM barrel family : sample all different kinds of conformations from the backbone coarse-grained information
    - e.g. NTF2 Folds

- Symmetric oligomers 

  - RF diffusion 把每次的$x^t$ 对称化
  - dihedral symmetric 成功率显著的高于RF hallucination

- Functional motif scaffolding  

  - benchmarking success translates to experimental success

  - E.g. p53 helix scaffolding problem

    - BG

      > p53是一种蛋白质，它在人类体内起着重要的抑制肿瘤基因的作用。它的名称是因为它含有一个由与DNA结合的螺旋结构（helix），被称为p53 helix。这个蛋白质可以识别和结合DNA，帮助维持细胞的正常生长和分裂，并在DNA受损时促进细胞进入修复状态或引发细胞凋亡。
      >
      > hMdM蛋白是p53的一个调节因子，它可以与p53结合并抑制其功能，从而影响细胞的正常生长和分裂。hMdM蛋白是一种九个外周螺旋（HECT）类的泛素连接酶，可以促进泛素化修饰，并促使p53的降解。
      >
      > 癌症通常与p53蛋白或hMdM蛋白的异常有关。例如，某些癌症细胞可能会发生p53基因的突变或缺失，导致p53蛋白的功能异常，从而阻碍细胞的正常生长和分裂，甚至导致细胞癌变。而hMdM蛋白则可以抑制p53的功能，从而促进癌细胞的增殖和转移。因此，p53 helix蛋白和hMdM蛋白都是癌症研究中的重要领域

    - Motivation : 如何可以阻止p53 Helix与Mdm2 的结合，就可以提供新的癌症疗法；希望设计一个新的蛋白和p53 结合，阻止其与Mdm2结合 (600 nanomolar affinity)

- Symmetric metal-binding oligomers (symmetric motif scaffolding)

- De novo binder design (**Multi-chain motif scaffolding problem**)

  - target is a protein
    - Motif is a target protein that you are trying to bind (the own chain); **we need to design a second chain globular that packs well against this protein**
  - target is a peptide





### Learning to Generate Data by Estimating Gradients of the Data Distribution

score matching -> 等价目标函数 -> denoising score matching or slicing score matching -> langevin dynamics -> Noise Control Score-based Model (NCSM) -> Infinity noise schedule to SDE -> Diffusion

核心：using score functions to represent probability distributions



Stein score function $\nabla _{x}\log p(x)$ also called scores for gravity



Objective : $\frac{1}{2} \mathbb{E} _{p_{data}(x)} [||\nabla_{x}\log p_{data}(x)-s_{\theta}(x) ||_2^2]$ which is called fisher divergence  外面的P data 可以通过蒙特卡洛模拟解决，里面的没有办法，只能继续推导，利用一些general的假设得到优化目标不包含ground truth distribution的解析解

- we can use an old method called score matching (integration by parts) to convert the fisher divergence into the following equivalent objective. IN this equivalent objective there is no dependency on the data score function

Score Matching
$$
\mathbb{E}_{p_{data}(x)} [\frac{1}{2}||s_{\theta}(x)||_2^2 + tr(\nabla _x s_{\theta}(x))] \\
\approx \frac{1}{N}\sum_{i=1}^N [\frac{1}{2}||s_{\theta}(x_i) ||_2^2+ tr (\nabla _x s_{\theta}(x_i))]
$$

- computational challenge of score matching can boil down to the jacobian matrix.



Langevin dynamics 每一步都在不同的noise sacle上进行采样，T越大，越随机；如果直接采样的话，low-density region score function并不准确

- 直接用score matching + Langevin dynamics 效果很差；主要是因为每个样本都是high probability region，周围low probability region 的 scrore function并不准确，所以initial structure后并不能采样到有效的样本
- 给data加入不同scaling的noise，这样整体的数据分布逐渐变得平缓，score function在输入不同的噪音的条件下都能进行langevin dynamics采样得到合理概率分布的样本



总结 Score-based generative modeling 

- Flexible models ----- 因为输出的东西不用归一化为1个概率，所以模型可以非常复杂
  - Bypass the normalizing constant  (引入score function)
  - Principled statistical method (最优化score function 等价优化一个相同的目标函数)

- Improved generation

  - Higher sample quality than GANS (绕过了flow model的模型限制，可以给很复杂的模型)

  - Controllable generation (with bayesian rule and classifier)

    > p(x)是对图片的概率生成, uncondition generative
    >
    > p(y|x) 是classifier generation , also called forward model

    

- Probability evalution

  - with sde 



Weakness

- speed
- there is no natural latent space with lower dimensionality, it is hard to get a representation





### Diffusion and Score-Based Generative Models

Deep generative model

- Explicit : Estimating the probability distribution of data
  - approximating the normalizing constant 
    - Energy-based models (inaccurate )
    - Flow
  - Score-based (estimate the score function)
- Implicit : GAN (can't evaluate probabilities)



From fisher divergence to score matching etc.

- naive fisher divergence  $\frac{1}{2} \mathbb{E} _{p_{data}(x)} [||\nabla_{x}\log p_{data}(x)-s_{\theta}(x) ||_2^2]$
- score matching  $\mathbb{E}_{p_{data}(x)} [\frac{1}{2}||s_{\theta}(x)||_2^2 + tr(\nabla _x s_{\theta}(x))]$ 
  - score matching计算需要1次forward运算得到$s_{\theta}(x)$, 再依次对$s_{\theta}(x)$的每一个维度$s_{\theta}(x_i)$进行反向传播 （反向传播理论上是标量 对向量或矩阵求导）
  - 共计1次forward + Dimension 次 backward, naive score matching is not scalable

- Slicing score matching : 利用随机投影来重写Fisher divergence，绕过Jacobin矩阵的计算
  - Sliced Fisher Divergence : $\frac{1}{2}\mathbb{E}_{p_v} \mathbb{E} _{p_{data}(x)} [||\bold{v}^T\nabla_{x}\log p_{data}(x)-\bold{v}^T s_{\theta}(x) ||_2^2]$
  - Sliced Score Matching : $\mathbb{E_{p_v}}\mathbb{E_{p_{data}(x)}}[\bold{v}^T \nabla _x s_{\theta}(x) \bold{v} + \frac{1}{2}(\bold{v}^Ts_{\theta}(x))^2]$
    - 不需要计算Jaccobian trace，只需要计算Jacobin  二次型内积 ，只需要计算1次backprop
    - 将梯度与内积结合 $\bold{v}^T \nabla _x s_{\theta}(x) \bold{v} = \bold{v}^T \nabla _x (\bold{v}^Ts_{\theta}(x) )$
  - $\bold{v}^T$  is a projection direction, $p_v$  is the distribution of this projection directions
    - projection distribution $p_v$ is typically Gaussian or Rademacher distribution

- Denoising score matching  : 直接利用denoising score matching来估计$p_{data}$

  - $p_{data}(x) -> q_{\sigma}(\hat{x}|x) -> q_{\sigma}(\hat{x})$

  - $\hat{x} = x + \sigma^2 I$

  - 缺点：

    - 估计的实际是noise distribution而不是noise-free distributions
    - 如果降低noise , variance就会变得非常大

    ![截屏2023-02-27 23.09.58](/Users/sirius/Library/Application Support/typora-user-images/截屏2023-02-27 23.09.58.png)

    ![image-20230227232556128](/Users/sirius/Library/Application Support/typora-user-images/image-20230227232556128.png)



- Sampling from score functions : Langevin dynamics

  - 如果只按按照score function的方向，所有的样本最终都会坍塌到概率最大的点
  - 使用langevin dynamics在一定条件下可以保证生成这些$p_{x}$的样本

  ![image-20230227233705471](/Users/sirius/Library/Application Support/typora-user-images/image-20230227233705471.png)























### Diffusion-IPA





### EWSC: Protein design using deep learning, David Baker

- physically based model
  - binder  to folded proteins (cytokine receptor)
  - binder to more flexible molecules
    - Design strategy for binding amyloid forming peptides
- protein design using deep learning
  - simple RL : Monto Calo Tree
  - RF fine tuneing 
    - **IPA and Triangle attention** are not critical
    - **FAPE and recycling are**
  - Protein/NA complexes with RF2
  - RoseTTAFOld All-atom
  - LigandMPNN : Incorporating lignad context for protein sequence design
  - Inpainting 
    - deterministic
    - fail in small description
  - RF diffusion



### OpenFold : Lesson learned and insights gained from rebuilding and retraining AlphaFold2

  why

- Because of the fact that training code is very entangled into kind of the Google infrasturcture system, **they did not release the training code.**
- Three initial motivations
  - full scale retraining (for new applications)
  - modular components (in PyTorch)
  - Knowledge acquisition / reproduce DeepMind's results
    - openfold faster than af2 as code is more plex 
    - **pytorch模型极快的速度就能收敛, fine-tuning 阶段主要解决physical violations 很贵**

## 文献中的生物实验

- **[Language models generalize beyond natural proteins](file:///Users/sirius/Desktop/2022.12.21.521521v1.full.pdf)**

  利用Language Model做了两个任务：inverse folding +无条件结构生成，证明了仅使用序列的模型可以生成大量有效的结构和序列

  方法：将attentiom map特征投影到18个distance bin 做模拟退火

>A total of 228 generated proteins are evaluated experimentally with high overall success rates (152/228 or 67%) in producing a soluble and
>
>monomeric species by size exclusion chromatography.   分子排阻层析 or 分子筛
>
>C 图是为什么仅使用语言模型就能做设计，模拟退火的源头
>
>D 图告诉我们ESM2会给surfrace hydrophilic aa ，hydrophobic aa in the core，可以学习一下此图氨基酸的分类
>
>- 可以利用的点
>
>  - 39个PDB中的de novo 蛋白，非常好的测试集
>
>  - review : ESM2的参数，一共有33 * 20 = 660个attention head 
>
>  - |   num_layers:    |  int = 33,  |
>    | :--------------: | :---------: |
>    |    embed_dim:    | int = 1280, |
>    | attention_heads: |  int = 20,  |



![image-20221225175751977](/Users/sirius/Library/Application Support/typora-user-images/image-20221225175751977.png)
