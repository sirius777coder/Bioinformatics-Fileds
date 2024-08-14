## 李沐深度学习视频

### GPT1/2/3  - 23

1. GPT1微调用最后一个token的embedding，进行信息抽取；GPT23和1的区别，不用微调下游任务，把下游任务的输入输出构造成和微调一样的形式 ; GPT2和1的区别：提前normalization,初始化方式不同
2. 现在的有用的生物大模型还处在GPT1和bert 的阶段，还需要fine-tuning ，GPT2/3的卖点是一个zero shot, few shot，将子任务构造成预训练的数据格式，进行promt engineering ，而不是基于梯度的模型更新，GPT2/3做few shot的原因是模型太大了，没办法更新
3. 但GPT3完全 few shot有自己的问题，1.当下游例子真的很多的时候，如英语翻译法语，不可能全部赛到一个example里面；2.模型无法把从一次forward中得到的信息永久的存储下来，每次都需要到一个example
4. 为什么需要多头注意力机制：GP T参数dmodel 增加1000倍，但每一个头的维度dread 大小不到2倍(如64->128)，只是会把头的个数nheads增加100倍(12->96)
5. 从模型效果来看，小模型适合小的批量，这样主动带来一些噪音防止过拟合；而大模型则更适合大批量大小，研究发现大模型即使一个批量里没有什么噪音，好像也不容易过拟合

### Llamma 3.1  - 54

GQA, mask self-attention, scaling laws

1. 大模型如何处理数据集？大公司很多人肉眼先看一些数据集，找到一些数据集的pattern，再对整个数据集进行清洗。这种方式称为启发式的数据处理方法
2. Llamma3模型架构
   1. 标准的稠密Transformer架构 -- dense Transformer architecture，与Llamma, Llamma2区别不大
   2. 与之前工作的一些区别：GQA，grouped-query attention
      1. 背景：在模型推理时，为了推理速度，现在大模型都使用了kv-cache保存之前token的Key和Value，但是key-value内存占用太大了，一个token经常就是1-2M，1000个token就有1-2个G，因此KV-cache很占用内存
      2. 目的：在保证推理速度的前提下（保留kv-cache），使用GQA节省内存，特别是对70B以上的模型，在训练时给query一个head数目，给key/value一个head数目
      3. 方法：多头注意力机制中，N个头本来会诞生N组Q，K，V；现在依然是N组Q，但是K和V不同的头group一下，比如两两K和V，共用一组投影矩阵，这样在推理的时候一来不需要全部的K, V weight matrix，二来不需要为每一个头保留K和V，只需要每一个group保留K和V即可。Lamma3用了8 key-value heads表示8个key-value共享一个投影矩阵，而Llamma3 70B模型默认是64个头，将KVcache的缓存压力降低了八倍。**注意：GQA是在降低每一个token内部多个head的KVcache，不涉及token之间的情况，也不会降低query的head。现在有Attention Heads与，Key/Value Heads两个头的大小，Group大小=Attention_heads / KeyValue_heads**
   3. 与之前工作的一些区别：如果进入模型序列中有多个文档的样本，做self-attention时会进行mask。模型训练时一个序列token 12K, 8K等等有可能来自多个文档的拼接，每个token应该只算自身文档内部的attention score，其他位置的score应该设置为0。这对特别长的token有用
   4. 与之前工作的一些区别：Llamma3有128K tokens，在GPT4 10k tokens前提上增加了多语言的tokens
   5. 与之前工作的一些区别：使用了更大超参数的RoPE
   6. Llamma3 scaling laws:模型越大、性能越好是指导大模型的思想
3. Llamma3 saling laws ：模型越大、性能越好，可以在小模型上训练将Loss与模型大小进行线性拟合，这样就可以大致预测最终想要到达某一个loss，模型大小是什么。但是每个大模型的scaling laws的拟合关系都需要自己来计算。
   1. 目前Scaling laws通常使用next-token prediction loss或者validation loss，而不是下游任务的loss
   2. 本文使用了两个阶段scaling laws
      1. 根据训练的FLOPs预测下游任务（e.g. 数学题）的negative Log-likelihood，做一个FLOPs-NLLL的线性关系预测
      2. 使用下游任务的NLLLoss和下游任务的精度做一个关系，比如在下游数据集预测下一个词误差0.1，精度0.8；误差0.05，精度0.9，这样下游任务精度就和模型架构没用什么关系
         > 任务1 scaling laws更适合预测NLLLoss，而且和模型架构数据强相关，不同的模型和数据都需要算一个新的scaling laws；而任务2NLLLoss与精度则和模型架构没什么关系。这样可以同时预测scaling laws和下游任务的性能与NLLLoss
         >
      3. 文章figure2画图表示不同的计算资源前提下，用多大的训练样本，可以实现最好的性能
