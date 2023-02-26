# Biochemistry Background 

[toc]

## 基本单词

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
- Tryptophan 色氨酸 N (吲哚)

- Aspartate 天冬氨酸 D
- Glutamate 谷氨酸 E

- Asparagine 天冬酰胺 N
- Glutamine 谷氨酰胺 Q

- Cysteine 半胱氨酸 C

- Methionine 甲硫氨酸

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
