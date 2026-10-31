https://transformer-circuits.pub/2021/framework/index.html
# Transformer
这里我们会用新的角度解释 Transformer，以便于后续的工作
## Model Simplifications
值得注意的一点，我们**抛弃之前为了加速运算而使用的种种技巧**（本文最有代表性的是多头注意力的独立、可加性），这有助于我们从更加“解释性”的视角看待模型

此外，本文还进行一些简化：
- 只研究不包含 MLP 层的纯注意力（MLP 部分一直更加难以理解）
- 忽略层归一化（除了一个随输入变化的缩放因子之外，层归一化可以合并到相邻的权重矩阵中）
- 不考虑偏置项
## High-Level Architecture
使用 Decoder-Only：
![[Pasted image 20261025165059.png|584]]
[[Transformer Circuits#Attention Heads are Independent and Additive|多头注意力具体见下面的推导]]

这里需要理解**残差流是贯通整个模型的信息通道**，注意力层和 MLP 层都会：
1. 通过线性投影，从残差流中“读取”输入
2. 计算得到结果
3. 再通过线性投影，将结果**以加法形式**“写入”残差流
## Virtual Weights & the Residual Stream
残差流相当于是一个不断被各种组件修改信息的一个通道，一段文字被输入进去后，经过多道工序的加工，最后输出。
残差流有一个特点：**旋转残差流并同时调整所有与之交互的矩阵，不会改变模型的输入输出函数**
比如，查询向量 $q=W_Qx$，当残差流换成 $x'=Rx$ 以后，只需要同时改变投影矩阵 $W_Q'=W_QR^{-1}$ 就能保持查询向量不变。所以以何种方式解释坐标轴，并不会影响模型的表示，所以残差流中**向量任意的维度并不具有所谓的语义**（因为这是可以随意更改的） ^834731

**虚拟权重**：
残差流还允许我们合并（相乘）某一层的输出权重与另一层的输入权重。
比如，前一个注意力头计算得到结果 $r$，通过输出矩阵 $W_O$ 写入残差流：
$$x'=x+W_Or$$
后面的注意力头通过 $W_Q$ 读取：
$$
\begin{aligned} q&=W_Qx'\\&=W_Q(x+W_Or)\\ &=W_Qx+\underbrace{W_QW_O}_{\text{虚拟权重}}r \end{aligned}
$$
## Subspaces and Residual Stream Bandwidth
按照之前所说的“信息通道”来理解残差流，不同层可以将信息存储在**不同子空间**中，从而向不同的后续层发送不同的信息。残差流的维度可以被看作**某种“内存”或“带宽”**

多头注意力的每个头**只作用于相对较小的子空间**，因此它们可以非常容易地向完全不同的子空间写入信息，彼此之间**几乎不发生干扰**。信息一旦被添加到某个子空间，就会一直保留下来，除非后续某一层主动将其删除。
原始 Token Embedding 以及 Unembedding，通常只与残差流相对较小的一部分维度发生相互作用

但是要注意到，信息并不会自动丢失，那么残差流的“带宽”一定会爆，信息只能以某种**叠加**的方式完成通信
> 本文得到了一些结论：某些 MLP 神经元和注意力头，可能承担某种“内存管理”的角色——它们通过读取其他层写入的信息，再写入这些信息的相反值，清除残差流中先前被设置的某些维度
## Attention Heads are Independent and Additive
本文将 $W_O^H$ **按照不同注意力头划分为大小相同的矩阵块**：
$$
W_O^H=[W_O^{h_ 1},W_O^{h_ 2},\ldots]
$$
于是有：
$$
 \begin{aligned} W_O^H \begin{bmatrix} r^{h_1}\\ r^{h_2}\\ \vdots \end{bmatrix} &= [W_O^{h_1},W_O^{h_2},\ldots] \begin{bmatrix} r^{h_1}\\ r^{h_2}\\ \vdots \end{bmatrix}\\ &=\sum_i W_O^{h_i}r^{h_i} \end{aligned}
$$
Transformer 原文使用了拼接，这只是为了加速的手法。加法方式会更加触及本质
## Attention Heads as Information Movement
注意力头最基本的作用是**移动信息**，它们从某个 Token 的残差流中读取信息，再将这些信息写入另一个 Token 的残差流中。也就是注意力头给每个 token 一个权重，乘起来作为读取到的信息，放到正在处理的 token 的残差流中

本节最重要的观点是：**决定从哪些 Token 移动信息，与决定读取什么信息、以及如何将这些信息写入目标位置，是完全可以分开的**
第一步： 根据残差流，为每个 Token 计算 Value 向量：
$$v_i=W_Vx_i$$
第二步： 根据注意力模式，对 Value 向量进行线性组合，得到结果向量：
$$r_i=\sum_j A_{i,j}v_j$$
第三步： 为每个 Token 计算注意力头的输出向量：
$$h(x)_i=W_Or_i$$
注意：$W_V$ 和 $W_O$ 作用于每个 Token 的向量维度，而 $A$ 作用于位置维度

三个步骤都是矩阵乘法，我们将他们合并为一次运算：
$$
 h(x)= (\mathrm{Id}\otimes W_O) (A\otimes\mathrm{Id}) (\mathrm{Id}\otimes W_V)x=(A\otimes W_OW_V)x
$$
> 张量积： $(A\otimes W)x=AxW^T$

如果将 $x$ 看作一个二维矩阵（每个 Token 对应一个向量）, $A$ 负责在不同 Token 之间混合信息，$W_OW_V$ 独立地作用于每个向量

注意到 $W_O$ 和 $W_V,\ W_K$，总是成对出现—— $W_OW_V,\ W_Q^TW_K$ 两个低秩矩阵，从某种意义上说，KQV 只是计算这些低秩矩阵产生的中间结果

**虚拟注意力头**： ^d5d6e3
$$
\begin{aligned} &(A^{h_2}\otimes W_{OV}^{h_ 2}) (A^{h_1}\otimes W_{OV}^{h_ 1})\\ &\qquad = (A^{h _2}A^{h_ 1}) \otimes (W_{OV}^{h _2}W_{OV}^{h_ 1}) \end{aligned}
$$
这个乘积在功能上等价于另一个注意力头——注意力模式为 $A_{h_2}A_{h_1}$，OV 矩阵为 $W_{OV}^{h_2}W_{OV}^{h_1}$
# Zero-Layer Transformers
该模型接收一个 Token，经过 Embedding 和 Unembedding 预测下一个 Token 的 logits：
$$
T=W_UW_E
$$
这个线性投影用于二元预测，结果近似 bigram 的对数似然
注意，残差流是一个通道，从模型开始连接到模型结束。所以，有一条路径： Token Embedding **直接**沿着残差流传递到 Unembedding，中途不经过任何其他层。我们称之为**直接路径**

虽然这一项单独来看是表示二元语法的，但是在大模型中其他部分也进行这种二元预测，所以这一项不会精确表示完整的二元语法。
> 这一项似乎经常帮助模型表示那些**无法通过更加一般的语法规则描述的二元语法统计关系**。例如，在文本中，`Barack` 后面经常接着 `Obama`。
# One-Layer Attention-Only Transformers
单层纯注意力 Transformer 可以理解为一个二元语法模型与若干“ *跳跃三元语法* ”模型的集成（`A ... B C` 的概率）
每个注意力头可以选择性的让当前的 token 关注之前的某个 token，复制其信息，并调整下一个可能出现的 token 的概率
## The Path Expansion
整个单层纯注意力 Transformer 可以由一个公式表示：
$$
{ T= (\mathrm{Id}\otimes W_U) \cdot \left( \mathrm{Id}+ \sum_{h\in H_1}A^h\otimes W_{OV}^{h} \right) \cdot (\mathrm{Id}\otimes W_E) }
$$
其中 $A^h= \operatorname{softmax}^{*} \left( t^TW_E^TW_{QK}^{h}W_Et \right)$
![[Pasted image 20261026090620.png|578]]
关键的来了——我们将乘积展开：
$$
\boxed{ T= \mathrm{Id}\otimes W_UW_E + \sum_{h\in H} A^h\otimes \left(W_UW_{OV}^{h}W_E\right) } 
$$
第一项是 0 层 Transformer 中出现的直接路径，他的**唯一贡献是二元语法**（因为 $\text{Id}$ 对 token 维度进行处理，不会产生任何改变）
## Splitting Attention Head Terms into QK and OV Circuits
对于每个注意力头 \(h\)，都有一项：
$$
A^h\otimes \left(W_UW_{OV}^{h}W_E\right)
$$
其中：$A^h= \operatorname{softmax} \left( t^TW_E^TW_{QK}^{h}W_Et \right)$

- **QK 电路**：$W_E^TW_{QK}^{h}W_E$ 称为 QK Circuit，为每一对 QK token 提供注意力分数
- **OV 电路**：${ W_UW_{OV}^{h}W_E }$ 称为 OV Circuit，描述了如果模型关注某个给定的 token 这个 token 会如何影响输出 logits

**OV 与 QK 的独立性**：
设想模型运行两次：
1. 收集每个注意力头的注意力模式。这些模式只依赖于 QK 电路
2. 将注意力模式替换为第一次收集到的、已经“冻结”的模式，那么 logits 就是输入 Token 的**线性函数**
## Interpretation as Skip-Trigrams
刚刚我们通过 OV 和 QK 的拆分，发现神经网络参数已经成为作用于 Token 的简单线性函数或双线性函数

变量之间的**可替代性**：例如，某个注意力头的一个权重可能为零，是因为另一个注意力头也会关注同一个 Token，并执行它本来要执行的相同任务。
### Copying / Primitive In-Context Learning
单层模型中的大多数注意力头，都将其很大一部分容量用于**复制**

OV 电路使得某个 Token 一旦受到注意力头的关注，就会提高对该 Token 本身的预测概率，并在较小程度上提高对类似 Token 的预测概率。随后，QK 电路只会**向前（不是向右而是向左）关注那些有可能成为下一个 Token 的 Token**
![[Pasted image 20261026143329.png]]

对于不常见的单词，如果前面有空格 `" Ralph" → [" Ralph"]` ；而没有空格就可能：`"Ralph" → ["R", "alph"]`
> 事实上，我们有时会观察到一些注意力头，它们似乎在一定程度上专门处理那些在没有前置空格时会被拆成两个 Token 的单词。
> 当这些注意力头遇到一个被拆分的 Token（例如 `R`）时，它们会向前关注那些可能包含完整单词的 Token（例如 `Ralph`），然后预测后续部分（例如 `alph`）。

![[Pasted image 20261026153532.png]]

这几种复制行为归纳为如下四种抽象模式：
![[Pasted image 20261026154118.png]]
此外，模型还能学出诸如 LaTex `\right` 闭合、Python 括号闭合等
### Primarily Positional Attention Heads
主要关注当前 Token 或前一个 Token。
![[Pasted image 20261026154843.png]]
### Skip-Trigram Bugs
我们把单层模型分解为两个电路——OV 和 QK，就像把一个函数表示为：
$$f(a,b,c)=f _1(a,b)f_ 2(a,c)$$
这种形式无法灵活地捕捉三个变量之间的相互作用

比如：
两个短语：
- `keep … in → mind`
- `keep … at → bay`

QK 学会了：当前 Token 为 `in` 或 `at` 时，都应该关注前面的 `keep`
OV 学会了：学会了只要关注 `keep`，就提高 `mind` 和 `bay` 的 Logits

所以输出也是合理的：
- `keep … at → mind`
- `keep … in → bay`

OV 电路只知道它读取的是 `keep`，并不知道当前 Token 是 `in` 还是 `at`。这种独立性又好又坏
## Summarizing OV/QK Matrices
**OV 和 QK 矩阵的秩极低**：
形状是 $50000\times50000$，但是秩只有 $d_{\text{head}}=64\text{ or } 128$

**观察单个矩阵元素时，我们经常能发现一些更加简单的结构线索**：
源 Token 是人名，那么与它对应的最高分 Query 往往是类似 `by` 的 Token；源 Token 是地点名称，对应的最高分 Query 往往是类似 `from` 的 Token——矩阵中可能存在某种聚类结构

**复制行为广泛存在于 OV 矩阵中，而且可以说是其中最有趣的行为之一**：
> 合适的矩阵分解方法或降维方法，可能会提供非常有价值的信息

**复制**：
一种自然考虑复制的方法，是研究矩阵的特征值、特征向量——OV 电路的某个特征向量具有正特征值，每个正特征值都意味着这个电路在每一个特征方向的“复制倾向”，一个 token向量表示为这样线性组合。
显然 OV 电路不大可能是正定的，同一集合内的 Token 会提高其他同集合 Token 的概率，同时降低另一集合中 Token 的概率，例如：
- 一组表示男性词汇，另一组表示女性词汇。
- 一组表示单数词汇，另一组表示复数词汇。

> 在实验中，12 个注意力头里，有 10 个似乎表现出显著的复制行为！
> ![[Pasted image 20261028114112.png]]
> 我们不认为它是完美的，但它似乎能够提供相当有力的复制行为证据，而且在实验中与人工检查以及其他定义高度一致。
# Two-Layer Attention-Only Transformers
> 深度学习研究的是具有许多层的深层模型。从经验上看，这些模型非常强大。那么，这种能力究竟来自哪里？
> 一种直觉是：深度允许模型进行**组合**（Composition），而组合又带来了强大的表达能力。（如第二层组合第一层的信息）
> 注意力头之间的组合，是单层与双层纯注意力 Transformer 的关键区别。
> **如果不存在组合，那么双层模型只不过是拥有了更多注意力头，用来实现更多跳跃三元语法。**
### Three Kinds of Composition
第二层的 $W_Q,W_K,W_V$ 读取第一层写入残差流对应子空间的信息，然后通过 $W_O$ 所确定的子空间把信息协会残差流
QK 是对注意力模式的影响，而 V 是对信息的影响。QK 能产生更加复杂的注意力模式（类似于函数的复合），而 V 对于信息的影响是独立于注意力的
## Path Expansion of Logits
$$
\boxed{ \begin{aligned} T={}& \mathrm{Id}\otimes W_UW_E\\[2pt] &+\sum_{h\in H_1\cup H_2} A^h\otimes(W_UW_{OV}^{h}W_E)\\[2pt] &+\sum_{h_2\in H_2}\sum_{h_1\in H_1} (A^{h_2}A^{h_1})\\ &\qquad\otimes (W_UW_{OV}^{h_2}W_{OV}^{h_1}W_E) \end{aligned} } 
$$
值得注意的是，单层出现的组件仍然独立的出现——这表明了残差流信息通道的作用。
我们关注第三项，我们称之为[[Transformer Circuits#^d5d6e3|虚拟注意力头路径]]。这和普通的注意力头很相似——由两个注意力头的注意力模式组合得到
## Path Expansion of QK Circuit
用 $h\in H_2$ 表示正在分析的第二层注意力头，$h_q,h_k\in H_1$ 表示第一层注意力头，$C_{QK}^h$ 是将 Token 映射为注意力分数的 QK 电路：
$$
\boxed{ \begin{aligned} C_{QK}^{h} ={}& \mathrm{Id}\otimes\mathrm{Id}\otimes (W_E^TW_{QK}^{h}W_E) \\[3pt] &+\sum_{h_q\in H_1} A^{h_q}\otimes\mathrm{Id} \\[-2pt] &\qquad\otimes \left( W_E^T(W_{OV}^{h_q})^TW_{QK}^{h}W_E \right) \\[3pt] &+\sum_{h_k\in H_1} \mathrm{Id}\otimes A^{h_k} \\[-2pt] &\qquad\otimes \left( W_E^TW_{QK}^{h}W_{OV}^{h_k}W_E \right) \\[3pt] &+\sum_{h_q\in H_1}\sum_{h_k\in H_1} A^{h_q}\otimes A^{h_k} \\[-2pt] &\qquad\otimes \left( W_E^T(W_{OV}^{h_q})^TW_{QK}^{h} W_{OV}^{h_k}W_E \right) \end{aligned} }
$$
这里的每一项都是如下形式：
$$
A_q\otimes A_k\otimes W
$$
- $A_q$ 描述 Query 一侧的信息如何在不同 Token 之间移动
- $A_k$ 描述 Key 一侧的信息如何在不同 Token 之间移动
- $W$ 描述两侧的信息如何组合形成注意力分数
## Analyzing a Two-Layer Model
> 小型双层模型似乎经常（但并非总是）具有一种非常简单的组合结构：唯一的组合形式，是第一层中的某一个注意力头与第二层中的若干注意力头之间发生的 K-Composition。

第一层的一个注意力头，主要关注前一个 Token，也会较小程度上关注当前位置恶魂再往前两个位置的 Token（红色）
第二层中的注意力头，就是我们所称的**归纳头**

定义了一个量，用于计算第二层注意力头的 QKV 多大程度上读取了某个第一层注意力头写入的信息（见原文）
大多数注意力头的组合强度都远小于随机矩阵产生的组合强度。
![[Pasted image 20261029194118.png]]
***（待研究）***

**Value 加权注意力模式**：
缩放因子为源位置 Value 向量的范数：${ \widetilde A_{i,j}^{h} = A_{i,j}^{h}\left\|v_j^h\right\| }$
> 这种方法特别有用，因为当注意力头没有找到自己正在寻找的 Token 时，有时会将某些 Token 当作默认位置或“休息位置”。这些默认位置的 Value 向量通常很小，因此 Value 加权注意力模式能够提供更多有效信息。

水绿色的归纳头，经常会关注此前出现过的、位于当前下一个 Token 的先前实例的位置。
## Induction Heads
![[Pasted image 20261029204700.png|486]]
在输入 `Pot` 这个 token 的时候，归纳头注意之前的 `ters`
> 归纳头会在上下文中搜索当前 Token 此前出现过的位置。如果没有找到，它就会关注第一个 Token（在我们的模型中，这是一个放在序列开头的特殊 Token），并且不执行任何操作。(Attention Sink)

单层模型的复制头：
$$
[b]\ldots[a]\rightarrow[b]
$$
双层模型的归纳头：
$$
\boxed{ [a][b]\ldots[a]\rightarrow[b] }
$$
可以看出，归纳头找到了某种**语言模式**——它不再只是笼统地寻找可能重复某个 Token 的位置，而是知道这个 Token 此前如何被使用，并寻找类似的情况（他能**从输入里随机应变**，而不是像复制头一样**统计训练数据里出现的规律**）

![[Pasted image 20261030151006.png]]
蓝色代表 logit effect，定义为 $\left( W_U W_O^h r_{\mathrm{pres\_tok}}^h \right)_{\mathrm{next\_tok}}$ ，表示这个头在当前 token 上对下一个 token 的 Logits 的贡献
红色就是注意力
### How Induction Heads Work
![[Pasted image 20261030152742.png]]
*Key 和 Query 的颜色强度表示各个 Token 对注意力分数的提升幅度*

前一层把前一个 token 的信息写入（大多是复制头）：
$$
x_j^{(1)} = x_j^{(0)}+ W_{OV}^{\mathrm{prev}}x_{j-1}^{(0)}
$$
第二层的 Key：
$$
K_j= W_K^{(2)}x_j^{(0)} + W_K^{(2)}W_{OV}^{\mathrm{prev}}x_{j-1}^{(0)}
$$
这说明位置 $j$ 的 Key，包含了位置 $j-1$ 的 Token 信息。同时，另一部分是来源于原始信息的（这里也许会表现为复制头），但是真正经过前一层加工的信息是前一个 token——原本为 `[A,B,C,A]` 在第二层 Key 中变为 `[_,A,B,C]`
假设目前读最后一个 A，注意这里的 Q：
$$
 Q_4=W_Q^{(2)}x_4^{(1)}
$$
而第二个位置的 B 包含第一个 A 的信息，所以 $Q_4K_2^T$ 是在比较 $A$ 和 $A$，同时注意到这种点积表示的是两个向量的相似度，这会使得位置 4（当前位置）对位置 2 非常关注
