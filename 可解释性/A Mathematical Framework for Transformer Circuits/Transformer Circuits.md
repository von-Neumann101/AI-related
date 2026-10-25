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
比如，查询向量 $q=W_Qx$，当残差流换成 $x'=Rx$ 以后，只需要同时改变投影矩阵 $W_Q'=W_QR^{-1}$ 就能保持查询向量不变。所以以何种方式解释坐标轴，并不会影响模型的表示，所以残差流中**向量任意的维度并不具有所谓的语义**（因为这是可以随意更改的）

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

如果将 $x$ 看作一个二维矩阵（每个 Token 对应一个向量）, $A$ 负责在不同 Token 之间混合信息，$W_OW_V$ 独立地作用于每个向量。
