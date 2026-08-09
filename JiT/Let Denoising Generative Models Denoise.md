当今的去噪扩散模型学习的是**预测噪声，或者被噪声污染的量**。
根据流形假设，**自然数据应该位于低维流形上**，而噪声污染的数据则不是。根据这一假设，最好直接**预测干净数据**，这能让容量不足(under-capacity)的模型能在高维的空间有效运行
> [!NOTE] "under-capacity"
> 例如一张$16\times16$的RGB图片，展平后$16\times16\times3=768$维，如果模型的hidden dimension有$384$维，那么模型实际上在做：
> $$\mathbb R^{768}\mapsto\mathbb R^{384}\mapsto\mathbb R^{768}$$
> 从直觉上说，768个方向具有的随机度，中间只保留384维的表示。这就叫做"under-capacity"
# Introduction
