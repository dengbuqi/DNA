# Dynamic Neural Architecture (DNA)

> 动态神经网络 | 去中心化人工智能 (DAI) 的基础架构

这是我的个人研究项目。目标是构建一个**可动态增删神经细胞**的神经网络，灵感来源于 Hebb 法则（Fire together, wire together）和生物神经元的可塑性。这个项目最终服务于去中心化人工智能（DAI）——让神经细胞可以分布在不同的设备上自主运行。

---

## 核心理念

### Fire together, wire together (Hebbian 法则)

> Let us assume that the persistence or repetition of a reverberatory activity (or "trace") tends to induce lasting cellular changes that add to its stability. ... When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased.

在我的理解中：
- **"reverberatory activity"** = 神经网络的训练过程
- **"cell"** = 网络中的一个神经细胞单元
- **"A repeatedly takes part in firing B"** = 某个激活模式反复出现

我对原始 Hebbian 理论做了一个修改：我不要求 A 在 B 之前激活，而是 **A 和 B 同时激活**。因为在生物上，它们最终会因为同一个刺激而同时放电。

### 神经细胞数学模型

一个神经细胞可简化为一个加权信号函数减去信号泄漏：

```
I = w * Vm - b
```

其中 `w` 是权重（信号增益），`b` 是偏置（泄漏阈值）。

### 理论基础

根据 **通用逼近定理**，一个含足够多神经元的单隐藏层前馈网络可以逼近任意连续函数。同时，**Kolmogorov-Arnold 表示定理** 说明任何多元连续函数可分解为单变量函数的叠加。

因此——理论上——我们的并行细胞结构具有足够的表达能力。**但需要注意**：存在性不等于可学习性，动态结构下的训练收敛性仍需实证验证。

---

## 架构总览

```
输入(784个特征)
  ├── Edge[0]  (flat)  ──→ [NeuralCell[0..N]]  ──→ sum/n ──→ Output(10类)
  ├── Edge[1]  (hier)  ──→ hidden[1→H]→ReLU──→ [NeuralCell[0..M]] ──→ sum/n ──→ Output(10类)
  ├── Edge[2]  (flat)  ──→ [NeuralCell[0..K]]  ──→ sum/n ──→ Output(10类)
  └── ...
```

每个 Edge 有两种模式：

- **扁平模式（默认）**：`Input[i] → 并行 NeuralCell[1→out] → sum/n → Output`
- **层级模式（自动生长）**：`Input[i] → hidden_layer[1→H] → ReLU → 并行 NeuralCell[H→out] → sum/n → Output`

生长触发：扁平 Edge 的细胞数超过 `MAX_CELLS_PER_EDGE`（默认 16）时自动升级。升级不可逆。

---

## 核心机制

### 1. 活力分数 (Vitality Score)

决定哪些细胞应该分裂或删除的依据，从固定的权重绝对值改为：

```
Vitality = grad_norm * avg_activation
```

- **grad_norm**: 该细胞所有权重梯度的 L2 范数（表示"正在学习"的程度）
- **avg_activation**: 该细胞输出激活值的绝对值在 epoch 上的均值（表示"被调用的频率"）

只有 **既被频繁激活、又在学习新东西** 的细胞才被认为有"生命力"。

### 2. 细胞分裂 (Cell Creation)

当一个细胞的活力超过 `vitality_threshold * 2` 时，它分裂出子细胞：

```
分裂前:
  Parent: w=0.8, b=0.1

分裂后:
  Parent: w=0.8, b=0.1 (完全不变，不拆分权重)
  Child:  w~N(0,0.01), b=0.0 (小随机初始化)
```

- Parent 的权重**完全不变**，模型输出不受影响
- Child 从随机权重开始，有 1 epoch 的 grace period 免于死亡
- 每个 epoch 只分裂 top 50% 的高活力细胞，防止爆炸式增长

### 3. 细胞死亡 (Cell Deletion)

当细胞的活力低于 `vitality_threshold` 时，它进入 **dying** 状态：

```
dying: 输出 = 输出 * decay_factor
decay_factor 每 epoch * 0.5
当 decay_factor < 0.01 → 彻底删除
```

- Dying 细胞输出仍被衰减（不是归零），因此仍能积累激活值和梯度
- 如果 dying 细胞的活力在后续 epoch 回升超过阈值，它**死而复生**
- 这种设计给了细胞恢复的机会，避免了"一次性判决"的武断

### 4. 灭绝事件 (Extinction)

当 loss 连续 `patience` 个 epoch 不再下降时触发。这是模拟生物史上的"物种大灭绝"：

```
灭绝事件:
  1. 将所有活力低于 阈值*2 的正常细胞标记为 dying
  2. 清理已死亡的细胞
  3. vitality_threshold /= 2 (门槛降低，后续更容易创建新细胞)
```

灭绝后阈值减半的原因是：随着训练深入，所有细胞的活力值会自然趋于稳定，降低门槛才能继续保持动态性。

### 5. 前向传播归一化

```
output = sum(cell_i(x)) / len(cells)
```

无论一个 Edge 中有 1 个还是 100 个细胞，输出信号的尺度保持一致。没有归一化，当并行细胞增多时输出会膨胀，sigmoid 饱和。

### 6. 自生长中间层

当扁平 Edge 的细胞数量超过 `MAX_CELLS_PER_EDGE`（默认 16）时，系统自动将其升级为层级结构：

```
扁平 (N > 16):
  Input ──→ [cell1..cellN] 各为 Linear(1, 10) ──→ sum/n ──→ Output

层级 (H = max(4, N//2)):
  Input ──→ hidden_layer Linear(1, H) ──→ ReLU ──→ [cell1..cellN] 各为 Linear(H, 10) ──→ sum/n ──→ Output
```

- hidden_layer 权重初始化为 N(0, 0.1) 小随机
- 每个旧细胞的 Linear(1, 10) 权重被复制到新细胞 Linear(H, 10) 的**第一列**，其余列全零
- 这样生长瞬间输出几乎不变，不破坏已有学习成果
- 升级**不可逆**，一旦成为层级模式不会回到扁平

---

## 模型初始化

```
每个 NeuralCell: w=1.0, b=0.0
每个输入特征: 1 个初始 NeuralCell
```

初始模型就是一个全连接层的等价物，但每个连接都是一个独立的 NeuralCell。

---

## 训练流程

```
for epoch in 1..EPOCHS:

    # 阶段 1: 重置活力统计
    model.reset_vitality()

    # 阶段 2: 正常训练 (forward + backward + track_backward)
    for batch in train_loader:
        output = model(data)
        loss = BCE(output, target)
        loss.backward()
        model.track_backward()   # 记录梯度范数
        optimizer.step()

    # 阶段 3: 评估
    val_loss = test(model)

    # 阶段 4: 结构更新 (分裂高活力细胞 / 标记低活力细胞为 dying / 清理死亡细胞)
    model.structural_update()

    # 阶段 5: 检查 loss 是否停滞 -> 触发灭绝事件
    if val_loss 连续 k 个 epoch 不再下降:
        model.extinction()
```

---

## TODO

- [x] NeuralCell 类（活力追踪、dying 状态、grace period）
- [x] NeuralCellEdge 类（管理并行细胞列表）
- [x] Brain 类（整体模型）
- [x] 活力追踪（梯度范数 + 激活值统计）
- [x] 基于活力的细胞分裂（w~N(0,0.01) 小随机初始化，保持输出不变）
- [x] 逐步细胞死亡（衰减因子 + 濒死状态 + 死而复生）
- [x] 前向传播归一化
- [x] 灭绝事件（低活力标记 + 阈值衰减）
- [x] MNIST 训练循环
- [x] 自生长中间层（MAX_CELLS_PER_EDGE 触发 → 自动升级为层级结构）
- [ ] Extinction 时低活力细胞权重重分配到邻近细胞（功能代偿）
- [ ] Forward-Forward Algorithm（无反向传播训练）
- [ ] 分布式 DAI（细胞在不同设备上运行）

---

## 下一阶段 (Next Level)

### Forward-Forward Algorithm

FF 算法不需要反向传播——每个层/每个 cell 有自己的局部损失函数：
- 正样本和负样本分别前向传播
- cell 学习区分两者的激活模式
- 每个 cell 只需要知道自己的输入和局部目标

这与我们的 DAI 愿景高度一致——每个细胞自治，只需交换激活值。

### DAI (Decentralized Artificial Intelligence)

将 NeuralCell 分布到不同设备：
- 每个设备管理一个子集的 NeuralCell
- 设备间通过轻量级通信（激活值交换）协作推理
- 无需全局梯度同步

### Self-Perception, Self-Learning, Self-Evolution

最终目标：模型能够在运行时无监督地感知环境变化、自主调整结构、持续演化。
