# Whole-Slide Histology BBox Proposal Troubleshooting（更忠实于原回答版本）

这不是普通的 tissue detection，而是 **multi-section slide 上的 instance-aware bbox proposal**：你当前最优方案已经把平均目标覆盖做得很高，但它仍然主要回答“哪里像 tissue”，还没有真正回答“这些弱信号到底属于这个 target section，还是属于旁边那个 section”。所以 hard case 里剩下的错误，很自然会集中在两端：**弱外缘被漏掉**，以及 **邻片被误吞进去**。

先指出一个评价层面的逻辑问题：你现在用 50/30/20 的加权均值选最优方案，这对找全局折中方案有用；但如果 downstream mask extraction 对截断极其敏感，那么 **coverage 不应该只体现在均值里**，还应该有 **hard floor** 或至少看 tail risk。`mean target recall ≈ 0.9936` 很高，但仍可能掩盖少量灾难性截断。再加上 `proposal-area / GT-crop-area ≈ 0.8841` 已经小于 1，说明平均紧凑性其实不是当前主矛盾，真正的问题是 **尾部失败**，不是平均值。

---

## 我对问题的理解

你现在的 pipeline 可以概括成三步：

1. 在 overview 上构造一个 **class-agnostic tissue score**
2. 从这个 score 的 connected component 得到 seed / candidate
3. 用 fixed expansion + directional projection 把 seed 变成 bbox

这个设计对“有没有 tissue”是合理的，但对“**多个相邻 section 中，当前 bbox 到底属于哪一个**”是不够的。你要优化的不是单纯 tissue recall，而是：

- 对 target section 要高 recall
- 对非 target section 要低 contamination
- 还要 compact

也就是：**这是一个“目标选择 + 边界外推 + 邻片排斥”的联合问题**，而你当前方法主要只有第一项的一半能力。

---

## 当前方法为什么还会错

### 1. 根本错位：你在做的是 tissueness，不是 section ownership

`hybrid = max(residual, legacy_score)` 本质上是一个 **OR 门**。
它能提高 recall，但不会告诉你某个 overview 像素为什么高：

- 是 target 的浅染边缘
- 是 neighbor 的深染 core
- 是背景阴影、折痕、非纯白玻片区域
- 还是局部 artifact

换句话说，**当前 score 没有“归属”概念**。这正是为什么一旦 slide 上 section 靠得近，proposal 会被邻片误导。

---

### 2. 信号选择本身带有系统性偏差

对 Gallyas / myelin，`residual = blurred_background - gray` 天然偏爱 **深染、低灰度的 myelin-rich core**；而浅染外缘、外层轮廓、局部亮薄区域更容易变弱。`legacy = max(inv_gray, nonwhite)` 又比较宽松，容易把任何“不是很白”的区域当成 tissue-like signal。两者取 `max` 后，等于把“高精度 core cue”和“高召回 weak cue”混在一起使用，结果通常是：

- **seed 只盯住深染 core**
- **expansion 却被宽松 cue 拉向不该去的地方**

数字病理里这类现象并不稀奇：scanner 或 thumbnail 级 tissue detection 对弱染、疏松、很薄的组织最容易漏检，thumbnail 分辨率还会丢掉 thin slivers 和 detached micro-islands。

---

### 3. 候选框定义问题：seed 一旦偏了，后面基本都在补锅

你现在从 overview 候选连通域生成 seed box。这里有两个典型坏情况：

- **split failure**：target 的浅外缘没有进连通域，seed 只包到内部深染 island
- **merge failure**：target 和邻片在 overview 上被弱桥接、blur、局部非白背景连成一个 component

一旦出现前者，后续 expansion 只能靠几何规则猜 missing footprint；
一旦出现后者，后续 expansion 再聪明也已经在错对象上做事。

这也是为什么你说问题不再只是 dorsal/top 漏掉，而是 middle/lower footprint 仍然不理想——这通常说明你修掉的是某个方向的手工偏置症状，不是底层的 target support estimation 问题。

---

### 4. 几何扩张策略的问题，不只是“参数还没调够”

固定比例扩张是 blind 的。
方向性投影扩张虽然比 blind 更好，但它把二维形状压成了一维 summary，本质局限很明显：

- target 的弱外缘如果是 **低幅、宽分布**，投影不一定显著
- neighbor 如果在同一行/列方向上出现第二个 peak，projection 会把它看成 target 的延续
- 如果 seed 本身偏心，固定或对称扩张会把 bbox 往错误方向放大

所以 projection-based growth 很适合修“某边明显缺一截”的问题，但不擅长处理 **close neighbor + weak contour + irregular footprint** 的组合。

---

### 5. 低分辨率信息不足，不是 bug，而是瓶颈

只用 overview 的最大问题不是 recall 均值，而是 **边界可判别性**。对于 myelin，真正难的往往不是“有没有组织”，而是“哪一圈弱纹理、弱染色、低对比外缘还算同一个 section 的一部分”。这个信息在 overview 上常常已经被平均化了。

---

### 6. 推理目标和评估目标不一致

你现在的评估已经是正确方向：在 level0 空间直接看：

- target GT mask recall
- other-section overlap
- compactness

问题在于：**这些 penalty 还没有进入 inference 本身**。当前 proposal 生成还是“看当前 box 该怎么长”，而不是“让 box 在 50/30/20 目标下最优”。

也就是说，**neighbor-overlap 只在评估时被惩罚，在生成时没有被显式阻止**。这是我认为现在最值得修的地方。

---

## 这两类错分别是怎么产生的

| 错误类型 | 常见根因 | 当前逻辑的失效点 | 最直接修补 |
|---|---|---|---|
| bbox 过大、吞邻片 | neighbor 的深染区域或 weak tissue signal 被当成 target 延伸；merged component；宽松 cue 被 `max` 放大 | 没有 rival penalty；directional expansion 只看“是否有信号”，不看“信号属于谁” | 多种子竞争式分配 + 显式 non-target penalty |
| bbox 截断 target | 浅染外缘、薄边、middle/lower footprint 在 overview 上弱；seed 只抓 deep core | residual 偏深染；weak fringe 不连续；projection 对 broad/weak 外缘不敏感 | strong-core / weak-fringe 重建 + 局部高分辨率边界校正 |
| 又大又偏 | seed 偏心，同时 target 外缘又弱 | fixed / directional expansion 在“补面积”，不是在“选对象” | per-side objective optimization，而不是单纯放大 |

---

## 最值得优先做的改进

核心思路不是再找一个更 clever 的单一 score，而是把表示改成三层：

- **core**：高精度、确定属于某个 section 的内部
- **fringe**：高召回、可能属于某个 section 的弱边缘
- **rival**：同 slide 上其他 section 的 core / support

你现在缺的不是更强的 tissue score，而是 **target-conditioned support map**。

---

## 1）在现有 classical pipeline 上，最该先做的改进

### A. 强 core / 弱 fringe 的两阶段表示，替代“同一个 hybrid score 既当 seed 又当 growth cue”

**方法原理**

不要再让同一个 `max(residual, legacy)` 同时负责 seed 和扩张。改成：

- **strong core map**：高精度，宁可小，不要脏  
  例：高阈值 residual + 光密度 / stain-related cue + 非背景抑制
- **weak fringe map**：高召回，宁可松，但只作为可生长区域  
  例：较低阈值的 residual / legacy / optical density / entropy / edge 的并集

然后从 strong core 出发，只允许在 weak fringe 里生长。

**为什么适合这个问题**

因为 myelin 的核心难点正是：深染 core 和浅染 fringe 的信号统计不同。你现在的问题，本质上是把这两种角色混用了。

**代价**

低到中等。只改 overview 级特征和后处理，不依赖 level0，不依赖读取后端。

**我会怎么做**

- 新增 2 个 cue，但只放进 fringe，不直接放进 seed：  
  - **optical density（光密度）或简单颜色轴**  
  - **local entropy / gradient magnitude（局部熵 / 梯度强度）**
- 不建议让 entropy 单独触发 tissue；它容易响应杂质、折痕、文字。要加上“靠近 core / 非背景 / connected to marker”的门控。

---

### B. 用“多种子竞争式分配”替代“单 seed 自己往外长”

这是我最推荐的 classical 实验。

**方法原理**

1. 在 whole slide overview 上先找出所有 **strong cores**
2. 构造一个较宽松的 **weak support mask**
3. 用 **marker-controlled watershed（标记控制分水岭）**、**geodesic reconstruction（测地重建）** 或最简单的 **Voronoi partition（按最近种子划归）**，把 weak support 分配给各个 core
4. 每个 target section 先得到一个 **instance-conditioned pseudo-mask**
5. 再从这个 pseudo-mask 出 bbox

**为什么适合这个问题**

因为你的核心 confounder 不是 background，而是 **同 slide 上别的 brain section**。只要所有 section 同时进入竞争，很多“吞邻片”的错误会立刻减少。

**它能解决什么**

- target / neighbor 弱桥接导致的 merged candidate
- 近邻 section 都很 dark 时，projection 被第二个 peak 误导
- 只靠 fixed expansion 导致的过度膨胀

**代价**

低到中等。几乎完全 classical，且和 backend 解耦。

---

### C. 把非目标 section 惩罚，显式写进 box 优化目标

这是第二个高优先级实验。

你现在评估已经有了三项指标，所以完全可以把它变成 inference objective 的近似版本：

```math
J(b_k)=0.5\cdot \widehat{Cover}_k(b_k)-0.3\cdot \widehat{Leak}_k(b_k)-0.2\cdot \widehat{Excess}_k(b_k)
```

其中：

- \(\widehat{Cover}_k\)：当前 box 对 target pseudo-mask 的覆盖率
- \(\widehat{Leak}_k\)：当前 box 对 rival pseudo-mask 的覆盖率
- \(\widehat{Excess}_k\)：box 相对 target pseudo-mask tight box 或 hull box 的过量面积

然后对 left / right / top / bottom 四条边做 coordinate descent（坐标下降）或 discrete search（离散搜索）：

- 每次移动一条边
- 只有当新增 strip 带来的 `target gain - rival penalty - area penalty` 为正时才继续长

**为什么适合**

这一步第一次把你的 50/30/20 目标从“评估”搬进了“生成”。

**它能解决什么**

- box 过大但没有更多 target coverage 的情况
- 相邻切片很近时的 leak
- 由于 seed 偏心导致的一侧无意义宽扩张

**代价**

低。这是最应该快速验证的 optimization 改造。

---

### D. 只在 bbox 边界附近读更高一层分辨率，做局部细化

这是第三个高优先级实验。

**方法原理**

不要上来就 level0 全图。只对当前 box 四条边附近各取一个 narrow strip，在更高一层分辨率上重新计算简单 cue：

- local residual / optical density
- edge / entropy
- 与 rival core 的距离

然后只允许边界在一个小范围内 inward / outward search。

**为什么适合**

因为你现在的瓶颈主要在边界，而不是全图。这种 selective zoom 最有可能以很小代价修掉：

- 薄而浅的外缘被 overview 吃掉
- middle / lower footprint 不够完整
- 某一侧其实已经碰到邻片，但 overview 看不清

**代价**

中等。但仍然和读取后端解耦：只要能按坐标读小 strip，就能做。

---

## 2）更强的目标选择机制

### E. 把同 slide 其他 candidate / GT 当成 hard negative，这件事非常值得做

结论很明确：**值得，而且优先级很高。**

原因不是“多一点负样本总归好”，而是因为 **这些 same-slide neighbors 才是你的真实负样本分布**。空背景是太容易的负样本，帮不了你解决当前错误。

可以这样用：

#### 在 classical inference 里
- weak support 只能分配给最近且最兼容的 core
- box 不能轻易越过 target core 与 neighbor core 的中垂线，除非 target evidence 明显更强
- 如果某一侧 projection 出现多峰，第二峰对应方向直接提高 leak penalty

#### 在 learning 里
- 针对每个 target，专门生成“扩进最近邻”的 bad proposals
- 用这些 bad proposals 训练一个 ranker 或 side-offset regressor
- 训练损失里直接加 `overlap with other GT masks`

这是当前最缺、但信息量最大的约束。

---

### F. 学一个轻量 box-delta regressor，比直接上 detector 更划算

这是我最推荐的第一条学习路线。

**方法原理**

保留你现有 pipeline，先出一个 baseline seed / box，再从这些 feature 预测四条边的 offset：

- seed 大小、长宽比、偏心程度
- per-side cumulative target-support curve
- per-side rival-support curve
- 最近邻距离、角度、同排密度
- entropy / edge / optical density 的边界统计
- 当前 box 与 pseudo-mask tight box 的差

模型可以非常轻：

- gradient boosting（梯度提升树）
- CatBoost / XGBoost
- 小型 multilayer perceptron（多层感知机）

**为什么适合**

- 你已经有 GT mask、手工 crop、proposal 全部映射到统一 level0 坐标
- 这个任务更像 **structured regression**，不一定需要完整 detector
- 样本量不算巨大时，树模型往往比 detector 更稳、更可解释

**代价**

低到中等。是 classical 到 learned 的最好过渡层。

**关键建议**

不要只用手工 crop 当监督目标。你完全可以先离线求一个 **oracle optimal box**：

```math
B_k^*=\arg\max_B\; 0.5\,Cover_k(B)-0.3\,Leak_k(B)-0.2\,Excess_k(B)
```

甚至再加 `coverage >= r_min` 的 hard constraint。然后用 \(B_k^*\) 作为 regressor 的监督目标。这样学到的是“你的任务目标”，不是“标注者习惯”。

---

## 3）更强的候选裁切框优化机制

### G. 先做 “oracle box” 计算，再谈新方法好不好

这一步非常值。

你现在有：

- target GT mask
- same-slide other-section masks
- manual crop
- proposal

所以你其实可以离线求 **这个任务下最优的 axis-aligned bbox 上限**。这有三个用处：

1. 看当前 baseline 距离 task-optimal 还有多远  
2. 看 manual crop 到底是 reference，还是其实本身就偏松  
3. 给后续 regressor / detector 提供干净监督目标

**这一步能直接回答一个关键问题**：
如果 oracle 比当前 `hybrid_topfloor55_wide24` 只好一点点，那继续堆方法的收益很有限；如果 oracle 好很多，说明你现在的瓶颈不是 box definition 上限，而是 proposal mechanism 本身。

---

## 4）轻量学习 / 弱监督 / object detection 路线，哪些值得试

### H. 我更推荐 “overview support-map model”，而不是一上来直接 detector

这是第二条更强路线。

**方法原理**

在 overview 或 candidate-level patch 上训练一个小模型，输出：

- target-support probability
- boundary probability
- center heatmap

再用 boundary + center 做 instance split，最后出 bbox。

这不是为了替代你后面的 tissue mask extraction，而是为了让 **bbox proposal 本身有一个更好的 instance-aware support map**。

**为什么适合**

- 你的难点主要在 **弱边缘 + 邻片分离**，这比“直接回归一个 box”更接近真实问题
- 你已经有 GT masks，可以自动生成 downsampled training target，不需要额外标注
- inference 仍可完全在 overview 级完成，必要时再加边界 strip refinement

**代价**

中到高。但这是比 detector 更对题的 learned route。

---

### I. 直接 object detector 可以试，但不是我第一优先级

**优点**
- 部署直接
- 输出就是 bbox
- 现有 GT 可直接转 box label

**缺点**
- 你的任务痛点不是“找一个 box”，而是“box 是否完整且不吞邻片”
- detector 的标准 box loss 更偏交并比，不天然等价于你的 50/30/20 目标
- 近邻、弱边缘、多实例拥挤时，detector 很容易学到不稳定边界

所以如果要试 detector，我会把它放在 **support-map model 之后**，或至少和 box-delta regressor 并行小规模试，而不是先做。

---

### J. 纯弱监督路线优先级不高

但对你这个任务，**你已经有比 slide-level label 强得多的 supervision**：

- instance GT mask
- manual crop
- proposal history
- same-slide neighbor masks

这时再刻意降到弱监督，通常不划算。弱监督只在一种情况下值得优先：**你打算扩到很多没有 mask/box GT 的 stain 或新批次 slide**。

---

## 对你特别关心的几个问题，直接给结论

### 当 bbox 过大时，通常是哪些信息在误导 proposal？

最常见的四类误导：

1. **neighbor 的 deep myelin core** 在同方向 projection 里形成第二峰  
2. **merged weak support** 让 target 与 neighbor 在 overview 上看成一个 object  
3. **宽松的 nonwhite / legacy cue** 把背景阴影、折痕、局部非白玻片也当成 tissue-like  
4. **用 box 扩张去补 seed 不完整**，结果变成“哪里缺往哪里放大”，而不是“这部分是不是属于 target”

---

### 当 bbox 仍然截断目标 section 时，通常是什么信息没有被利用？

最缺的通常不是更多 dark-signal，而是这些信息：

1. **浅染但连续的 outer fringe**  
2. **弱纹理 / 弱边缘的空间连续性**  
3. **与 target core 的连通关系**  
4. **边界附近更高一级分辨率的局部证据**  
5. **同 slide 其他 core 提供的归属竞争关系**

---

### 如何显式利用“非目标 section 惩罚”？

最直接的方式不是改评估，而是改生成：

- 所有强 core 先同时检出
- 用竞争式分配得到每个 section 的 pseudo-mask
- box edge 每走一步，都计算：
  - target coverage gain
  - rival overlap penalty
  - excess area penalty
- 只有净收益为正才继续扩张

这个改造一旦做了，你的 inference 才真正和评估逻辑对齐。

---

### 是否值得把同一张 slide 上其他已知 candidate / GT 位置作为负样本或几何约束？

**非常值得。**
这是当前最该利用而尚未充分利用的信息。

在训练里，它们是最有价值的 **hard negatives**；在推理里，它们是最有价值的 **geometry prior**。

我甚至会说：对你这个任务，**same-slide neighbor information 的价值，高于再发明一个新阈值函数。**

---

## 一个可执行的后续实验路线图

先加一句：在做任何新模型前，我会先补 3 个分析量，做 failure stratification：

- **isolation score**：target core 到最近 neighbor core 的相对距离
- **completeness risk**：weak support 中落在当前 box 外的比例
- **projection multimodality**：每个方向 projection 的第二峰 / 第一峰比值

这三个量会直接告诉你：错误更多来自 **近邻混淆**、**seed 不完整**，还是 **overview 边界分辨不足**。

### 实验计划表

| 顺序 | 路线 | 需要输入 | 预期解决的问题 | 如何接入你现有 GT 评估 | 重点看什么 | 什么结果算真的更好 |
|---|---|---|---|---|---|---|
| 0 | **Oracle optimal bbox** | target GT mask + same-slide neighbor masks | 判定当前方法离任务上限多远；给后续学习方法提供监督目标 | 直接在 level0 上用你的三项指标搜索最优轴对齐 box | 加权分数、coverage floor、neighbor leak、compactness | 明确 oracle gap；若 gap 很大，说明仍有显著 headroom |
| 1 | **Strong-core / weak-fringe + 多种子竞争分配** | overview + 现有 score map + 新增 OD/entropy/edge cue | 主要打掉吞邻片，同时补回一部分弱外缘 | 输出 bbox 后照常用现有 level0 评估 | mean / 5th percentile coverage；mean / 95th percentile neighbor overlap | 在 coverage 基本不掉的前提下，neighbor overlap 明显下降；或 tail recall 改善 |
| 2 | **Competitor-aware box optimization** | 实验1得到的 pseudo-mask 或当前 candidate map | 让 inference 目标和 50/30/20 目标对齐；减少无意义大 box | 完全沿用现有 GT 评估 | joint pass rate；P95 leak；P95 area ratio | 相比 baseline，joint pass rate 提升，并且 leak tail 明显收缩 |
| 3 | **边界 strip 的局部高分辨率 refinement** | 当前 bbox + 读取小 strip 的能力 | 专门修截断，尤其是 middle/lower 弱边界 | 输出 refine 后 bbox 进入现有评估 | coverage failure rate；lower-5% recall | recall tail 提升，而 neighbor overlap 不明显恶化 |
| 4 | **轻量 box-delta regressor** | 当前 pipeline features + oracle / GT target box | 学习 case-adaptive per-side offset；减少手调规则 | 直接预测 box，映射回 level0 评估 | 综合分数 + hard-case subset | 在 held-out slides 上稳定优于 classical baseline，尤其是近邻 hard case |
| 5 | **Overview support-map model** | overview 图 + downsampled GT instance masks / boundaries / centers | 从源头改善 target support estimation 和 instance separation | 先出 support，再出 bbox，进入现有评估 | 三项主指标 + tail metrics + oracle gap closure | 同时改善 recall tail 和 neighbor leak，且跨 slide 稳定 |

---

## 哪些指标最该补

你现有三项指标要保留，但我会再加四项：

1. **coverage failure rate**：例如 recall 低于某个 floor 的比例  
2. **neighbor-overlap 95th percentile**  
3. **joint pass rate**：同时满足 coverage floor、neighbor leak 上限、area 上限的比例  
4. **oracle gap closure**：新方法相对 oracle 关掉了多少 gap

原因很简单：你现在已经不缺平均值，缺的是 **hard case 可控性**。

---

## 当前最优方案该不该继续沿用

### 结论
**继续沿用 `hybrid_topfloor55_wide24` 作为默认 baseline 和当前生产默认值。**

原因：

- 它在你现有 50/30/20 加权下确实是最好的综合折中
- 平均 coverage 已经很高
- 平均 compactness 也不差

但不建议再把主要精力放在它的 global threshold 或固定扩张参数微调上。这条路大概率已经进入收益递减区。

---

## 什么时候切到更保守 / 更激进的方案

### 更保守
当出现这些征象时，应偏保守：

- 最近邻 section 很近
- weak support 与邻片几乎接触
- 某侧 projection 明显多峰
- target core 靠近当前 box 边界，但 box 外立刻出现 rival core

这类 case 的主风险是 **leak 到邻片**。

### 更激进
当出现这些征象时，应偏激进：

- target 很孤立，最近邻远
- strong core 明显小于 expected footprint
- weak fringe 大量落在当前 box 外
- projection 近似单峰，但边缘宽而弱

这类 case 的主风险是 **截断 target**。

### 现实建议
在新的 competitor-aware 方法出来前，如果今天就要上线一个切换逻辑，我会用：

- `hybrid_topfloor55_wide24` 做默认
- 对 **高 isolation + 高 incompleteness** 的 case，切到较激进的 seed-relaxed 备选
- 对 **低 isolation + 高 multimodality** 的 case，坚持保守，宁可少放大，也不要吞邻片

你给的两个备选里，我会优先把 seed-relaxed 系列当作 **isolated undercoverage fallback**，而不是全局默认。

---

## 如果只能优先做 3 个新实验，我推荐这 3 个

### 1. Strong-core / weak-fringe + 多种子竞争分配
这是我认为 **性价比最高** 的实验。它直接测试你的主要瓶颈是不是 **instance separation** 而不是单纯 signal strength。

### 2. Competitor-aware box optimization
这是把你现在“评估逻辑是对的、生成逻辑还没跟上”的缺口补上。如果这一步有效，说明你不是缺 feature，而是缺 objective alignment。

### 3. 边界 strip 的局部高分辨率 refinement
这是最小代价验证“是否确实受 overview 分辨率限制”的办法。如果它能明显拉升 recall tail，说明之后值得继续往 multi-scale 方向走；如果没效果，说明主问题还在 assignment / optimization。

---

最简洁的判断是：

**当前最优方案可以继续当 baseline，但下一阶段不要再把重心放在“更好的单一 tissue score + 更宽的扩张”。最值得做的是：**
1. **把弱信号变成 target-conditioned support**
2. **把 same-slide neighbors 变成显式 negative**
3. **只在 bbox 边界局部引入更高分辨率证据**

这三件事最可能在不大改架构的情况下，把你现在剩下的 hard