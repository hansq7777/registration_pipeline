# Whole-slide histology brain section 的 bbox proposal 失效机理与高优先级改进路线

## 问题复述与成功标准

你在做 entity["company","Hamamatsu Photonics","photonics company, jp"] NanoZoomer 扫描的 whole-slide histology（重点是 Gallyas / myelin）自动裁切：对每张 slide 里的每个 brain section，先在低分辨率 overview 上生成一个 bbox proposal，然后在该 crop 上再做 tissue mask。你已经构建了一个“统一映射回 level0 坐标系”的评估体系：对每个 section 的 proposal bbox，在 level0 空间评价三件事——(i) 是否覆盖目标 section 的 GT mask（优先级最高），(ii) 是否与同 slide 的其他 section GT mask 重叠（次高），(iii) bbox 是否过度膨胀、浪费空间（最低）。你把这三者按 50% / 30% / 20% 进行加权，并做了 GT 驱动的策略搜索。

当前最优策略是 **hybrid_topfloor55_wide24**，其均值指标（你给出的粗略数字）大致为：target-mask coverage recall ≈ 0.9936、neighbor-overlap ratio ≈ 0.1072、proposal-area / GT-crop-area ≈ 0.8841。它相较旧版本已经显著提高 coverage，但仍有一批 hard case：要么 bbox 仍然截断目标组织（不是单纯 dorsal/top 漏掉，而是中下部 footprint 也不理想），要么 bbox 过大把邻近的 section 带进来。你希望从“信号选择 / 几何扩张 / 候选框定义 / 目标-非目标区分 / 低分辨率信息不足”等角度系统拆解原因，并给出按优先级排序、可快速验证的 bbox proposal 改进方案（重点是 bbox，不是后续 mask 提取）。此外你特别关心如何显式引入“非目标 section 惩罚”、以及是否值得把同 slide 的其他 candidate / 已知位置当作负样本或几何约束。

这一类设计在数字病理中非常典型：许多 whole-slide 工作流会在低分辨率 overview/thumbnail 上先做 tissue detection（以节省计算、避免在空背景上扫描/处理），但低分辨率的组织分割在“弱对比、染色不均、伪影多、组织彼此靠近”时会失败。citeturn0search7turn0search0

## 现有 pipeline 的关键结构与误差传播路径

你当前方法的核心是：**overview 上构造一个“组织候选强度图”→ 连通域 → seed box → 扩张（固定比例 + 信号投影方向扩张）→ 合并得到最终 bbox**。在 myelin/Gallyas 上，你用 `residual = blurred_background - gray` 捕获深染区域，同时保留 legacy 的 `max(inv_gray, nonwhite)`，最后用 `hybrid = max(residual, legacy)` 做候选。

这类“thumbnail-level（低分辨率） + classical rules”的组织检测通常在速度与工程可控性上很强，但它有一个结构性弱点：**任何在强度图上“漏掉的组织边界”与“引入的伪阳性”都会被后续的几何扩张放大**。这并不是你实现有问题，而是算法形态决定的误差传播：候选图偏向深染区域→ seed 本来就不完整→ 方向扩张依赖投影信号，遇到弱边界会停、遇到邻片强信号会冲。文献对这点的共识是：经典阈值/聚类在“染色变化、亮度不均、伪影”下鲁棒性不足，往往需要更强的局部自适应、后处理与（或）学习式分割来补足。citeturn0search0turn2search3turn0search7

对 Gallyas / myelin 场景，你还额外遇到两个物理现实约束：

- **染色强度本就可能不一致**：Gallyas 银染（及其改良）强调髓鞘/神经纤维的显色，但实际操作中“过浅/偏淡”的情况需要延长显影或重复显影，意味着你在数据层面确实可能遇到“边界或浅染区信号弱到接近背景”的 section。citeturn0search5turn0search9turn0search1  
- **低分辨率 overview 的信息上限**：下采样会把“薄边界、浅染外轮廓、细碎结构”平均掉；scanner/软件常在 overview 上做初步 tissue segmentation，但也承认 overview 分割可能漏检组织区域。citeturn0search7turn0search0

因此，当前表现“均值很好但仍有 hard case”并不意外；你真正要解决的是：**让错误主要集中在均值，而不是尾部（tail）**。这里有个逻辑风险需要点名：你现在报告的多是 mean 指标，但裁切流水线里最致命的是 **P1/P5（最差 1%/5%）** 或 “失败率（coverage < 某阈值 或 neighbor-overlap > 某阈值）”。只盯均值会系统性低估 hard case 的工程危害。

## 目标组织被截断时通常缺了什么信息

把“截断”按你 pipeline 的信息流拆开，根因常见落点是：**目标组织的外轮廓/浅染部分没有在 overview 强度图里形成足够连续、可投影的证据**。这会造成两类具体后果：

第一类是 **候选连通域本体不全**  
`hybrid = max(residual, legacy)` 的设计本质是“宁可把深染抓住”。但 Gallyas/myelin 的浅染外圈、某些层状区域、或边缘淡染，可能同时对 `residual` 与 `inv_gray/nonwhite` 都不占优势，于是被阈值/连通域切掉，seed box 的几何包围在一开始就偏小。Gallyas 相关方法学本身就强调“理想情况下髓鞘应显著深于背景；若太淡需再显影”，这等价于告诉你：数据里确实会出现对比度不足、难用全局强度规则捕获的组织部分。citeturn0search5turn0search9turn0search1

第二类是 **方向性扩张“看不见该扩的方向”**  
方向扩张基于 overview 信号投影，本质是用 1D 统计量估计“还有没有组织”。当外轮廓信号弱且不连续时，投影曲线会过早回落，扩张停止，于是形成你描述的“middle/lower footprint 仍不理想”。这在低分辨率下更明显，因为边界像素被平均后对投影贡献进一步变小。citeturn0search7

第三类是 **没有利用“边界证据（edge/gradient）”**  
你当前信号主要来自强度残差与非白度，属于“区域证据”；但组织外轮廓在很多 bright-field 染色中更稳定的线索往往是 **梯度/边缘能量**（即便内部纹理/强度不稳定）。经典分割里常用基于梯度的 watershed/marker-based 方法，把高梯度作为分割屏障、低梯度作为区域内部；这类思路至少能提供一个“边界在哪里可能存在”的几何约束，而不仅靠强度阈值。citeturn1search4turn1search5turn1search1

第四类是 **缺少局部自适应对比度（local adaptivity）**  
全局的 `blurred_background - gray` 相当于一种大尺度背景校正，但当组织边缘处的背景/照明/玻片非均匀性与组织弱信号叠加时，全局阈值与单一尺度的背景估计会不稳。经典文档/图像二值化领域大量工作表明：局部自适应阈值（例如 Sauvola 系列或其快速实现）能在低对比、亮度不均时比全局阈值更稳，并且可以用 integral image 把计算加速到接近全局阈值的复杂度。citeturn2search4turn2search8turn1search10

## bbox 过大并覆盖邻片时通常被什么误导

“过大”在你的 pipeline 中几乎总是以下几种误导叠加：

第一类是 **投影扩张被邻片强信号牵引**  
当相邻 section 在投影方向上几乎同一条扫描线（例如上下排列、纵向投影），只要邻片比目标片更深染，投影曲线就会在 gap 之后再次升高，方向扩张机制会把这解释成“目标组织仍在延伸”，于是跨过空隙继续扩。你把 `hybrid` 定义为 `max(residual, legacy)` 也会加重这一点：只要任一分量在邻片上强，max 就会把邻片当作组织延伸的证据。

第二类是 **候选区域在二值化/连通域阶段发生“桥接”**  
overview 的组织候选图在阈值化后，如果做过形态学 closing/dilation（哪怕是隐式的，例如模糊+阈值就可能把窄 gap 填平），两个相近 section 很容易在二值 mask 上连成一个连通域，seed 就变成“两个 section 的 union”，后续 bbox 再怎么优化都会大。分割领域解决“粘连/接触物体”的标准套路之一是：对二值 mask 做 distance transform，取局部极大值作 marker，然后做 marker-based watershed，把粘连区域沿“距离鞍点”切开。citeturn1search4turn1search16turn1search24turn1search5

第三类是 **伪影与背景纹理在 myelin 染色上很容易变成假组织**  
银染/髓鞘相关流程容易出现沉积、局部高密度颗粒、边缘脏污等，这些在低分辨率下会呈现为“非白且偏暗”的块状结构，恰好命中 `inv_gray/nonwhite` 与 residual 的偏好，从而把 bbox 拉向错误方向。银染标准化研究也强调显影/清洗条件对结果曲线影响显著；这意味着你的 overview 信号分布在不同批次间可能存在系统差异。citeturn0search9turn0search5

第四类是 **固定比例扩张在 section 间距可变时必然失配**  
你已经观察到 mount 不规整、邻片距离可变。固定比例扩张在“邻片很近”的样本上必然更危险，因为它等价于在 gap 上随机下注：扩张尺度一旦超过 gap，就会吃到邻片。这个问题不靠调一个全局比例就能根治，只能通过“局部自适应停止条件”或“显式邻片惩罚/约束”来解决。

## 改进方法清单与优先级

下述方案按“最快能验证、最可能直接降低 hard case”的优先级排序，并按你希望的四类（经典改进 / 更强目标选择 / 更强 bbox 优化 / 轻量学习）组织。每个方案都给出原理、适配原因与代价。

### 经典 pipeline 上的高收益改动

最优先的是 **把你的 bbox 生成从“seed→扩张”改成“先得到实例级粗 mask→再算 bbox”，并在 mask 阶段做粘连处理**。原因是：你现在所有 hard case（截断与邻片覆盖）都可以追溯到“候选图不可靠 + 几何扩张放大误差”，最短路径是让候选图变成“更接近实例 mask”的东西。

**局部自适应阈值补弱信号**  
原理：在 overview 上对灰度或某个颜色通道做局部阈值（Sauvola/Bradley/改良实现），生成更连续的组织候选，再与 residual/legacy 融合（建议从加权和或 softmax-like 融合开始，而不是 max，以免单一分量劫持）。局部阈值可以用 integral image 加速，使得窗口大小不显著增加计算量。citeturn2search4turn2search8turn1search21turn1search10  
适配原因：它专攻“低对比、照明不均、浅染边界断裂”，对应你的截断问题。citeturn2search4turn0search7  
代价：需要新增参数（窗口大小、k/偏置等），但这些参数通常能通过少量 GT 做 grid search；计算仍非常轻量。citeturn2search4turn1search10

**distance transform + marker-based watershed 做邻片分离**  
原理：先得到二值组织区域（哪怕粗糙），对其做 distance transform，在距离图上找局部峰值作为 marker，再做 watershed 切分粘连连通域。经典 watershed 以“把像素值视作地形、从 marker 灌水直到相遇”为直观解释；实现上 entity["organization","scikit-image","python image processing lib"] 与 entity["organization","OpenCV","computer vision library"] 都提供了标准实现范式。citeturn1search4turn1search5turn1search1turn1search24turn1search16  
适配原因：它直接瞄准你最痛的“bbox 过大把邻片带进来”，因为在 bbox 之前把实例分开，后续 bbox 只需围绕单实例。citeturn1search16turn1search4  
代价：marker 选取不稳会导致过分割或欠分割；但你这里只需要“把明显相邻的 section 分开”，可以用面积/形状先验与合并规则（例如过小区域合并回最近大区域）降低过分割风险。对 brain section 这种大而连贯的对象，distance-watershed 往往比在细碎对象上更稳。citeturn1search16turn1search8

**边界能量辅助的“停止条件”替代纯投影阈值**  
原理：在候选 bbox 的扩张过程中，不只看投影强度是否还“有组织”，还看边界处是否出现“稳定的高梯度带”（可能是组织外轮廓）。这可以通过 gradient magnitude 或形态学梯度近似。marker-based watershed 的经典用法也强调“高梯度作为屏障”。citeturn1search1turn1search5turn1search4  
适配原因：你的 weak edge 被强度方法漏掉时，梯度往往仍有信号，能减少截断。  
代价：在玻片边缘、划痕、灰尘处也可能有梯度伪影，所以要和“连贯性/形状先验”联合使用。

### 更强的目标选择机制

你当前“每个 section 独立出 bbox”的策略，在 multi-section slide 上天然吃亏，因为它没有利用“同一张 slide 上实例之间互斥”的信息。更强的做法是：**先在 slide 级别检测出所有 section 实例，再为每个实例分配 bbox**——这从问题定义上就更贴合你的 30% 邻片惩罚项。

**slide-level 实例集合先验（joint detection + assignment）**  
原理：在 overview 上先得到所有候选实例（连通域/聚类/分水岭实例），再对每个实例生成 bbox。然后用一个全局规则保证：任意两个实例的 bbox 不交叠或交叠受限（soft/hard）。  
适配原因：你的邻片距离可变但“排列有规律”，这类弱几何先验非常适合做全局 assignment（例如按质心做行列聚类，再在每行内按 x 排序），即便不引入强刚性模板也能提升稳定性。  
代价：工程上要从“单实例 bbox 提案器”升级为“多实例分解器 + per-instance bbox”；但这一步一旦做了，后续很多问题会更容易处理。

**显式使用“非目标 section 惩罚”的两段式决策**  
原理：把流程改为：  
1) 为同一张 slide 先产出一个“所有 section 的粗实例 mask 集合”（哪怕不完美）；  
2) 对每个目标实例做 bbox 优化时，将“其他实例的 mask”作为显式负项（penalty）或硬约束（constraint）。  
适配原因：这正面回答你提出的关键问题——“如何显式利用非目标 section 惩罚”。它会把“过大包含邻片”的问题从后验评估变成前验搜索目标。  
代价：需要你在 overview 上维护一个实例集合（labels），并确保 label 稳定（可以用简单的合并/过滤规则保证）。

### 更强的候选裁切框优化机制

这部分是我最建议你立刻上的，因为它能把你现有 GT 评价函数“直接变成优化目标”，并且计算代价可控。

**把 bbox 选择写成可搜索的目标函数，并用 integral image 加速**  
原理：定义一个 bbox objective（与你 50/30/20 一致），例如在 overview 空间：  
- 目标项：bbox 内目标实例的置信度/像素质量（proxy mask 的像素和）越大越好；  
- 负项：bbox 与非目标实例集合的重叠（像素和）越小越好；  
- 正则：bbox 面积或相对 seed 的扩张量越小越好。  

关键点是：如果你把“目标/非目标的像素权重图”做成 **summed-area table / integral image**，那么任意矩形区域的像素和可以 O(1) 计算，从而可以做非常快的坐标下降、局部网格搜索或 beam search。integral image 的“任意矩形求和只需常数次数组访问”是经典结论，常被用于快速矩形特征计算。citeturn1search21turn1search10turn1search2  
适配原因：你已经有 GT 驱动的评分框架；现在欠缺的是把它用于 **proposal 生成阶段**，而不是仅用于策略选择阶段。该方法能直接压低“bbox 过大引入邻片”，同时不会牺牲 coverage（因为 coverage 项仍在目标函数里）。  
代价：需要你先构造一个 proxy 的“实例级目标 mask 与非目标 mask”。这可以来自 watershed/连通域，也可以来自你现有 hybrid 图阈值化后的实例分割。工程复杂度中等，但不需要训练模型。

**冲突消解（conflict resolution）作为最后一道闸**  
原理：即便每个 bbox 都局部最优，仍可能出现两两 bbox 冲突。可以在最后加一层冲突消解：若 bbox_i 与 bbox_j 重叠超过阈值，则沿着两实例质心的垂直平分线方向收缩，或对重叠区域执行“归属竞争”（把重叠像素归给更像目标的实例）。这相当于在 bbox 级别近似 Voronoi 划分。  
适配原因：你的目标明确要求“尽量避免覆盖其他 section”，因此任何后验冲突消解都会带来确定性收益。  
代价：需要定义冲突阈值与收缩策略；但可用 GT 数据把阈值选在硬约束风格（例如 neighbor-overlap ratio 必须 < 0.05）。

### 轻量学习与弱监督路线

当 classical 方案把“结构性错误”解决后，学习法的价值主要在于：**让弱信号边界在 overview 上更可见**，减少截断；并且在跨批次/跨扫描条件下更稳定。数字病理中已有大量深度学习框架用于 WSI 预处理与组织区域分割，通常比纯阈值法更鲁棒，但代价是需要标注/训练与监控漂移。citeturn0search8turn0search4turn0search0

**训练一个 overview-level 轻量 tissue/section segmentation 网络（弱监督也可）**  
原理：你已经有 level0 对齐的 GT mask，可以下采样到 overview 分辨率，训练一个小 U-Net（或更轻的 encoder-decoder）输出组织概率图；或直接借鉴现成 WSI 质控/组织分割网络（例如 UNet++ 变体），只在你的 stain/扫描域上 finetune。GrandQC 体系就属于“组织区域 + 常见伪影”的深度学习分割思路。citeturn2search7turn2search3turn0search0  
适配原因：它对“浅染边缘、背景非均匀、伪影多样”往往比全局强度规则更稳，从根上减少截断与假阳性。citeturn0search0turn0search7  
代价：需要维护训练集/验证集划分、避免过拟合特定批次；也需要监控 domain shift。工程与实验成本高于 classical，但仍可做得很轻（只在 overview 上推理）。citeturn0search8turn2search3

## 可执行的后续实验路线图与决策建议

下面给出一条“快速 classical 验证 → 更强方法探索”的路线图。每条路线都说明要解决什么、需要什么输入、如何接入你现有 GT 评估、看哪些指标，以及什么结果算真正赢过当前最优策略。

### 快速验证路线

**路线一：实例分离优先的 classical 升级（推荐先做）**  
要解决的问题：bbox 过大引入邻片（主攻 30% 项），同时不让 coverage 回退（守住 50% 项）。  
需要输入：overview 图；你现有 hybrid score；（新增）阈值化与形态学后处理参数。  
方法步骤：  
1) 在 overview 上生成二值候选组织 mask：从 `hybrid` 或 `residual + local-adaptive` 得到（建议做一次小型 ablation：现有 hybrid vs 加入局部阈值）。citeturn2search4turn2search8turn0search0  
2) 对该二值 mask 做 distance transform + marker-based watershed 分离粘连实例，得到 instance labels。citeturn1search4turn1search16turn1search5turn1search24  
3) 对每个 instance 直接取 tight bbox，再加一个很小的安全 margin（margin 可以是分辨率相关、或按 instance 尺寸的百分比）。  
4) 若某 bbox 与其他 instance labels 发生明显重叠，触发冲突消解（例如按重叠区域收缩）。  

如何结合现有 GT 评估：完全不改你的评估框架，只把新 proposal bbox 映射回 level0，计算你已有三指标。  
建议新增的关键指标（强烈建议加入）：  
- coverage recall 的 **P5 / min**（而不仅是 mean）  
- neighbor-overlap ratio 的 **P95 / max**  
- “失败率”：coverage < 0.99 的样本比例；neighbor-overlap > 0.05 或 >0.1 的样本比例（阈值可由你当前 mean 反推更严格的目标）。  

什么结果算更好：在不降低 mean coverage 的前提下，使 neighbor-overlap 的 P95/max 明显下降，同时 coverage 的最差尾部（P5）不变或改善。你现在 mean coverage 已经接近 0.994，继续抬均值意义不大；最关键是压尾部。  
风险点与应对：watershed 可能过分割；用“面积阈值 + 合并到最近大实例”的规则即可缓解。citeturn1search8turn1search4turn1search16

**路线二：显式非目标惩罚的 bbox 优化器（推荐与路线一并行）**  
要解决的问题：把你的 50/30/20 评价变成 proposal 生成阶段的优化目标，系统性压低“过大/引邻片”。  
需要输入：overview 上的目标实例 mask（可来自路线一的 labels）；非目标实例 mask（同一张 slide 的其他 labels）；或至少它们的概率图。  
方法步骤：  
1) 预计算三个 integral images：  
   - T(x,y)：目标实例像素权重图（binary 或 soft）  
   - N(x,y)：非目标实例像素权重图（union of others，binary 或 soft）  
   - A(x,y)：常数 1（用于快速求 bbox 面积）  
2) 定义 bbox 得分：`S = 0.5 * sum_T_in_bbox - 0.3 * sum_N_in_bbox - 0.2 * area_bbox`（你可以把 0.2 项改成相对 seed 的扩张量或相对 tightbbox 的面积增量，更贴合“尽量不浪费空间”）。  
3) 用坐标下降/局部搜索：从 tight bbox 开始，四条边每次向外/向内移动若干像素，选择能提升 S 的动作，直到收敛。  
integral image 能保证每次评估矩形求和是 O(1)，使得搜索非常快。citeturn1search21turn1search10turn1search2  

如何结合现有 GT 评估：同上，输出 bbox 映射回 level0，跑同一套评估；另外你可以把现有加权 score 与优化器目标做一致化，减少“训练目标”和“评估目标”不一致导致的策略偏移。

什么结果算更好：neighbor-overlap 明显下降，且“bbox 过大”样本（proposal-area / GT-crop-area 远大于 1 或者远大于某阈值）显著减少，同时 coverage 的尾部不变或变好。  
风险点与应对：如果 instance mask 本身错了（例如目标和邻片在 mask 上已粘连），优化器无法凭空分离。因此它需要路线一那种实例分离做前置，二者是互补关系而不是替代关系。

**路线三：局部自适应阈值补边界（优先级略低但仍建议做）**  
要解决的问题：截断（主攻 50% 项的尾部）。  
需要输入：overview 灰度/颜色通道；阈值窗口大小等参数。  
方法步骤：在 overview 上用快速局部自适应阈值生成一个“弱边界补全图”，与现有 `hybrid` 融合；再走你现有的连通域/seed/扩张或走路线一的实例分离。局部自适应的快速实现可基于 integral images。citeturn2search4turn2search8turn1search10  
什么结果算更好：coverage recall 的 P5/min 上升，且不会显著抬高 neighbor-overlap（若抬高，说明补全图把 gap/伪影也补进来了，需要加上“非目标惩罚”或更严格的形态学过滤）。

### 更强方法探索路线

**路线四：overview-level 轻量学习分割 + classical 实例/优化（建议作为第二阶段）**  
要解决的问题：跨批次/跨染色强度变化导致的弱边界缺失与伪影误检，进一步压低 hard case。  
需要输入：你已有的 GT mask（下采样）；训练/验证划分；推理时只需要 overview。  
方法步骤：训练一个轻量分割模型输出 tissue/section 概率图；随后仍用路线一的实例分离与路线二的 bbox 优化器（关键是保留“非目标惩罚”结构）。数字病理里已有组织分割与质量控制的深度学习体系（如 UNet++ 变体的 GrandQC 思路）验证了这种路线的可行性。citeturn2search7turn2search3turn0search8  
什么结果算更好：在不增加过多工程复杂度的前提下，显著改善“浅染边界导致截断”的尾部，同时对邻片覆盖不变或更好。  
代价：训练与域漂移监控成本；但因为只在 overview 推理，算力成本可控。

**路线五：把问题转成 bounding-box/instance detection（可选，只建议做 1 条原型）**  
要解决的问题：直接输出每个 section 的 bbox（或 mask），绕过大量手工规则。  
适配原因：你已有 GT bbox/GT mask 与 level0 映射，天然可以生成监督信号。深度学习 WSI 分析综述也指出：在 patch/thumbnail 层面做检测与分割是常见范式。citeturn0search8turn0search4  
代价：标注量需求更高；模型调参、失败模式更难解释；对于你这种“必须强约束邻片覆盖”的任务，纯检测往往还要加后处理约束（最终仍会回到类似路线二的冲突消解/约束优化）。

## 明确建议与可参考的解决方案列表

**是否继续沿用当前最优方案**  
hybrid_topfloor55_wide24 作为默认 baseline 可以继续沿用，但更合理的定位是：它是“强 coverage 的粗 proposal”，而不是最终形态。你现在的 mean coverage 已接近满分，继续靠调 topfloor/扩张参数榨取小数点后收益，边际收益会很快变低；同时你已明确看到过大与邻片引入问题，这通常不是再调一个全局阈值能根治的。

**何时切换到更保守 or 更激进**  
建议从“规则切换”升级为“风险感知切换”（只用 overview 就能算出的风险指标），例如：  
- 若检测到 **最近两实例的 gap 很小** 或 **目标实例外接框到邻实例外接框距离 < 某阈值**：切到更保守（更强非目标惩罚、更小固定扩张、或直接启用冲突消解）。  
- 若检测到 **目标实例的候选 mask 很碎（多连通域）** 或 **候选强度图的边界置信度低**：切到更激进（启用局部自适应补边界、或允许更大 margin，但必须同时启用非目标惩罚来防邻片）。  
这类风险指标与“同 slide 多实例互斥”天然相容，且能直接服务你的 50/30/20 优先级。

**如果只能优先做 3 个新实验，最推荐的 3 个**  

- **实验一：distance transform + marker-based watershed 的实例分离（路线一的核心）**  
  这是最直接、最快把邻片问题从 bbox 阶段前移到实例阶段解决的方法；其理论与实现都成熟，且专门针对“接触/粘连对象分离”。citeturn1search4turn1search16turn1search5  

- **实验二：引入“非目标 section 惩罚”的 bbox 优化器（路线二的核心）**  
  这是把你的评价体系变成生成体系的关键一跃；借助 integral image 可以做到计算极快，工程上也能与现有 GT 框架无缝对接。citeturn1search21turn1search10turn1search2  

- **实验三：局部自适应阈值补边界（路线三的核心）**  
  这是最可能改善“浅染外轮廓导致的截断”的 classical 手段之一，并且存在成熟的快速实现（integral image 加速）。citeturn2search4turn2search8turn1search10  

**一个可直接照着落地的“组合方案”**（把上面三项串起来）  
1) overview：`hybrid` + 局部自适应阈值 → 得到组织候选二值图（可带 soft 权重）。citeturn2search4turn0search0  
2) 实例分离：distance transform markers + watershed → 得到每个 section 的实例标签。citeturn1search4turn1search16turn1search24  
3) bbox 初值：每个实例 tight bbox + 小 margin。  
4) bbox 精炼：用 integral image 的目标函数优化（含非目标惩罚），并做最终冲突消解。citeturn1search21turn1search10  
5) 评估：除现有 mean 三指标外，强制报告 coverage 的 P5/min、neighbor-overlap 的 P95/max、以及失败率（coverage < 0.99、neighbor-overlap > 0.05/0.1）。  

这条组合方案的核心思想是：**先把“实例是谁”尽量做对，再把 bbox 当作一个可优化对象显式地最小化邻片代价**；它最大化利用了你现有 GT 体系与“同 slide 多 section”的结构信息，比继续调单一扩张参数更可能把 hard case 压下去。