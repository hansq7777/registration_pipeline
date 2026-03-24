# Gallyas / Myelin Whole-Slide Crop-Level Tissue Mask 生成的系统诊断与实验路线图

这是一个“亮场组织前景提取（tissue masking）+ 目标组件选择（component/instance selection）+ 边界定位（boundary localization）”三合一的问题：在每个 section crop 内，把“真实组织前景”从玻片背景、邻近切片、玻片边缘、条带/标记等结构化伪影中分离出来，同时保证边界贴近人工标注（ground truth），并且在双侧组织或弱染边缘场景不发生坍缩。其难点不是单纯的二分类，而是 **强非均匀、强干扰、强约束的前景分割**：Gallyas 银染对髓鞘的显色极强、对其它组织抑制但不等于“无背景结构”，且信号会因制片/处理条件显著波动；同时 whole-slide 扫描背景也可能存在颜色温度漂移与 tile 边缘效应，使“背景=平滑白色”这一假设经常失效。citeturn4view0turn7view0turn8view0

## 问题重新定义与优先级

从算法视角，这类任务可拆成三个子问题（它们彼此耦合，导致你观察到的几类 failure mode 往往会连锁出现）：

第一类是“可分性”问题：在 Gallyas / myelin 切片中，组织外周/浅染边缘与背景的亮度、对比度、纹理差异可能很弱，而深染区域又非常强，这会让任何依赖单通道阈值或单尺度背景估计的策略处于两难：阈值低一点能捞回边缘但会引入背景纹理与伪影；阈值高一点能抑制 leak 但会丢边缘并诱发 mask collapse。类似问题在组织多孔、异质、弱染时是公认的 tissue masking 难点。citeturn8view0turn10view0

第二类是“目标选择”问题：crop 内常常不止一个连通域（双侧组织、碎片、邻近切片、玻片边缘、扫描框线/污点），因此你做的其实是 **“从多个候选前景组件里选出属于本 section 的那一组组件”**。一旦你的 core/ownership/center 逻辑只允许单一“赢家”，双侧组织被拆开后只保留一边是结构性必然结果，不是偶发 bug。

第三类是“边界定位”问题：哪怕“选对了组件”，边界仍可能系统性外扩一圈或内缩一圈——这通常不是 Dice/IoU 能敏感捕捉的错误，却会直接伤害背景清理、tissue crop 导出、跨染色配准，以及任何依赖 mask 作为硬约束的 downstream（例如 stain translation 训练时的背景一致性）。边界类指标在医学影像与分割评估中被反复强调需要单独关注。citeturn6search2turn2search7turn6search1

优先级上，最该先压住的是两类“硬失败”：  
其一是 **leakage（把组织外背景/邻近结构/伪影吞进来）**，因为它会把下游所有任务的输入分布污染；其二是 **collapse（只剩组织的一小块）**，因为它会导致 crop 导出缺失与配准失败。这两类先稳住后，再用边界 refinement 去把 BF score / HD95 / ASSD 拉起来更划算。

## 失败模式到算法根因的映射

你描述的每一种失败现象，几乎都可以对应到“候选生成—约束/选择—形态学—传播/重建—后收缩”链条里的某一类机制性偏差；关键是把它们从“现象”映射到“哪一步产生了不可逆信息损失或不可控扩张”。

### 边界偏离 GT 一圈

最常见的根因是“低频背景估计 + 全局阈值 + 大核 closing”的组合产生系统性膨胀：  
你 legacy residual 做法本质上是在用模糊版本近似背景，再做差得到“前景增强图”。模糊会在高对比边界附近产生 halo；随后 Otsu 这类全局阈值法假设直方图近似双峰（前景/背景可分），当背景存在结构、噪声或弱纹理时，该假设会失败，阈值会被拉向“把 halo/弱纹理也当作前景”的方向。citeturn9search0turn10view0turn8view0  
再叠加大核 closing，会把边界外侧一圈“接上去”，形成你观察到的“外扩一圈、边界松散”。这里的关键不是 closing 本身，而是 closing 在“候选里已经包含边界 halo/弱背景纹理”时会把误差变成拓扑不可逆的连通扩张。

另一个常被忽略但值得怀疑验证的点：你评估的 BF@32/BF@64 若以像素计，而不同 crop 的分辨率（μm/px）或缩放链路不一致，会让“外扩一圈”的定量表现看起来更严重或更轻微。citeturn9search3turn6search1  
在 Hamamatsu NanoZoomer 的常见扫描设置下，20×/40×对应约 0.46/0.23 μm/px 量级，像素尺度与形态学核大小若没统一到物理尺度，会直接导致某些样本“膨胀感更强”。citeturn9search3

### mask collapse 到很小

collapse 通常是“约束过强 + seed 偏置 + 传播受阻”的合成结果：

如果 core 来自“深染高置信区域”，那么在 myelin/Gallyas 的强非均匀场景里，core 会天然偏向深染块；而 weak-stained outer cortex / 边缘区在候选里本就不稳定，于是传播阶段（reconstruction/region growing）会在“低对比、低置信”处停止，最终只保留深染核心。该现象在 EntropyMasker 等工作里被明确点名：透明/弱染/低对比使传统阈值与简单前景提取难以泛化。citeturn8view0

如果你的“center-constrained / ownership”逻辑实质上是“离中心最近/覆盖中心最多的组件胜出”，那么双侧组织或被断桥后分裂的组织会被强行单实例化；即使候选有两侧，赢家也只留一侧，另一侧被当作伪影丢弃——这不是参数问题，而是目标函数定义的问题。

### leak 到组织外或邻近结构

对于 leak，经典形态学操作（尤其是 closing）和基于连通性的重建/传播是“双刃剑”：它们不理解“哪些连通是伪连接”。只要候选里存在细桥或弱纹理连接，重建会把连接另一端的大片区域“合法化”。你已经观察到 bridge/appendage 的问题，这通常意味着你缺少一个“阻断机制”：要么在传播时引入背景种子/禁止区域，要么在候选阶段显式切断细桥再做组件评分。

另外，全局阈值在有伪影时会被显著干扰：在 H&E 的系统评估里，Otsu 在含伪影样本中会把伪影当组织，甚至因为伪影影响阈值导致组织被当背景。虽然你的染色不是 H&E，但这个结论在形式上直接适用于“阈值依赖双峰假设、伪影改变直方图形态”的场景。citeturn10view0

### 双侧组织时只保留一边

根因几乎总是“单核心/单赢家”的选择策略：  
只要你的 core 或 seed 只有一组、目标是单 connected component，双侧结构在被开操作或断桥后会变成多个 component，于是只保留一边。修复思路不应是“减少断桥”，而应是让“正确答案允许多组件”，并用更合理的组选择准则（例如 top-K 组件 + 形状/位置一致性）代替 winner-takes-all。

### structured artifact 被吞进去

结构化伪影（玻片边缘、条带、扫面框线、邻近切片、深色标记）之所以难，是因为它们往往具有“稳定的几何结构 + 不同于噪声的纹理/边缘”，更像一个“对象”而不是随机噪声；因此纯形态学很难可靠排除。数字病理质量控制与伪影定位工具（例如 HistoQC、GrandQC 的思路）普遍不是只靠阈值，而是组合颜色直方图、亮度/对比度、边缘检测器，甚至加监督分类器去识别伪影区域。citeturn4view1turn3search8  
你当前 artifact mask 为空，意味着 pipeline 里没有“专门针对 structured artifacts 的负类建模”，于是只要它们在候选中被包含且与组织连通，后续传播/closing 就会把它们吞进去。

## 针对 Gallyas/myelin WSI 的 tissue masking best practices

这一部分按你指定的五类方法组织，但每类都尽量贴近“crop-level、Gallyas/myelin、边界弱+局部深染强+结构化干扰”的约束。

### 经典图像处理与规则法

在 whole-slide / crop-level tissue masking 领域，最稳的一类最佳实践不是“找一个更聪明的阈值”，而是 **先构造一个更分离的表示（representation），再做阈值/聚类**。H&E 里常见的是对亮度/色彩通道做变换来获得双峰分布；你这里的对应策略是：针对 Gallyas 的“银沉积导致深色结构”与背景“低纹理/高亮度”为主的特性，构造能同时抑制背景纹理、保留弱染组织边缘的组合特征（而不是单一 residual）。citeturn10view0turn8view0

可直接借鉴的两个“代表性思路”：

其一是 entropy/texture 驱动的前景提取：EntropyMasker 用局部熵来分离“背景（低熵、同质）”与“组织（高熵、异质）”，并指出 WSI 背景常因扫描/处理条件出现颜色不一致与 tile 边缘伪影，使单纯 HSV 排除白色也不可靠。citeturn8view0  
对你而言，这个思路的价值在于：**弱染但仍有细纹理的组织边缘，往往比玻片背景更高熵**，因此熵能当作“召回边缘”的支撑证据，而不是靠降低亮度阈值去硬捞。

其二是 stain-aware 的分离：颜色去卷积（color deconvolution）把 RGB 在光学密度空间按染料吸收向量分解，是数字病理里分离染色贡献的经典工具；即便你的 stain 不是 HED（hematoxylin-eosin-DAB）那套，“先做光学密度变换再在某个主轴/主成分上做阈值或聚类”的思想仍然能把“染色变化”与“光照/背景变化”解耦。citeturn1search0turn1search4  
进一步，染色归一化（例如 Macenko 方法）被广泛用于降低批次间颜色/强度差异；在你强调跨切片配准和潜在 stain translation 的场景下，哪怕只在 mask 生成的特征通道上做“轻量归一化”，也可能显著降低“同一规则在不同样本上阈值不一致”的问题。citeturn2search0turn2search24

工程落地层面，许多开源工具都把 tissue masking 作为基础模块，并默认采用“低分辨率阈值 + 形态学”作为 baseline（例如 entity["organization","TIAToolbox","python computational pathology"] 的 tissue masking 示例、luminosity threshold mask）。这类 baseline 的优点是快、可解释；缺点是在强伪影与弱染边界下不稳，需要你现在这种更强的 hybrid 约束。citeturn4view3turn3search1

### 基于 marker / geodesic / reconstruction / active contour 的方法

对你这种“既要抑制漏检又要抑制外泄”的场景，marker-based 方法的核心优势是：**可以把“前景应该从哪里长出来”和“哪些地方绝不能长进去”编码成硬/软约束**，从而降低对单一阈值的敏感性。

形态学重建（morphological grayscale reconstruction）提供了一类非常实用的“形状保真”工具：相比普通 opening/closing（会改变大结构的边界形状），opening/closing by reconstruction 更倾向于“去掉不符合 marker 的连通部分，同时保持保留下来的主体形状”。这对你现在“closing 放大错误、导致外扩和桥接”的症状尤其对症。citeturn1search5turn1search1

更强一档的是图模型/随机游走/图割：  
随机游走分割（random walker）在有少量前景/背景标记点时，可以计算每个像素到各类标记的到达概率，从而得到高质量边界；它的典型优势是对噪声与弱边界更鲁棒，且天然支持多 label。citeturn0search2turn0search10  
图割（graph cuts）与 GrabCut 系列把分割写成能量最小化：同时利用区域项（像素/颜色模型）与边界平滑项，并允许用前景/背景种子作为硬约束。对你来说，最关键的是可以显式把 crop 边缘一圈当作“背景种子”，把一些高置信伪影当作“背景/禁入种子”，这样传播就不会靠连通性“误吞”。citeturn5search2turn5search3

边界精修方面，“主动轮廓”两条路线值得区分：  
一类是基于梯度的 geodesic active contour（适合边界梯度清晰时“贴边”）；另一类是区域模型（active contours without edges / region-based），适合边界梯度不清晰但区域统计可分时做平滑分割。你的 Gallyas 场景往往是“外周边界梯度弱、但组织/背景统计仍可分”，所以 region-based 模型在边界 refinement 上反而可能比纯梯度驱动更稳。citeturn5search1turn1search2  
如果担心 PDE 数值成本，形态学蛇（morphological snakes）用形态学算子近似主动轮廓演化，在工程上更容易落地并控制迭代次数。citeturn5search16turn5search8

### topology-aware 或 component-selection-aware 方法

你当前的“collapse / 双侧只保留一边”说明：component selection 需要从“单 component winner”升级到“component set selection”。

一个有效且易验证的最佳实践是：把问题改写为“从候选连通域集合中选择一个子集，使其同时满足面积、形状、位置、纹理一致性，并最大化与 core seeds 的覆盖”。这类方法不需要深度学习，也不需要复杂拓扑约束理论，只需要你把“允许多组件”写进目标函数里。

具体建议是把评分拆成四组可解释的子打分（每个都能做 ablation）：
面积与面积比（pred/GT proxy）、形状紧致度（solidity、convexity、周长-面积比）、位置与覆盖（与 bbox/中心线/多 seed 的 geodesic 距离）、纹理一致性（熵/局部对比度分布是否像组织而非玻片边缘）。  
之所以强调“可解释 + 可 ablation”，是因为在你已有丰富指标体系的情况下，最怕的是引入一个黑箱规则导致某类 hard case 被系统性误杀。

另一个常被低估的 best practice 是引入“桥接惩罚”：对候选 mask 的细桥（narrow isthmus）做显式检测并切断，然后在切断后的 component 图上做 selection；这比用 opening/closing 盲目“打断细桥”可控得多，因为你能把“桥宽阈值”定义在物理尺度上，并观察它如何影响双侧保留率与 leak。形态学重建/骨架/距离变换都可以支持这种桥检测，但关键是把它当作 **可调的拓扑操作**，而不是隐含在 opening/closing 里。citeturn1search5

### 小样本 GT 条件下的轻量学习与弱监督

当 classical/hybrid 已经把 leak/collapse 压到可接受范围，但边界仍“差一口气”时，小样本学习往往能给你一个更好的“特征融合器”：它能把颜色、纹理、弱梯度边缘等信息用数据驱动方式组合起来，而不是手工加权。

最常见的轻量方案是 U-Net：它的原始设计目标之一就是在数据量有限时配合强数据增强仍能训练出可用的分割模型。citeturn2search1turn2search5  
对你这个任务，建议把学习目标定在“低分辨率/中分辨率的 tissue mask”（例如把 crop 下采样到固定 μm/px），让模型聚焦于组织边界的宏观形状而不是髓鞘细纤维细节；再用一个确定性的后处理把边界上采样回原尺度并做少量 contour refine。这样更符合你的用途（背景清理、crop 导出、跨染色配准），也更贴合 GT mask 的语义。

弱监督路线可以考虑“自动种子 + 图模型”作为中间态：用你现有 high-recall candidate 生成前景种子、用 crop 边缘生成背景种子，再跑 random walker / graph cut。它比训练深网更省标注，但比纯规则更鲁棒，是很适合“有少量 GT、但希望快速验证”的折中。citeturn0search10turn5search2

### boundary-aware loss / postprocessing / contour refinement 思路

你已经把 boundary F1、HD95、ASSD 纳入评估，这是非常正确的方向，因为体素/像素重叠指标可能在边界小偏差时不敏感。citeturn6search2turn2search2

如果走学习路线，最直接的 best practice 是把损失函数也边界化：  
Boundary loss 通过距离变换把优化重点放在轮廓界面上，专门用于改善边界相关表现。citeturn1search3turn1search7  
针对 Hausdorff 距离（尤其你用的 HD95），已有直接面向 Hausdorff 的可微损失近似，目标就是减少大偏差点而不牺牲整体重叠。citeturn2search2

如果仍以 classical/hybrid 为主，后处理可以用“轻量边界吸附（snapping）”而不是大核形态学：例如用 geodesic/region-based active contour 从当前 mask 出发做少量迭代，把边界贴到最可能的组织-背景分界上；其好处是 **把边界误差当作连续优化问题**，而不是一次性膨胀/腐蚀。citeturn5search1turn1search2turn5search16

## 候选改进路线

这里给出 6 条“下一步最值得做、且能被你现有评估体系清晰验证”的方向；它们按“对症程度 + 工程代价 + 可解释性”排序，但不等于最终优先级（优先级在下一节会给出）。

### 物理尺度统一与重建型形态学替换

主要解决：边界外扩一圈、不同样本/分辨率下参数不稳定、closing 放大错误。  
改动位置：把所有 kernel size、桥宽阈值、最小组件面积从“像素”改为“μm”，并用 opening/closing by reconstruction 取代大核 closing/opening。citeturn1search5turn9search3  
需要输入：metadata.json 里的 μm/px（或从 NDPI 元数据读取）。citeturn9search3turn9search2  
评估重点：BF@32/BF@64（改成以 μm 表示的 tolerance），HD95、ASSD、FP area/GT area。citeturn6search1turn6search2  
预期改善：边界系统性外扩会下降，尺度泛化更稳。  
副作用：若 reconstruction marker 选得过严，可能增加 collapse（因此必须配合下一条“多种子/多证据”）。

### 多证据候选：熵纹理 + 亮度/光学密度的联合候选

主要解决：弱染边缘漏检 vs 背景纹理误检的两难。  
改动位置：候选生成阶段不再只用 residual+Otsu，而改用“（局部熵阈值）∪（亮度/光学密度阈值）”的高召回候选，再交给后续约束收紧。熵作为“弱染边缘支撑证据”尤其关键。citeturn8view0turn3search1  
需要输入：RGB crop。  
评估重点：recall、predicted area/GT area（防 collapse）、但同时要看 FP area/GT area（防 leak）以及 boundary band 内的 FN（看边缘是否被救回）。  
预期改善：弱染 outer cortex 的召回上升，collapse 几率下降。  
副作用：候选更“脏”，若后续没有强约束/背景种子，leak 可能上升。

### 滞回阈值 + 受限传播：从高置信 core 只在“允许域”内增长

主要解决：collapse、细桥导致的误吞、边界不稳。  
改动位置：把传播写成“hysteresis + reconstruction”：  
高阈值得到 core seeds（高 precision），低阈值得到 support mask（高 recall），最终前景 = 从 core 在 support 内做形态学重建/地质膨胀得到的可达域；并引入“禁止域”（见下一条）。这种结构在形式上与 Canny 的双阈值思想相似，但在二值区域层面可控得多。citeturn1search5turn4view2  
需要输入：core、support、background barrier（至少 crop 边缘一圈）。  
评估重点：collapse cases 的 predicted area/GT area、双侧保留率（组件数/左右覆盖）、FP area/GT area（尤其远离 GT 的 FP）。  
预期改善：既能保留弱边缘（因为 support 更宽松），又能防止无 seed 的地方被误吞。  
副作用：如果 core seeds 偏置（只落在一侧），仍会保留单侧；因此需要“多 seed 策略”（见后文优先级计划）。

### 显式背景/伪影种子：random walker 或 graph cut 的自动化版本

主要解决：结构化伪影被吞、细桥连通导致的 leak、“背景必须被排除”的硬约束。  
改动位置：在 crop 内自动生成三类种子：前景 seeds（来自 core）、背景 seeds（来自 crop 边缘一圈、以及高亮低熵区域）、伪影 seeds（来自线状/高对比异常区域），然后跑 random walker 或 graph cut 得到最终分割。图模型能够把“边界平滑”与“区域相似”结合，并允许硬约束。citeturn0search10turn5search2turn5search3  
需要输入：种子生成规则（可先用启发式）。  
评估重点：leak 相关指标（precision、FP area/GT area、HD95 的尾部）、以及“伪影附近局部区域表现”。citeturn10view0turn2search2  
预期改善：对 structured artifacts 的稳健性显著提升，且对双侧组织可天然多标签/多连通。  
副作用：实现复杂度上升；若权重/平滑项过强，边界可能变得过平滑、损失细节（但对 tissue mask 通常可接受）。

### 组件集合选择：从“赢家通吃”到“多组件保留 + 评分合并”

主要解决：双侧只保留一边、断桥后误删、深染块劫持。  
改动位置：在“桥切断”之后，不强行取单最大 component，而是允许取 top-K（K=2 或 3）并用评分函数过滤（面积下限、形状紧致、与种子可达、与 bbox 中心/对称轴的几何关系）。  
需要输入：候选二值 mask、桥宽阈值（μm）、评分特征。  
评估重点：双侧覆盖率、recall、predicted area/GT area，同时监控 FP area/GT area。  
预期改善：双侧组织与 hard case 的 collapse 明显减少。  
副作用：如果评分函数偏松，可能把邻近切片也保留进来；因此必须与“背景 barrier/禁止域”联动。

### 轻量边界吸附：主动轮廓/形态学蛇作为最后一公里

主要解决：边界偏离 GT（一圈外扩/内缩）、边界不贴轮廓。  
改动位置：把你当前 hybrid 的输出当作初始化，然后在某个“更适合边界的能量图”上做少迭代 contour refine：  
边界能量图建议来自（1）梯度幅值（适合边界清晰处）+（2）区域统计项（适合弱梯度边界）。geodesic active contour 与 region-based active contour 分别对应这两种机制。citeturn5search1turn1search2turn5search16  
需要输入：初始 mask、能量图、迭代步数与平滑参数。  
评估重点：BF score、HD95、ASSD（尤其是 boundary band 内的误差），并检查是否引入新 leak。citeturn6search1turn6search2turn2search2  
预期改善：边界指标显著提升，Dice/IoU 可能变化不大但下游更受益。  
副作用：若能量图在玻片边缘/伪影处也有强梯度，可能把边界拉向错误位置；因此要把 contour refine 限制在“距初始边界一定范围内”的窄带上。

## 优先级排序后的实验计划

这里给出一个“先易后难、先稳后精”的计划：前四个实验尽量只动 classical/hybrid/postprocessing；最后两条是更强方法（contour-based、learning-based）。每个实验都给出参数扫描、hard case 选取、成功标准与失败含义，便于你用现有 GT 测试集做快速闭环。

| 方法名 | 核心思路 | 主要扫参 | Hard cases 选择建议 | 成功标准（建议以 μm 尺度统一） | 失败说明什么 |
|---|---|---|---|---|---|
| μm-统一 + 重建型形态学 | kernel/面积阈值全部转为 μm；用 opening/closing by reconstruction 取代大核 closing/opening，减少边界系统性膨胀 | 重建 marker 的生成阈值；reconstruction 的结构元素半径（μm）；最小组件面积（mm²/μm²） | “外扩一圈”最典型的样本；背景有弱纹理但组织轮廓清晰的样本 | FP area/GT area 明显下降；BF@t（t 以 μm）上升；HD95/ASSD 减小且不过度损失 recall | 你的边界误差主要不是形态学膨胀，而是候选本身就偏外/偏内（需要从候选表示入手） |
| 熵+亮度联合候选 | 用局部熵“救回弱染边缘”，用亮度/光学密度“压住纯背景”，形成更鲁棒的高召回候选 | 熵窗口半径（μm）；熵阈值选取（直方图谷值/分位数）；亮度阈值 | weak-stained outer cortex；边缘发虚但内部纹理存在；背景有 tile/温度漂移 | boundary band 内 FN 明显下降；predicted area/GT area 更接近 1，且 FP area/GT area 不显著上升 | 熵无法区分背景纹理与组织纹理（需要增加禁止域/背景种子或更强的特征分离） |
| 滞回+受限传播 | 高阈值 core seeds + 低阈值 support；从 core 只在 support 内重建；并引入 crop 边缘背景 barrier | 高/低阈值对；边缘 barrier 宽度（μm）；重建/生长迭代策略 | collapse 样本；深染块劫持样本；细桥连接伪影样本 | collapse 率显著下降；双侧保留率提升；远离 GT 的 FP 面积降低 | core seeds 偏置或 support 域仍包含太多伪连接（需要多 seed 或桥切断/伪影禁止域） |
| 多组件集合选择 | 桥切断后允许保留 top-K 组件；用“可达性+形状+位置+纹理”评分过滤 | 桥宽阈值（μm）；K；评分权重/阈值；组件最小面积 | 双侧组织、分裂组织、邻近切片靠得很近的样本 | 双侧覆盖率上升但 leak 不上升；predicted area/GT area 稳定接近 1 | crop 内负类（邻近切片/伪影）与正类在评分特征上不可分（需要显式伪影模型或图模型） |
| 图模型分割（random walker/graph cut） | 自动生成前景/背景/伪影种子；用图模型结合区域与边界信息做全局最优/概率分割 | 种子生成规则阈值；边界权重（平滑项）；特征通道选择（亮度/熵/OD） | structured artifacts 最重的样本；玻片边缘/条带/框线显著的样本 | FP area/GT area 与 HD95 尾部显著下降；局部伪影区域 precision 提升 | 种子生成不可靠或特征通道不匹配（优先先把种子规则做成可解释可控） |
| 小 U-Net + 边界化损失 | 下采样到统一 μm/px 训练轻量 U-Net；用 boundary loss 或 Hausdorff 相关损失强化边界；输出概率图再阈值+窄带 refine | 下采样尺度；损失权重（Dice+Boundary/Hausdorff）；数据增强强度；后处理阈值 | classical 已经稳住但边界仍差的样本；跨批次强度差异样本 | BF 与 Surface Dice 明显提升且 leak 不反弹；对 hard case 泛化不崩 | 数据分布差异过大或标注一致性不足（需做 stain normalization/更强增强/补硬例标注） |

上述计划中，前三个实验的“验证成本最低”：它们基本不改变整体框架，只是把关键操作从“隐式不可控”改为“显式可调、可解释”，并且能直接反映在你关注的 boundary/leak/collapse 指标上。

## 边界为中心的评估与调参框架

你已经避免了“只看 Dice/IoU”，这是非常关键的判断；但要让这些指标真正指导调参，需要进一步把 metric 体系“结构化”，让每个指标都对某类 failure mode 有明确指向。

### 建议重点加入或强化的边界指标

边界 F1（Boundary F1 score，常见实现为 BF score）用一个距离容忍度判断预测边界点是否匹配 GT 边界点，本质是边界上的 precision/recall 的调和平均，因此非常适合描述“外扩一圈/内缩一圈/边界锯齿”的问题。citeturn6search1  
建议你把 BF@32/BF@64 从“像素”转换为“μm 容忍度”，否则跨分辨率比较会混淆。citeturn9search3turn6search1

Surface Dice similarity coefficient（表面 Dice，相当于“在给定距离容忍度内，两条轮廓表面的重叠比例”）在放疗轮廓等高度边界敏感任务里被提出，目的就是比体积 Dice 更贴近“人工修边”的工作量。citeturn2search7turn2search19  
对你这类 tissue mask，Surface Dice 往往比 Dice/IoU 更能反映“边界清理与配准友好度”。

Hausdorff distance 与其百分位版本（HD95）用于衡量最坏情况下的边界偏差（HD95 用 95% 分位减弱极端离群点）；ASSD（average symmetric surface distance，平均对称表面距离）更像“平均边界误差”。两者结合能区分“偶发的大 leak（HD95 高）”与“系统性一圈偏移（ASSD 高）”。citeturn2search2turn6search2  
同时要保持怀疑：距离类指标的实现细节（边界定义、采样方式、体素各向同性处理）很容易引入坑，建议固定实现并做单元测试（例如用简单几何形状验证）。citeturn6search13

### 局部区域分段评估的可操作方案

你提出的 top/middle/bottom、left/center/right、boundary/core 分段非常对症；建议把它做成三个相互正交的切片方式：

空间网格分段：把 crop 均匀切成 3×3（或 4×4）网格，对每格分别算 precision/recall、FP area/GT area、FN area/GT area。这样能快速定位“某些 scan strip 在上半部导致误检”之类的结构化问题。

边界/核心分段：对 GT mask 做距离变换，定义一个“边界带”（例如距 GT 边界 ≤ d μm）与“核心区”（GT 腐蚀 d μm 后的内部）。所有指标都分别算在边界带与核心区：  
边界带：主要看 BF/Surface Dice、边界带内 FP/FN；  
核心区：主要看 recall、防 collapse。  

近边界 vs 远边界的 FP 分解：把 predicted 的 FP 分成两类——靠近 GT 边界的 FP（可能是“外扩一圈”）与远离 GT 的 FP（更可能是“leak 到背景/邻近结构/伪影”）。这能让你在调参时不至于被“总 FP”误导：有时总 FP 不变，但远离 GT 的 FP 大幅下降，实际上对下游更友好。

### 用局部指标指导调参的规则化方法

当 boundary band 的 FN 高而 core recall 还可以：优先调“候选召回边缘”的那一段（例如熵候选阈值、support 域阈值），而不是放宽 core 或减少断桥。

当远离 GT 的 FP 高、但 boundary band 的 FP 低：优先做背景 barrier/禁止域/伪影种子（图模型或规则伪影检测），而不是收紧全局阈值（收紧会先杀死弱边缘，导致 collapse）。

当双侧组织只保留一边：优先动 component set selection（允许多组件、K、评分），不要把精力花在“把桥尽量不断”（不断桥会让 leak 更隐蔽且更难控）。

当 BF/Surface Dice 低，但 Dice/IoU 已经不错：这通常意味着“边界需要最后一公里吸附”，优先尝试窄带 contour refinement（主动轮廓/形态学蛇），并监控它是否带来新 leak。citeturn5search16turn1search2turn6search1