# DeepSeek：清色重复验证与只读指标

你在 `D:\code\image-maker` 工作。先完整阅读 `PROJECT_REQUIREMENTS.md` → `AGENTS.md`，再读：

- `docs/261003-color-improve/STAGE2-CHROMA-REVIEW-AND-PLAN-20261006.md`
- `docs/style-extraction/STYLE-METRICS-GRAM-V2-20261006.md`
- `utils/color_direction_pack.py`、`tools/color_knowledge.py` 的冻结、发送计数、成功缓存与盲评机制。

Codex 已修复 Gram 归一化、版本缓存与真实图片配对入口。本轮主任务仍是验证清色提示词，深度指标只读辅助。不要再修公式，不要覆盖历史 v1 数字，不把深度距离换成配色成功率。

## 先检查有没有已经执行，禁止重复计费

检查已有清色重复实验包、ledger、请求快照与报告，尤其 `data/test-result/20261006/color-clear-repeat-validation-v3/`，也核对同内容的其他目录。上一份交接的预算与本份是同一轮 **9 次图片请求＋3 次视觉评审**，不是新增额度。已经成功的槽位直接复用；已发送失败也不自动重发；sending/unknown_after_send 必须暂停后续联网，报告可能收费的槽位。不能清 ledger、换目录或改 ID 绕过发送预算。若整轮已结束，只补离线核验、只读指标和报告。

如该轮尚未准备，按下述方案离线建立新冻结包，沿用上述目录。不自行扩大实验。已有冻结包必须先核对是否符合本方案；存在冲突只报告差异，不修改已发送证据。

## 未执行时的实验规格

一个简化桌面静物场景，优先复用上一轮 S2 的对象与保护色绑定，固定正面视点、对象位置和温和单侧光。不要人物、天空、复杂玻璃、额外画风参考、衣装、氛围或重绘。

三条件各三次独立生成：A 第一阶段固定色板基准；B 当前生产清色规则；C 与 B 完全相同，仅追加下面候选正文。主次、面积、辅助落点、其他参数各条件一致。C 仍是 clear 方向，不能用原浊色 C 的解码器冒充。

```text
On the explicitly authorized blue and red surfaces only, favour a modest reduction of grey admixture so their existing hue families read more distinctly. Keep the original illumination, local light-and-shadow modelling and material character; do not achieve this by increasing exposure, bleaching highlights or darkening the scene. Red accents must remain recognizably red, rather than pink, salmon, beige or neon. Preserve all protected colours and the existing geometry. This is a qualitative local preference, not a global saturation filter.
```

不得覆盖生产模板。冻结完整最终正文、base 正文与候选追加正文各自 hash、颜色计划、来源 hash、模型/端点/尺寸/分辨率、9 槽位与 3 评审组、预算与自动重试关闭状态。逐槽位复算 App 基准/B 规则；C 验证“B 原文＋唯一追加正文”，不要伪造生产 plan hash 来掩盖实验追加。

如既有 runner 不支持重复 ID、候选追加及平衡顺序，允许在 `utils/color_direction_pack.py`、`tools/color_knowledge.py` 增加最小通用协议支持并在 `tests/` 用临时目录回归；不得写根目录一次性脚本、跳过 verify、硬编码某一轮结果或改 App/生产模板/共享配置。仅离线扩展，验证通过才允许发送。

R1 展示 A/B/C，R2 B/C/A，R3 C/A/B；每组只含本重复的三张图，标 N1/N2/N3。评审模型看不到 A/B/C、实验措辞与预期方向。映射、绑定图 SHA-256、展示顺序在发送前冻结；解码时严格还原。3 次独立评审，一次一组；每槽位/组最多发送一次，无补图、挑图、重试或额外探测。沿用已冻结配置的 Gemini 纯文字首图和视觉评审端点，主机 `new.aigc2d.com`；参数不足停在离线阶段。

依次走现有 direction-pack 的 verify → preflight → run → review → evidence → report → status。具体参数按新包文件路径生成并记录；每步检查退出码，不无条件串行下一步。preflight 必须显示总槽位 9、评审组 3、上限 9/3，已有发送数须与 ledger 一致。报告渲染失败仅离线修报告，禁止重发收费操作。不得执行 replay 脚本。

## 视觉评审与验收

逐个指定蓝／红表面比较 B-A、C-A、C-B。分别记录去灰／色相清晰度、鲜艳程度、明度，不能只以变亮或更饱和判 clear。红色偏粉、鲑粉、米色、消失、串色单列；保护色、对象数、光照、材质、画法与可读性检查保留 uncertain。region_id 原样保留，漏项不得算通过。

将普通随机轮廓变化、影响局部颜色判断的实质混杂、导致无法判断的缺失分开报告。观察到方向、候选合规、完整可比对照三个数分别列出；不得修改前轮冻结门禁来获得更高通过率。至少两组同向且无新增保护色失败，可作为继续验证的实用信号，不是生产硬门禁或统计显著性。如果三组仍无一致信号，停止继续叠加清色形容词，下一步转主次关系。

## 免费的本地只读指标

用同一系统 Python `C:\Program Files\Python310\python.exe`，`-X utf8 -u`。不安装包、不建 venv、不升级 torch、不下载权重、不改驱动、不停止/重启 app。当前 lpips/clip 可能缺失，如实记 unavailable，不以其他方法替代。不要因指标缺项重跑图片或阻断已有视觉证据整理。

Codex 新版 CPU Gram/AdaIN 校准已经落盘，可复用 `data/test-result/style-metrics-cpu-gram-v2-20261006.json`。不要使用旧 Gram v1 校准数。新版 CUDA/NPU 尚未验证；本轮为统一口径使用 CPU/FP32 读取原图即可，不自动探测或切换 GPU/NPU。

从成功 ledger 的最终原图路径构建三组 B-A、C-A、C-B 共最多 9 对。缺图的配对记缺失，不补图。一次进程通过重复 `--pair` 传入所有配对，避免重复加载权重。参数示例（把示例路径替换为 ledger 真实路径）：

```powershell
$metricArgs = @('tools/style_metrics_verify.py', '--device', 'cpu', '--precision', 'fp32', '--metrics', 'gram,adain,lpips,csd', '--no-compare-cpu', '--pair', '<R1-B原图>', '<R1-A原图>', '--pair', '<R1-C原图>', '<R1-A原图>', '--pair', '<R1-C原图>', '<R1-B原图>', '--out', 'data/test-result/20261006/color-clear-repeat-validation-v3/local-metrics/pairs-gram-v2-cpu-fp32.json')
& 'C:\Program Files\Python310\python.exe' -X utf8 -u @metricArgs
```

实际加入其余两组，不把示例占位符直接执行。目标文件已存在时先核对图片 hash、公式版本、权重/预处理与配对是否完全一致；一致复用，冲突保存新版本文件，不覆盖旧证据。退出码 1 时区分 unavailable 与计算错误。各有效指标必须有限、同图自检通过、实际设备为 cpu，Gram=`gatys-layer-sum/v2`；LPIPS/CSD 缺项保留空值。核对计算前后原图 hash 未变。

只报告距离/相似度及局限；整体指标变化可能来自颜色、几何或画法，不能归因于局部去灰。尤其不能用“Gram 更像”推翻保护色失败或主体漂移。不要新增总分、阈值或百分比。

## 交付及下一步

输出运行状态、发送次数、冻结包/hash、完整证据、原图画廊、盲评/映射/解码、三个独立统计、只读指标 JSON、缺项及混杂说明。在 `docs/261003-color-improve/` 写本轮报告，明确实际完成/失败/未验证内容。不得把未知当成功，不输出密钥或认证头；记录必须脱敏。

推荐下一轮按证据决定：清色有信号则先跨场景/色板；没有则保持试验状态，进入二阶段宽松主次关系，以可比大面积表面测试视觉份量，不要求精确 50/50，不新增或放大物件；辅助绑定与主次层级分开验证。第三阶段才研究既有材质与光线、知觉突出/后退、画风迁移和重绘保色。情绪/季节是用户选择后的 refine 选项，不覆盖手动固定色板。

全程禁止本地蒙版计算后羽化、叠加、上色、局部补丁贴回；只读特征计算不修改图片。任务结束不要自动发布、重启 app 或改生产清色规则。
