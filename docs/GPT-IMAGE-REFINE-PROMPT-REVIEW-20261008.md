# 本轮 refine prompts 微调整理（2026-10-08）

有值得微调的内容，但要区分「措辞/判据更清楚」与「已经证明出图更好」。本次只整理证据与短条款草案，不发请求、不修改4份生产模板、不接新阶段。范围包括10月7日人物/背景保真实验与10月8日背景颜色对照；均来自同一精灵源图，没有跨角色稳定性证据。

## 优先级与依据

| 优先级 | 文件 | 建议 | 证据强度与限度 |
|---|---|---|---|
| 1 | refine-audit.md | 明确 preserved=与对照相近；增加饱和度也失败；不确定不能同时填true | 直接观察到旧D2明显增加饱和度但本轮校准chroma=true，且部分true伴随不确定。判据澄清有明确依据；新增措辞的判读改善尚未证明，最终复核超时 |
| 1 | refine-audit.md | 两眼分别观察，以较差的一眼决定对应颜色字段 | 本轮4张等尺度裁剪都显示两眼变化不一致，画面左侧眼睛尤其容易灰化；笼统“眼睛仍是绿的”会遗漏。没有逐眼自动审计的成功复验 |
| 2 | refine-background.md | 当前工序输入负责具体外观，原图文字锚负责身份约束；不得照文字重新画脸/眼 | 背景只附人物结果，却重复“原图身份锚”。输入中的具体色块和原图文字可能不完全一致，存在重新解释的空间。属于提示词逻辑澄清，未被单变量验证为漂移原因 |
| 2 | refine-diagnosis.md、refine-background.md | 把清理对象、保留对象和替代材质写明确；区分真实草/碎屑与噪声 | 本轮3/4仍留大量细丝，1张清成石板；当前模板保留植被，与本轮清理目标之间可能有歧义。本轮背景任务来自人工固定描述，未重新调用诊断模型，不能声称证明了诊断模型失败或该条款造成失败 |
| 暂缓 | refine-person.md | 保留现有主要裂口、材质、裸手、人体连接与金边路线合同；先不增加更强修复条款 | 前轮C2/R-person主要裂口和可见绿色保留，但细金边/甲片仍重排；本轮没有重跑人物步骤，不足以选择新的衣装措辞 |

## 可用的短条款草案

以下是审阅草案，不是已启用固件。保持现有JSON键及调用接口，不需要为了这些文案改Python。

### A. 审计：先观察，再判“保持”

建议追加到 `refine-audit.md` 的颜色比较段，避免另起一套总分：

> “Preserved” means visibly comparable to the reference in that dimension, not merely that the feature remains present. Increased saturation is a chroma failure just as desaturation is; increased brightness is a value failure just as darkening is. Describe each visible iris separately in visual_evidence, excluding pupils, highlights and sclera. Mark a colour field true only if every assessable visible iris matches. Uncertainty about an iris requires the affected field to be false, not true with a qualifying limitation.

对应中文：保持指接近对照，饱和度提高和降低都可能失败；先分别描述两只可见虹膜，再给布尔结论；不能一只正常掩盖另一只漂移，也不能填true后在限制里写“不确定”。

注意遮挡不等于失败：源图不可见的虹膜不要求补画，也不要求凭空判断。已明确可见却因小尺寸/高光无法确认的比较，不能标true。增加文字不能保证模型执行，仍需要已有反例检验。

建议紧接背景比较段追加归因说明，**保留原图与当前输入两层检查**：

> Compare with Image 1 for accumulated source fidelity and with Image 2 for changes introduced by this stage. State which changes already existed in Image 2 and which are new in Image 3. Existing drift is still a source-fidelity failure; do not blame it on this stage or excuse new drift because the input was already different.

这能把「人物步骤已经改了」和「背景又改坏了」分开，但不会放宽原图保真门槛。

### B. 背景：澄清文字锚与当前图的关系

建议在 `refine-background.md` 的身份锚前追加一句，不再堆绿色/草绿色等颜色名称：

> Image 1 controls the exact appearance of the accepted person, including the existing iris pigment, highlights, face and garment details. The original-source anchor text identifies protected features; it is not an instruction to redraw them or normalize their appearance to a verbal colour label.

对应中文：当前人物图的具体色块、暗层、高光与衣装细节是保持依据；文字身份锚不是重新绘制或按颜色名称纠正人物的任务。原图总体保真仍由审计把关，不以当前输入取代原图。

这是明确分工的候选微调，**尚无图片实测证明收益**。本轮weak只替换末尾颜色段，并未测试这句，不能借weak的结果证明它有效或无效。

### C. 诊断/背景：先界定“什么可以清”

建议在诊断的背景段增加输出要求，继续写入现有 `background_tasks` 字符串，不必增加运行时必填字段：

> For each repairable background region, state the visible substrate, the specific marks permitted to change, their visual classification and the intended replacement surface. Distinguish actual vegetation, debris, cracks and cast shadows from unrooted line noise. Name protected objects separately. If classification is uncertain, preserve the marks rather than silently deciding to remove them.

背景模板只需一个优先规则：

> Only the marks explicitly identified as editable noise in background_tasks may be merged. Preserve real vegetation and objects; do not classify them as noise merely to obtain a clean floor. Explicitly authorized removal tasks must name their targets and existing underlying material.

这会减少“既要清干净，又要保留所有植被”的歧义。若用户明确要求清掉真实草/杂物，需要任务中有明确目标；当前通用诊断只收到源图，不能单凭这句获得新的删除授权。不能把本轮石板效果推广为“所有地面都变石板”。

## 不采用的方向

- **不把weak设成默认**：两个人物输入上均未显示明确优势，也未解决地面清理。每格一张，不能说weak在所有情境都无效，只能说此次没有替换依据。
- **不继续强调“看得见绿色”**：前轮强锚伴随亮黄绿，本轮弱锚又未解决灰化；继续堆颜色名没有稳定支持。
- **不把一个干净地面样本当稳定成功**：Q-baseline有清理收益，但眼睛变淡/偏黄；身份失败不能由整洁抵消。
- **不据此改变人物衣装力度**：本轮没有人物步骤的受控对照。保留有效的损坏状态和材质条款，金边重排仍列未解决项。
- **不从18像素样本制定通用HSV阈值**：初始共用mask已撤回，修订样本仍有覆盖/几何限制，不能用数字替代逐眼裁剪判断。

## 建议落地顺序

若之后决定微调，首先只改审计的语义和逐眼要求；再单独尝试背景输入权威那一句。背景对象分类属于另一个变量，不与颜色条款一起换。现在可以明确的是“哪些句子值得澄清”，还不能宣称某份新增prompt已经验证能改善出图。

依据：`data/test-result/20261007/elf-fidelity-round2/final-visual-review.json`、`data/test-result/20261008/refine-background-colour/calibration-review.json`、`manual-screen-review.json`、`face-screen.jpg`、`summary.json`。10月8日最终独立审计超时，没有可引用的新图自动pass。
