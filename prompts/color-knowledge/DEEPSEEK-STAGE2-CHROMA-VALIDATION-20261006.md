# 二阶段下一轮：清色／浊色与红色强调色保留

你在 `D:\code\image-maker` 工作。先读 `PROJECT_REQUIREMENTS.md` → `AGENTS.md`，再读 `docs/261003-color-improve/STAGE2-FIXES-AND-NEXT-20261006.md`。本任务已经授权下面的 6 次图片请求和 2 次视觉评审请求；不授权额外补图、重试或探测请求。代码和实验包已准备好，请执行验证，不要重新设计实验或修改生产模板。

上一轮的深沉方向两场景均可见，但“方向有变化”不能直接等同于完整可用成功；明亮与并重证据不足，三层有红色变粉的问题。本轮只测试清色／浊色，并检查强调色保留。主次面积、情绪／季节、材质与重绘不混入本轮。

## 冻结输入与预算

- Pack：`prompts/color-knowledge/theme-color-chroma-validation-v2-pack.json`
- 场景：`prompts/color-knowledge/theme-color-chroma-validation-v2-scenes.json`
- 盲评模板：`prompts/color-knowledge/theme-color-chroma-validation-v2-review.md`
- Pack canonical SHA-256：`e8ef077a0079b91ba5829d120a6d719762b0608ed38d7611f0f380c249058334`
- 唯一输出目录：`data/test-result/20261006/color-chroma-app-v2-validation-v2`
- S1 单人海边露台、S2 桌面静物；各 A 基准、B 清色、C 浊色。每槽位最多发送一次图片请求，每场景 A/B/C 一组盲评，最多一次评审请求。总预算 6＋2，不保证成功图数。
- 只走当前配置的 Gemini 纯文字首图与支持图片的视觉文本评审通道，授权主机仍为 `new.aigc2d.com`。模型、尺寸、分辨率从现有配置读取并冻结；不得改配置、探测模型或换端点。参数缺失则停在离线阶段报告。

## 执行顺序

用系统 Python `C:\Program Files\Python310\python.exe`，所有动作都使用同一组参数。在 PowerShell 定义：

```powershell
$colourArgs = @('tools/color_knowledge.py', 'direction-pack')
$colourInputs = @('--out', 'data/test-result/20261006/color-chroma-app-v2-validation-v2', '--pack', 'prompts/color-knowledge/theme-color-chroma-validation-v2-pack.json', '--scenes', 'prompts/color-knowledge/theme-color-chroma-validation-v2-scenes.json', '--review-prompt', 'prompts/color-knowledge/theme-color-chroma-validation-v2-review.md')
& 'C:\Program Files\Python310\python.exe' -X utf8 -u @colourArgs verify @colourInputs
```

verify 退出码必须为 0、所有来源/hash/逐槽位 App 复算通过；否则停止，禁止联网。再依次执行，每一步检查退出码，失败不要无条件跑下一步：

```powershell
& 'C:\Program Files\Python310\python.exe' -X utf8 -u @colourArgs preflight @colourInputs
# 必须显示 planned_slots=6、planned_reviews=2、image_sends=0、review_sends=0。
# 检查 RUNTIME-FROZEN.json：预算6/2、同一套参数、自动重试关闭、无密钥值。
& 'C:\Program Files\Python310\python.exe' -X utf8 -u @colourArgs run @colourInputs
# 仅在未暂停、图片动作正常结束后执行：
& 'C:\Program Files\Python310\python.exe' -X utf8 -u @colourArgs review @colourInputs
& 'C:\Program Files\Python310\python.exe' -X utf8 -u @colourArgs evidence @colourInputs
& 'C:\Program Files\Python310\python.exe' -X utf8 -u @colourArgs report @colourInputs
& 'C:\Program Files\Python310\python.exe' -X utf8 -u @colourArgs status @colourInputs
```

失败或取消：保存已落盘的证据。`sending`／`unknown_after_send` 必须暂停后续联网，报告哪个槽位可能已经收费；不得自行 `authorize-resend`、清 ledger 或换目录重跑。明确失败也不得补图。报告渲染失败仅允许离线报告重建，不得重复生图或评审。成功缓存须复用，不能为改善结果再抽一次图。

## 观察与判定

1. 每组随机中性 ID，不向视觉模型泄露 A/B/C、清色／浊色或预期方向；保留绑定图、原始模型响应、映射、哈希和实际请求证据。
2. 比较蓝色绑定表面与红色强调区域的鲜明／灰浊程度。白底面积、明度、光线或物件变大本身不能证明清色／浊色效果；说明混杂因素，不能打审美总分。
3. 低饱和红仍可保留红色色相；粉红、鲑粉、米色、强调区域消失或串色须单独指出，不能自动算“浊色成功”。不要求精确 RGB、饱和度阈值或面积百分比。
4. 检查所有绑定表面；`region_id` 必须逐字保留英文绑定名，中文解释另写。遮挡／缺失仍要返回 uncertain 条目；漏项不能假设通过。
5. 检查人数、物件数、人物发色／瞳色／肤色／衣服／靴子、绿叶／棕盆、暖灰地面／桌布、天气光向、画法和可读性。未授权新增装饰必须报告。
6. 汇报三个独立数字：方向观察数、候选合规成功数、完整对照成功数。基准不合规或构图/光线混杂会降低对照可信度，不得因此抹掉实际方向观察。单次随机生成只能说明这些样本，不能宣称普遍有效。

## 禁止与交付

不重启/停止现有 app，不改生产代码、提示词、共享配置，不调用衣装或氛围，不发图像参考，不重绘，不做任何本地蒙版、羽化、图片叠加或上色。网页画廊只展示原图。不要执行生成的手动 replay 脚本，它会产生额外收费请求。

正常 replay JSON 已修复为完整脱敏认证头；环境变量名可以报告，密钥值不得输出。现有 GUI 仍是旧模块，不能声称其已热更新；只使用本次新启动的 CLI。

交付该目录的 `FROZEN-PACK.json`、`RUNTIME-FROZEN.json`、`ledger.json`、逐槽位请求/颜色计划/hash/原图、逐组原始及解析评审/中性映射、`RESULTS.json`、`RUN-REPORT.md`、`gallery.html`。补充 `run-notes/DS-SUMMARY.md`：实际发送次数、失败/未知/缺项、每个红色落点的保留情况、清色和浊色方向是否观察到、限制。不要自行给 hierarchy 填成功结论，也不要改上一轮证据和报告。
