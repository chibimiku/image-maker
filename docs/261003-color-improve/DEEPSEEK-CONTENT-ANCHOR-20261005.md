# DeepSeek 执行指令：内容参考图锚定固定配色

请直接执行已冻结的参考图实验，不再制作预算清单，不核对账单。用户要求将这轮交由你执行；会有 API 消费。仅上传本协议指定的已生成内容源及本轮候选图到已配置的 `new.aigc2d.com` 生图/视觉评审端点，不能上传 PDF、整个项目目录、配置、密钥或其他资料。

先读项目根目录 PROJECT_REQUIREMENTS.md、AGENTS.md，再读 docs/261003-color-improve/BOOK-KNOWLEDGE-INTEGRATION-20261005.md。复用 tools/color_knowledge.py 的 generation-knowledge 子命令及 utils/color_experiment.py，不创建一次性脚本、不改生产 GUI/全局配置/画风或质量身份门禁。

独立输出目录：data/test-result/20261005/book-content-anchor-v1/。已有 book-frozen.json、preview/*.request.json 和仅含内容源的离线页，全部是预检，尚未生成。不得覆盖此前 book-palette-ablation-v1/v2 的协议、产物或报告。

协议：prompts/color-knowledge/book-content-anchor-v1.json。
冻结 hash：590133a50bc0e745bb49dff974fde5ec8c4e0fa52e2430176418df43090130d9。
唯一内容源：data/test-result/20261005/book-palette-ablation-v2/samples/C13-simple-r1/book-C13-simple-r1_105300-2c8d70.jpg。
源图 SHA256：63a39e46a2ece312a2f703b2a7561f5c285d3d598f1c565a20c053cdbc97a6c1。
它是按旧排程选出的最早单人、连续画幅、无色卡样本，不得按美感换图。参考图只提供内容，绝非画风参考图。

计划 8 个逻辑样本：C13/C16 两套固定色板 × text/content 两分支 × r1/r2。模型、参数与排程均冻结：Gemini Flash、1K、3:2。text 只发文字；content 发同样色板/区域合同及这一张内容源，明确保持原图人数、人物身份、姿势、相对位置、互动、服装/鞋设计、物体和取景。保持单一连续画幅，禁止分栏、重复人物、对照图、文字色卡。

辅助色只作用于人物脚下前景干岸地，不包括对岸、草和花。对岸地面、近景植物及小花保持源图/文字原色；远景植被仍服从基色。不能沿用旧 v1/v2 全岸地判据。本轮不添加新材质、局部 NCD 色调规则或其他知识模块，以免混淆参考图效果。

执行时始终 IMAGE_MAKER_IMAGE_MAX_RETRIES=0。只发计划请求；成功产物复用，失败、拒绝、不可验证与内容失败保留，不补抽、不换图、不修改提示词重跑。未执行项可用同一普通 run 续跑，不使用 --retry-failed 追加请求。若请求/身份保护冲突或冻结 hash 不一致，停下说明，不关闭门禁或偷偷重冻。

用项目规定系统 Python，命令不套 PowerShell 管道或重定向：

```powershell
$env:IMAGE_MAKER_IMAGE_MAX_RETRIES="0"
& "C:\Program Files\Python310\python.exe" -m unittest discover -s tests -p "test_book_*.py"
& "C:\Program Files\Python310\python.exe" -u tools/color_knowledge.py generation-knowledge preview --spec prompts/color-knowledge/book-content-anchor-v1.json --out data/test-result/20261005/book-content-anchor-v1
& "C:\Program Files\Python310\python.exe" -u tools/color_knowledge.py generation-knowledge run --spec prompts/color-knowledge/book-content-anchor-v1.json --out data/test-result/20261005/book-content-anchor-v1
& "C:\Program Files\Python310\python.exe" -u tools/color_knowledge.py generation-knowledge review --spec prompts/color-knowledge/book-content-anchor-v1.json --out data/test-result/20261005/book-content-anchor-v1
& "C:\Program Files\Python310\python.exe" -u tools/color_knowledge.py generation-knowledge report --spec prompts/color-knowledge/book-content-anchor-v1.json --out data/test-result/20261005/book-content-anchor-v1
```

评审每组一次，参考分支必须真正带原内容源，不能只读文字。候选三视图顺序为完整彩色图/同图身体裁剪/同图灰度代理，不是三个人或三张样本。灰度代理仅作可辨性证据，不能替换/修改实际产图。检查配色与落点、主辅点缀面积、保护色、实际人数、单画幅/色卡、源构图与姿势、灰度轮廓可辨性；肤色/发色保护与人数问题分开。text 分支没有精确参考真值，source_count_ok=null、source_layout_status=unverifiable；仍需记录 visible_person_count。候选可用必须实际人数=1；content 另要求源人数与源布局明确通过。

逐张完整查看实际图，不仅采信视觉模型 JSON。错误评审用绑定产图字节 hash 的 adjudications.json 单独更正，保留 model_review/原始请求/原始响应，不覆盖证据。没有明确区域边界就记录 partial/unverifiable，不用美感代替固定配色判断。

交付独立执行报告（详细版另写 EXECUTION-REPORT.md，RUN-REPORT.md 会被 CLI 重生成）、book-per-image.json、book-summary.json、冻结协议与请求 hash、内嵌源图/全部实际产图/灰度证据的 evaluation.html、脱敏请求与原始响应、调用账本。失败不从分母消失，不把预留槽当收费次数；本轮结果不与旧全岸地范围直接比较。完整布局复核失败不能称为保持身份/构图成功。

完成后停下汇报，不自动进入修图、追加样本、材质试验或生产接线。
