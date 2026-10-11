# 图片审计中英文展示与手动 refine 交付修复

所有生产图片审计模板（首图、身份、refine 保真、质量/终审、人体、重绘内容及衣装）在同一次 JSON 响应保留原英文 schema，并增加 `display_zh.summary` 与 `display_zh.details`。中文完整说明包含区域、证据、修复建议、容差及限制；只用于显示。英文原结论、置信度及列表继续用于门禁和纠正 prompt。规范化保留中文对象，缺失中文的历史结果回退英文，不触发翻译调用。首图代码调整模型判定时，中文摘要同步注明代码最终裁决。历史 refine 审计断点补跑只追加双语输出要求，原审计规则不替换，原文件不改写。增加输出 token 上限并同步 refine/内容审计费用预估，无额外翻译请求。

手动采用/重画/复审原先只调用 refine，并直接显示 `gpt_image_repaint` 内的中间图。现在在后台线程检查新记录已 finished 后，复用原清单保存的本地后处理参数，在新候选上运行 structure/contrast/tone/ink；不会再次处理旧 final，也不会重复 refine/衣装或追加收费调用。无本地工序时直接导出 final-refine PNG 与关联清单。未完成仍保留断点，不声称最终交付。

新流水线清单保存本地后处理 recipe。历史没有参数的清单只恢复已记录 ink 的历史 GUI 默认 target_sep=8、amount=1、max_darken=40、sigma=6、protect_light=0；其它未记录工序不猜参数，日志明确说明。最终输出保持与 refine 输入/候选的精确路径关联。

状态栏手动操作立即离开完成状态，定时读取正在请求/重试的工序，显示背景/衣装/整体修复及审计；末段显示提线/最终输出，异常或未通过保持未完成提示。按钮一直受 worker 生命周期约束。

指定任务：`gptimage2_output_184245_0_16b045-184410-c94184-final-ink.png` 的原背景审计失败，人工采用后子记录 `refine-65d56037decf-refine.json` 已完成衣装、整体审计。此次仅补本地 ink，输出 `data/20261011/refine-65d56037decf-whole_184853-fe0970-190222-1be15a-final-ink.png`，1696×2528 PNG，已完整解码及查看。没有新增图片/文本收费请求，原图、候选、历史审计和原清单不覆盖。

证据：`data/test-result/20261011/bilingual-audit-refine-delivery/recovered-delivery.json` 与 `pytest.txt`。双语响应仅离线 schema/显示回归，未发新线上审计请求；UI 用离屏 Qt 回归，正在运行的桌面 app 需重启才能加载修改。未 commit 或重启。

验证：12 组相关离线回归 379 passed、7 subtests passed；13 个 Python 模块编译与 git diff --check 通过。同步了此前审计异常重试导致费用上限由 5→17 的旧测试断言，以及历史断点新请求追加双语要求的断言。未运行全套，未线上验证双语响应完整度。
