# millon-knots：12 图、5 轮画风提取（2026-10-04）

已完成原 App 的多图画风提取流程，采用第 5 轮 Gemini 直出的已测试提示词包。不是模型权重训练，也不能据此宣称精确复现或所有主体均稳定。

## 调用链与参数

`app.py` 创建 `StyleAnalyzerWidget`，本次无头直接调用同一 `StyleIterativeWorkerThread.run()`，未修改 App 业务代码。文本配置及密钥用 `api_backend.load_config()` / `resolve_text_api_key()`；用户明确授权后调用 `https://new.aigc2d.com`。文本模型来自配置（`gpt-5.6-luna`），图片节点来自配置（Gemini / GPT-image）。

- 数据源：`C:\Users\ashsu\Downloads\千千結 - pixiv`。从 142 个文件中浏览挑选 12 个完整作品，避开重复裁剪、线稿和对照页。
- 独立目录：`data/style-datasets/millon-knots/`；保留原文件名，原下载文件不改动。
- 5 轮，每轮全部参考图对账 + 随机抽查 3 图；随机种子 20261004。
- 测试主体沿用配置：`a girl sitting beside lake, she has pink hair and brown eyes, spring,`。
- 每轮固定同一参考图 `100158181_p0.jpg`，比例自动跟随参考图。
- 最后全量参考图裁成 60 个区域、15 批跨图局部分析、合并、全量终审和最终 Prompt 包，共 37 条迭代记录。

## 实际出图情况

| 轮次 | Gemini 直出 | GPT 首图 | GPT→Gemini |
|---|---:|---:|---:|
| 1 | 1 | 1 | 1 |
| 2 | 1 | 1 | 1 |
| 3 | 1 | 1 | 1 |
| 4 | 1 | 1 | 1 |
| 5 | 1 | 0 | 0 |

合计 13 张。第 3 轮 GPT generations 请求反复报 `Unknown parameter: image` 后，项目现有逻辑转 edits 端点成功；这不是与其他轮完全相同的首图请求方式。第 5 轮也遇到该错误，edits 回退最终遇到 `The request rate limit has been reached`，无 GPT 首图和重绘图。收费金额没有核验。

## 选择依据与限制

逐轮评估曾偏向第 1 轮；统一跨轮评估将所有 13 张图与全部 12 张参考图同时比较，选第 5 轮 Gemini。采用这一统一比较结果并人工查看产物。分数为主观定性量表，不能当相似度百分比，排序有不稳定性。

第 5 轮保留粉发、棕眼和湖边坐姿；线条、块面和场景层次较协调，但仍偏通用柔和插画，未充分复现参考作品的强对比、流动装饰和复杂材质。第 2 轮 GPT 两图、第 3 轮重绘明显复制参考角色角饰/服装，因此排除；第 4 轮 GPT 图还把粉发改成深发。

保留终审后母版和 Prompt 包，但它们未经单独生图验证，所以不冒充已测试的最终版本。运行配置取第 5 轮 `prompt_variants.style_entry`，附完整 Gemini 描述、571 字符 GPT 短版、7 条重绘条款、633 字符人工压缩版和原参考图。自动 LLM 压缩两次都超长，未把超长结果写入最终配置。可选装饰母题保留在配置，但 `motif_enabled=false`。未自动开启额外五官修订或绕过质量/身份门禁。

## 文件与配置

- 素材清单及预览：`data/20261004/style-extraction/millon-knots/selection.json` / `selected-references.jpg`。
- 全流程断点：`millon-knots_style_iter_result.json`；`finish.json` 为 `success`。
- 逐轮视觉评估：`round-01-evaluation.json` 至 `round-05-evaluation.json`。
- 同一请求跨轮比较：`cross-round-evaluation.json`；选择记录：`selection-result.json`。
- 可离线对照页：`comparison.html`；总览：`five-round-comparison.jpg`；选中图：`best-output.jpg`。
- 独立配置包：`best-style-entry.json`。
- 运行配置和子模块配置各增加 `millon-knots` 一项；只合并此键，其他条目逐项核对未变。
- 画风参考图：`data/style-ref/millon-knots.jpg`。
- 原画风配置备份在本次输出目录的 `config-styles-before-runtime.json` / `config-styles-before-versioned.json`。

启动或重启 App 后即可从画风列表选择 `millon-knots`。本次没有改 Python 业务代码，也没有运行无关测试；校验两份 JSON 一致、旧条目保留、参考图存在、GPT 短版合法、压缩版不超过 700 字符。

子模块提交：`87ed12fe742d792459794712e4f13401dff30943`。暂存内容由子模块 HEAD 加入本次唯一新增键构建，原有未提交画风改动仍留在工作区。

## 2026-10-04 补充：终审版本的三路测试

此前第 5 轮 Gemini 已在原选择之前测试，未测试的是局部细化与终审之后修改出的最终母版。现已补跑终审 Prompt 包：Gemini 直出、GPT 首图、GPT→Gemini 重绘各成功 1 张。`final_test_images` / `final-test-result.json` 记录实际最终提示词包和三路产物，产物目录 `test-generations/final/`，对照图 `final-version-comparison.jpg`，原对照页也已增加终审版测试区。总产物数现为 16 张；生图成功并不代表风格或人体质量自动通过。原配置仍为已选择的第 5 轮包，没有未评估就自动替换成终审包。

App 已补上终审后测试步骤。Gemini 原本每轮都在运行；新测试发生在最终 Prompt 包派生之后，结果单独保存并显示。空图标失败，未启用测试标未测试，部分失败在界面显示错误。修复 Gemini 配置失败时局部变量未初始化导致 GPT 测试连带失败的问题。14 条回归用例通过（指定 Python 缺少 pytest，使用标准库 unittest runner 执行现有测试函数与 unittest.mock 注入）；编译及导入通过。必须重启 App 加载新模块。

人工查看终审补测：Gemini 保留粉发棕眼和湖边主体；GPT 首图发色错误；GPT→Gemini 重绘复制参考角色角饰、服装及构图，不可用于晋升最佳画风。三路 success 仅代表成功出图，不是风格/身份门禁通过。
