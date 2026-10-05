请在 D:\code\image-maker 执行独立穿衣风格 Round5。Codex 已修正候选提示词分支并冻结请求，你只负责跑图、保存结果和记录执行情况，效果由 Codex 检查。

先读 PROJECT_REQUIREMENTS.md、AGENTS.md 和 docs/style-extraction/WARDROBE-ROUND5-READY-20261005.md。使用 prompts/wardrobe/round5/generation-pack.json，共 28 个请求（GPT 14、Gemini 14），每槽一次。正文、衣装快照、画风参数、参考附件及顺序均已冻结。不要改词、换模型、加模板、重新提取、调用评分/文本模型、重绘或发布。无需重启 app，不要中断现有任务。

本轮重点：GP/GF 检查 Lolita 的原衣装保留；LP/LF 检查内衣目标的保留；HA/LA 检查保留类别改款；UC/HC 检查完整 Lolita；LC/LU 检查成人内衣完整转化与无衣装描述补全。四组画风对照的正文、参数完全一致，只差画风图附件，不能自行改成别的参考模式。这是首图效果实验，并非完整审计/发布链路。

系统 Python C:\Program Files\Python310\python.exe，工作目录 D:\code\image-maker。凭据沿用 conf/config.json + .env，不输出密钥。Python 命令禁止管道和 shell 重定向。先零 HTTP 预检：

```powershell
& 'C:\Program Files\Python310\python.exe' -u tools/wardrobe_experiment.py --frozen-pack prompts/wardrobe/round5/generation-pack.json --run-dir data/test-result/20261005/wardrobe-round5-r1 --prepare
```

校验通过后顺序跑两通道（不并行启动）：

```powershell
& 'C:\Program Files\Python310\python.exe' -u tools/wardrobe_experiment.py --frozen-pack prompts/wardrobe/round5/generation-pack.json --run-dir data/test-result/20261005/wardrobe-round5-r1 --generate --channel gpt
& 'C:\Program Files\Python310\python.exe' -u tools/wardrobe_experiment.py --frozen-pack prompts/wardrobe/round5/generation-pack.json --run-dir data/test-result/20261005/wardrobe-round5-r1 --generate --channel gemini
```

个别图片失败时记录原因，继续另一通道；包校验/账本冲突则停止。成功产物复用，失败和超时占槽位，不自动重试，也不换输出目录重置。如果用户主动追加额度，属于有效授权：保留账本，记录授权与具体失败 ID，然后使用工具现有 --grant-frozen-budget generation / --grant-limit / --authorized-by / --grant-reason / --grant-source / --retry-slots 参数补跑。先运行 --help 核对当前参数名。只重试获授权的失败项，不重发成功项；授权后不必再重复请求同一权限。

完成后 --status 检查，将报告写到 docs/style-extraction/WARDROBE-ROUND5-GENERATION-RESULTS-20261005.md，列明实际调用数、成功/失败/未知/未尝试数量和失败 ID、追加授权与每次尝试。保留 frozen-pack.json、samples.json、send-budget/ 和全部图片。“接口成功”不等于“服装效果通过”。回复报告路径，等待 Codex 看图。
