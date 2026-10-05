请在 D:\code\image-maker 完成 Round4 的生图执行。Codex 已完成代码、UI、规则、素材修正和请求冻结，你只负责调用图片接口并保存结果。

先阅读 PROJECT_REQUIREMENTS.md 与 AGENTS.md，再阅读 docs/style-extraction/WARDROBE-ROUND4-READY-20261005.md。本任务使用 prompts/wardrobe/round4/generation-pack.json，不使用旧 requests/ 草案、旧 spec 的 --run，也不用 DEEPSEEK-PREP-TASK.md。不要再做筛样、提取、编译、提示词改写、看图评分、模型探测或代码修改。

运行环境：系统 Python C:\Program Files\Python310\python.exe，工作目录 D:\code\image-maker。命令不能套管道或 shell 重定向。端点和凭据沿用 conf/config.json + .env 的现有配置；不得打印、复制或写出密钥。若网络沙箱拦截，对同一命令申请正常联网权限，不改端点绕过。

本次执行冻结包的 40 个生图槽位，每槽一次，GPT 20 次、Gemini 20 次。文本／视觉评价／模型列表请求均为 0。失败和超时占用槽位，不自动重发。若需要补跑或增加额度，先向用户说明具体失败槽位并取得追加授权，保留本轮账本；用户主动追加授权是有效授权，不应当作程序故障。当前工具不提供失败槽位重发，获得追加授权后交回 Codex 安排，不能擅自换输出目录重置。

先进行零 HTTP 预检：

```powershell
& 'C:\Program Files\Python310\python.exe' -u tools/wardrobe_experiment.py --frozen-pack prompts/wardrobe/round4/generation-pack.json --run-dir data/test-result/20261005/wardrobe-round4-r1 --prepare
```

退出码为 0 后，顺序执行两条命令，不要并行启动：

```powershell
& 'C:\Program Files\Python310\python.exe' -u tools/wardrobe_experiment.py --frozen-pack prompts/wardrobe/round4/generation-pack.json --run-dir data/test-result/20261005/wardrobe-round4-r1 --generate --channel gpt
& 'C:\Program Files\Python310\python.exe' -u tools/wardrobe_experiment.py --frozen-pack prompts/wardrobe/round4/generation-pack.json --run-dir data/test-result/20261005/wardrobe-round4-r1 --generate --channel gemini
```

第一条若因个别生图失败返回非零，先保存原因，第二通道仍可正常执行；若是冻结包校验或账本冲突，停止并报告，不修改包。中断后运行同一条命令即可续跑未尝试槽位，已有成功产物复用，已失败／未知槽位跳过。该模式不进行自动重绘、后处理、身份修订或发布；这是用于检查首图效果的实验。

完成后查看状态：

```powershell
& 'C:\Program Files\Python310\python.exe' -u tools/wardrobe_experiment.py --frozen-pack prompts/wardrobe/round4/generation-pack.json --run-dir data/test-result/20261005/wardrobe-round4-r1 --status
```

交付保留图片、frozen-pack.json、samples.json、send-budget/。在 docs/style-extraction/LOLITA-WARDROBE-ROUND4-GENERATION-RESULTS-20261005.md 写实际成功／失败／未知／未发送数量，按通道列出失败槽位及错误，给出输出目录和实际调用数。不要把“生成成功”写成“服装效果通过”。不要改模板或发布到正式作品目录。跑完回复“Round4 图片执行完成，待 Codex 检查效果”，并附报告路径。
