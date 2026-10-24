# GPT Sunburst 默认模型与按附件路由（2026-10-15）

用户明确要求默认采用 Sunburst，并根据输入调整 GPT 系列请求路径。本次只做配置和代码接入，没有新增收费出图或文本审计。

## 生效行为

- new.aigc2d 的 API 默认与 GPT 生图页保存模型改为 gpt-image-2.5-sunburst；省略模型的后台/CLI 使用配置，不再被函数签名里的 Image 2 固定默认覆盖。
- 显式模型选择继续生效；已登记仅支持 Image 2 的 autodl 默认保留 Image 2。历史实验及断点中记录的模型不改写。
- 所有 GPT Image 请求在发送前按最终有效附件选择路径：有内容图或画风图走 /images/edits（multipart image[]）；无图走 /images/generations（JSON）。即使旧调用传 mode=generate，也不向 generations 塞入非标准 image 字段。
- 显式 edits 且无有效图片仍中止；删除参数报错后切换接口重发的兜底。参数/认证等终止性 HTTP 错误不重复 POST，原有配置的暂时性网络/上游重试未全局调整。
- GPT 生图页、分析首图、单图调试与两条 CLI 对齐；分析页默认显示自动选择接口，旧保存的 generate 选择回到自动。
- 单图调试保留共享构造器产出的完整提示词及附件，让实际后台路由看到画风图。
- 画质默认保留 high。2.5 可显式选 xhigh/max，后台不再过滤掉它们；旧模型使用这两档会在发送前报错。xhigh/max 的 UI 费用显示尚未校准，避免误用 medium 的 token 估算；普通档位估算使用当前选择的模型。
- 官方 GPT Image 请求不发已不支持的 response_format；openai-image 根地址正确解析到 /v1/images/*。

## 配置修改

采用读旧、合并指定键、原子写回；没有写入或更改凭据、地址、超时、尺寸和质量。三个变更键：

1. apis.aigc-2d-gpt.model = gpt-image-2.5-sunburst（原节点由环境定义，这个磁盘键覆盖模型，地址/密钥仍按原环境配置合并）。
2. gpt_image2.sites.new.aigc2d.model = gpt-image-2.5-sunburst。
3. analysis_gpt_pipeline.first_pass_mode = auto。

实际变更清单：data/test-result/20261015/sunburst-default-routing/config-changes.json。运行时 get_api_config 已核对为 Sunburst。

## 离线验证与限制

- 五组相关测试：239 passed、1 deselected、6 subtests passed。包含各模型有图/无图路由、max 不丢失、终止性错误单次发送、显式模型选择、CLI 预检、UI 与提示词优化。
- 最后费用显示修改后相关复验：31 passed；新增 max 费用提示测试 1 passed。
- 12 个 Python 文件编译、app 导入、指定修改文件 git diff --check 通过。
- 扩展检查曾为 232 passed / 1 failed：test_analysis_worker_direct_has_no_pipeline_or_audit 预期返回原输入路径，但既有交付逻辑产出 final 路径。显式恢复旧 Image 2 模型仍复现同一断言，原图与新 final 字节一致；不把这一项算通过，也没有为本任务修改交付逻辑。最初临时目录中的假 .env 原子写测试遭沙箱 WinError5，改用工作区临时目录后该测试通过。
- 仅 Qt 离屏检查，非桌面交互验收；没有本次修改后的生产整链线上验证。前面三图比较属于本会话独立实验，不能当成所有新入口的完整生产验收。

证据位于 data/test-result/20261015/sunburst-default-routing/。未 commit，未重启运行中的 app；重启后加载新代码和默认配置。

## 官方依据

- [Generations 参数](https://developers.openai.com/api/reference/resources/images/methods/generate)：Image 2 与两款 2.5 均支持；没有 image 输入参数。
- [Edits 参数](https://developers.openai.com/api/reference/python/resources/images/methods/edit)：图片输入与两款 2.5、xhigh/max。
- [模型选择与迁移提示](https://developers.openai.com/api/docs/guides/image-prompting)：Sunburst 注重质量，Flare 注重速度；相同档位不保证相同成本或效果。


## 同日追加：用户明确改为 max

用户随后要求默认画质改为 max。当前默认已是 Sunburst/max，覆盖 API 节点、生图页保存项与分析首图设置。单图调试和省略质量的两条 CLI 读取同一配置；显式选择其它质量仍保留。Image 2.5 的无配置默认也为 max，旧模型无显式质量时使用 high，不继承仅 2.5 支持的配置 max。

三张对照的输出 token 都为5488，所以本轮估算几乎同价；这不是固定单张价格承诺。实际费用仍随输入/输出token变化，max费用未校准提示继续保留。

追加验证：三组223 passed；5个运行时Python文件编译、app导入、运行时Sunburst/max与旧模型默认high检查、diff-check通过。无新增收费请求，未重启。配置差异另存 max-default-config-changes.json。
