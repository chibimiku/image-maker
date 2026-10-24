# GPT 出图独立配置与默认直出 — 2026-10-13

2026-10-14 更新：配置入口已移动到「设置 → GPT 出图 API」；Gemini 首图审计现在独立控制、默认关闭，GPT 模式显示未勾选且不可用。最新界面与 tooltip 修复说明见 `PUBLISH-MISSING-FILES-AND-GPT-UI-20261014.md`，以下保留本轮最初交付记录。

用户要求采用本会话验证过的 openai-gpt 专用令牌，独立保存；默认关闭 Gemini 重绘与校验等工序，并接入 GUI。随后要求生图区单独配置入口，支持「和 Gemini 一样」勾选及字段置灰。

## 界面与运行行为

- GPT 生图/编辑页、单图分析的 GPT 通道、单图调试生图均提供「GPT配置」入口与「后续工序」开关，共享设置并同步刷新。
- 「GPT配置」对话框默认不勾选「和 Gemini 一样」，可独立填写接口 URL 和 key。key 使用密码输入，不显示已保存值，留空保留已有专用 key。
- 勾选「和 Gemini 一样」后 URL 与 key 字段置灰，运行时动态读取 `apis.aigc2d` 的统一配置。Gemini 的 `/v1beta/models/` URL 转为 GPT 对应的 `/v1` 根路径。GPT 模型仍使用生图页面原有模型选择。
- 取消共享后恢复独立 URL 与 key，不把统一 key 复制到独立存储。
- 已将用户提供的专用 key 安装到独立变量 `IMAGE_MAKER_GPT_IMAGE_API_KEY`，位于 Git 忽略的 `.env`。本说明、源码与测试没有保存真实 key。
- `conf/config.json` 的 `gpt_image_workflow` 节点只保存共享方式、独立 URL 和后续工序开关；当前 `same_as_gemini=false`、`postprocess_enabled=false`。
- AIGC GPT 图片后端、模型列表、主窗口 GPT 节点以及分析/调试入口的 key 预检均读取独立配置。Gemini、文本分析和 autodl 不使用此专用 key。

## 默认关闭工序

新 GUI GPT 请求冻结 `postprocess_enabled=false`。首图即最终输出，跳过 Gemini 重绘/refine、身份/人体/衣装/质量/最终审计、首图审核后的文字复核，以及本地色调/加墨等后处理。不会把“未审计”记作审计 pass。

相关子工序控件保留用户已有勾选配方，但在总开关关闭时置灰且不参与请求与预算。打开「后续工序」后按页面具体勾选执行。手动 Gemini 重绘、refine 重试/续跑、旧分析断点续跑同样需要先打开总开关。

旧记录、图片和历史审计不改写；历史工作流只有显式操作才继续。未配置新工作流节点的旧 CLI 配置继续使用原节点 key，以免破坏既有历史配置；此机器已配置新节点，默认使用独立 key。

代码接入完成后需要重启应用加载新模块。本轮未重启或提交。

## 验证与边界

- 7 组主要离线回归：253 passed / 1 deselected。排除项为已有未提交“文件时间 tooltip”改动导致的 `test_result_list_never_widens_the_page` 旧等值断言；该失败已实测，未回退原改动。
- 专用配置、新默认直出 UI 与密钥注入补验：33 passed / 4 deselected；4 个完整设置页构造测试未在补验中启动。两组含重复用例，不相加作为独立测试总数。
- 新测试覆盖默认直出、独立与共享 key 分离/切换、共享字段置灰及恢复、密码输入、保留其他配置、缺专用 key 不偷偷回落、关闭工序后不执行文字复核或隐藏审计，以及旧配置兼容。
- 8 个实现模块编译检查、`git diff --check`、`.env` 忽略规则通过。
- 实际运行时核实 GPT key 来源为 `env:IMAGE_MAKER_GPT_IMAGE_API_KEY`，与 Gemini key 不同；独立 URL 为用户现有 AIGC 地址，后续工序关闭。
- 离屏 Qt 保存了中文配置对话框和生图区截图，验证共享字段禁用及有效工序为空；不是运行中桌面应用的交互验收。独立页截图会按既有页面最小高度扩展，未宣称 1100×750 整页验收。
- 无新增图片生成请求。早期完整设置页测试触发了旧采集页后台线程，使用测试 key 返回 401；之后回归在进程级禁止真实 socket 连接。未查询计费账单，不把测试错误当作真实收费成功。

证据目录：`data/test-result/20261013/gpt-independent-workflow/`，包含 `regression.xml`、`key-regression.xml`、`ui-check.json` 和离屏截图。
