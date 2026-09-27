# sheya Gemini 内容泄露修复

样例：`33fb144d_222944-2abc24.jpg` 几乎复刻 sheya.png；`94c2ddf6_222012-81eb12.jpg` 改姿势却仍复制短蓝发角色、服装、水面与骰子。两份分析要求棕发、闭眼笑、海军蓝洛可可裙、黑手提包与木门场景。

旧 sheya prompt / prompt_compressed 将漂浮姿势、幻想服装、玻璃鞋等内容误作画风规则，且 motif_enabled 开启骰子母题。已将 Gemini 规格改为纯绘画技法，关闭母题，在 priority 模式使用完整 gpt_image_prompt 内容锚。配置定向同步到画风子模块，仅更新 sheya 的五个字段。

分析 Tab 带画风参考图的 Gemini 请求，在图后追加本次明确内容与优先级，模板为 prompts/gemini-analysis-content-lock.md；无图请求与 GPT 链不受此补充影响。原图与分析文件保留。

验证：py_compile、GUI 导入、四种参考模式、两份原分析的内容锚及图后内容断言通过。系统 Python 缺 pytest，未运行 pytest。两次真实出图请求均接口失败，无新图，不能宣称视觉效果已解决。请求快照位于 data/test-result/20260927/sheya-content-fix/。修改 Python 后须重启 GUI。
