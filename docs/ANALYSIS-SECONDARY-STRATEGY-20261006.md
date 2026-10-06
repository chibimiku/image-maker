# 视觉观察 → DeepSeek 次级文本转换

用户授权：先 commit 备份，再将原视觉提示中的四类生成改写移到 DeepSeek；效果不合适可以回滚。

修改前工作区备份 commit：`9d3c6b4`。它包含当时已有的相似度实验修改及报告。
本策略另作独立提交。回滚时对本策略提交执行 `git revert <本策略提交>`，不要 reset 到备份，避免覆盖其他并行工作。

## 行为

- Step 1 使用原先选择的视觉通道，忠实描述可见事实。保留照片媒介、真实服饰术语、模糊、遮挡和不确定性；不强制 girl、不补造五官、不凑字数。附加生成方向只进 Step 2。
- 原来的 Step 2 复用为次级转换，不另加一轮请求。单图、批量和完整无头分析都使用设置中的 NSFW 文本 API（本机配置为 DeepSeek），仅发送 Step 1 文本，没有图片附件，也不做视觉模型替换。
- 四类生成改写在 Step 2 模板中：通用清晰面部绘制目标、女性角色称呼、精确措辞与适度扩展、照片转插画及 Lolita 服饰改为 rococo-inspired fashion 表述。明确成年仍称 adult woman，年龄不明且敏感时称 female character；不改年龄、真实遮挡、服装覆盖和身份，不编造隐藏瞳色或五官。
- `original_english_description` 与新增 `source_analysis` 保留原文及原图标签；改写内容在 `english_description` / `short_description`，并标记 `analysis_strategy=observe-then-secondary-edit/v1` 与 `secondary_text_model`。最终作品标签仍沿用原图标签。
- 次级配置缺失或请求拒绝时停止，不静默发回视觉主模型。原有 Step 2 请求/提示词 hash/Step 1 快照仍保留；拒绝不重复提交同一请求。
- 单图与批量页默认执行 Step 2；批量页显示「DeepSeek 二次加工」。手动关闭二次加工（`enable_refine=False`）时仍只保留事实观察，不执行四类转换；`analyze_image_step1` 也是事实分析入口。
- 修复 Step 1 的元组响应软拒绝漏检；拒绝 JSON 不能被当成功。正常描述中的 safety pin / unknown 不再被宽泛子串误判为拒绝。

## 验证与边界

新增 `tests/test_analysis_secondary_strategy.py`：9 项离线 unittest 通过，覆盖主次客户端路由、纯文本请求、原文与遮挡保留、四类模板、缺失配置、次级拒绝、主端点软拒绝及同请求备用重试、正常描述不误判。
既有 `test_analysis_cli_unittest.py` 的7项通过，其中原有用例尝试连接图片服务但被沙箱阻止，没有成功请求；本策略用例全用假客户端。修改模块 py_compile 与导入检查通过。系统 Python 未安装 pytest，pytest 用例未执行。

没有在线复测拒绝率，因此不宣称拒绝问题已解决。没有重启 app 或修改用户端点配置。prompt 文件按请求读取，而常驻 GUI 的 Python 路由需空闲时重启后加载；重启前新旧部分可能并存，不应将其当新策略效果测试。
