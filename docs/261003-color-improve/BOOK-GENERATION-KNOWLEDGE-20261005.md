# 书籍知识到生图：首批离线交付

日期：2026-10-05。目标是固定色彩风格及其区域落点，不是提升画质。本次无收费请求、无新生图，不改生产 GUI、全局配置、画风固件及门禁，不覆盖旧实验。

## 本轮提取

- 案例色板 31 条：沿用原来源；色格顺序不推定主次，原作品、画师及故事不注入新图。
- 教学色板 28 条：PDF 16 的清/浊色调各 4 条；PDF 17 的多色相、对比色相、少色相各 4 条；PDF 18 的渐变 2 条、分离 2 条、主次 4 条。
- 共 59 条**来源记录**，未按色值去重，不能称为全书 59 套独立方案。全部新目录项的生图验证状态仍为 untested，不继承此前 P1/P2/P3 的晋级状态。
- 12 类简化 NCD 色调：PDF 14 的原名、代号、四大类和定义；不虚构 HSV 阈值，不混入 25 类完整版或无彩色代号。
- 面积模式：大基调/小强调、两色系等面积、室内三层比例。室内 60–70% / 20–30% / <10% 只允许 interior 场景；任何比例都不能通过增加物体、改构图或涂掉保护色来满足。
- 排列模式：渐变或分离，仅作用于授权区域，不授权更改光照、天气、布局或添加渐变背景。
- 既有 430 条知识保留为结构化检索资料；不整包注入，尤其不把绘画工序、故事与构图规则当作颜色合同。

## 证据边界

新增教学色板的色名来自页面转录中的视觉描述，不等同于色格上印有这些名称；HEX 全部留空。PDF 18 四个主次示例才有显式 base/accent，其他色板不猜角色。

案例色板继续区分扫描实测、NCD 标称和仅色名。扫描实测不是原稿准确值；不能声称已补齐全书色值。

PDF 59 的表头明确是「基调色 : 强调色」：7:3、9:1、9.5:0.5 是该案例分别对各强调色的比较，**不是强调色彼此的比例**，也不能直接相加为通用全画幅百分比。已单独存为不自动编译的案例证据。

PDF 74/147 的意象坐标图仍待逐组定位、去重和实测；其可视色板尚未计入 59 条。光效、材质、知觉等规则已有检索资料，但尚未完成逐条生成适用条件审核。本轮不称“全书知识已完成”。

## 生图落点

现有 CLI 新增 `generation-knowledge extract/compile`，纯离线。产物：

- `prompts/color-knowledge/book-generation-knowledge-v1.json`：59 条色板、12 类色调、编译策略、430 条检索规则与来源 hash。
- 同名 `.html`：离线目录，缺色值的格子不猜色。
- `book-generation-rules-v1.json`、`book-generation-contract-v1.md`：明确标为 derived 的生图指令与保护块。
- `book-generation-example-v1.json`：深蓝基调 + 橙色腰带/两只鞋蝴蝶结的显式区域绑定示例，不是已测结果。
- `data/test-result/20261005/book-generation-knowledge-v1/example.compiled.json` 与 `.prompt.txt`：完整可送生图的提示词及合同快照，尚未发送。

编译时必须声明实际生效的画风图片附件；非空即禁止配色选择。调用方必须从实际请求解析这个列表，不可拿界面选项或旧路径代替。该工具尚未接生产请求，不能声称 GUI 已可选择。

所有色格必须显式分配角色和现有区域，不允许静默丢掉颜色；授权区域不能与保护区域相交。关闭时正文原样返回。知识、模板、选项、色板和最终正文保存 hash；hash 是规范 JSON 文本的 SHA256，不是输出文件字节 hash。实际生图适配器仍需另存实际请求、附件及文件字节 hash。

区域存在性、色相是否相反及实际面积由调用者声明，本地编译器不具备视觉验证能力。仅做集合检查不等于身份/保护色审计已通过。

```powershell
python tools/color_knowledge.py generation-knowledge extract --out prompts/color-knowledge/book-generation-knowledge-v1.json
python tools/color_knowledge.py generation-knowledge compile --selection prompts/color-knowledge/book-generation-example-v1.json --out data/test-result/20261005/book-generation-knowledge-v1/example.compiled.json
python -m unittest discover -s tests -p "test_book_*.py" -v
```

## 下一步验证

先补图表色格的离线实测与人工核对，再选少量已绑定方案测试：固定同一内容锚、模型和参数，对照无合同、仅色板、色板加色调/面积规则。对每个方案使用预定轮次，不挑图。

判据分别记录色系、主次面积、区域落点、保护色、身份内容；新增色块和不可验证原因单列。画质不得充当配色成功证据。正式执行复用 `palette` / `cross` 与现有生成接口，关闭底层自动重试、保留请求/响应/账本，不能把此离线 compiled 快照冒充实际请求记录。

验证后再决定生产入口接线；不自动将检索规则或未测色板加入推荐默认值。

## 离线验证

本轮 `py_compile` 通过；`test_book_generation_knowledge.py` 16 条与既有目录回归 3 条合计 19 条全部通过。覆盖提取数量、缺色值不编造、显式主次、12 类色调、参考图禁用、保护区域冲突、比例适用场景、知识 hash 篡改、关闭原文保持与编译重复一致性。这不是在线生图效果验证。
