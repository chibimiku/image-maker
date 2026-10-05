# Round3 启动检查（2026-10-05）

修正 DeepSeek 准备材料后，离线检查通过。尚未启动生图。

- 控制臂改回真实 Round2 v3 的保留规则和明确上胸裁剪；移除把 v2 缺陷误称为 v3 的描述。T06 是新增近景改款用例，控制规则做了最小授权适配，不声称它已在 Round2 验证。
- 校准 CAL-CROP 本地预期改为 A 较宽、长衣身接近腰部，B 较紧上胸；不声称独立生图是同一身份。
- 视觉预算仍为 20：3 校准 + 16 对照评分 + 1 预留；生成 32、文本 2、模型列表 1 上限均不变。
- 修复审查工具沿用 Round2 专有 prepare 字段造成的 KeyError，以及旧 Round2 审查说明。ready 必须为布尔 true 且没有 blocking。图片 data URL 按实际 JPEG/PNG 编码标注。spec 指纹覆盖 variants 与 characters 模板。
- 重新组装 12 快照、32 槽位，A/B 控制各 8 对；prepare 成功。spec SHA256：9f30cc71d252848f4cd55728c8e46898fdb204e1c8fe3fe862a98d99c11c3f17。

在线审查第一次尝试遇到 WinError 10013，未收到模型回复；账本保守计作 1 次文本请求。随后提升网络权限被自动审批拒绝，未再发 HTTP：审批认为向 new.aigc2d.com 发送内部实验计划、人物和提示词缺少对该目的地的明确授权。不得绕过拒绝。

当前剩余：文本 1、视觉 20、生图 32、模型列表 1。不得换 run-id 重置预算。用户明确允许目的地后，沿用 wardrobe-round3-prep，依次单独运行 --review、--pick-vision-model、--calibrate、--gate；门禁通过后 --generate、--pairs、--score。不要使用 --run 重复审查和校准。生产预设与配置未改。

实验目录：data/test-result/20261005/wardrobe-round3-prep。材料：prompts/wardrobe/round3/spec 与 pack。
