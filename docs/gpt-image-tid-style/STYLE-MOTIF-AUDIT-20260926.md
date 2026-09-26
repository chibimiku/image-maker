# 画风参考图可选装饰母题审计（2026-09-26）

母题只在首次生成时以低权重加入；必须与用户场景兼容，只能少量出现在背景或画面边缘。Gemini/GPT 重绘不使用母题，避免把新增装饰误当成源图内容。

| 画风 | 显示 | 启用母题 | 审计结论 |
|---|---:|---:|---|
| `默认(无附加)` | 是 | 否 | no reference image |
| `puracotte-style` | 否 | 是 | tiny star-shaped sparkles；small jewel-like glints |
| `puracotte-style-v2` | 是 | 是 | tiny star-shaped sparkles；small jewel-like glints |
| `puracotte-v3` | 否 | 是 | tiny star-shaped sparkles；small jewel-like glints |
| `tinkle-style` | 是 | 是 | a few blue rose petals；small crystalline shard accents |
| `fuzichoco` | 否 | 是 | small floating picture-frame borders；a few paper-like bird silhouettes |
| `fuzichoco-v2` | 是 | 是 | small floating picture-frame borders；a few paper-like bird silhouettes |
| `dall-e` | 否 | 是 | small floral clusters；a restrained crescent or halo ornament |
| `dall-e-v2` | 是 | 是 | small floral clusters；a restrained crescent or halo ornament |
| `dall-e-v3` | 否 | 是 | small floral clusters；a restrained crescent or halo ornament |
| `satou_kuuki-style` | 否 | 否 | disabled: reference evidence is identity/garment/scene-specific or too weak |
| `satou_kuuki-style-v2` | 是 | 否 | disabled: reference evidence is identity/garment/scene-specific or too weak |
| `waterink-style` | 是 | 是 | sparse ink-wash blossom or branch marks near the frame edge |
| `iris-mix-style` | 是 | 否 | disabled: reference evidence is identity/garment/scene-specific or too weak |
| `noir-art-style` | 是 | 否 | no reference image |
| `shiratamaco-style` | 否 | 否 | disabled: reference evidence is identity/garment/scene-specific or too weak |
| `shiratamaco-v2-style` | 是 | 否 | disabled: reference evidence is identity/garment/scene-specific or too weak |
| `sheya-style` | 是 | 是 | a few translucent dice or cube glyphs；small snowflake-like geometric particles |
| `say-hana-style` | 否 | 是 | a few painterly orange blossom flecks；restrained circular arch brush marks |
| `say-hana-v2` | 否 | 是 | a few painterly orange blossom flecks；restrained circular arch brush marks |
| `say-hana-v3` | 否 | 是 | a few painterly orange blossom flecks；restrained circular arch brush marks |
| `say-hana-v4` | 是 | 是 | a few painterly orange blossom flecks；restrained circular arch brush marks |
| `say-hana-v5` | 否 | 是 | a few painterly orange blossom flecks；restrained circular arch brush marks |
| `tid` | 是 | 是 | a few small butterflies；sparse drifting leaves or grass blades；a few clear water droplets |
| `ajicoma` | 是 | 是 | small postage-stamp or perforated frame accents；tiny flowers and frill-like corner ornaments |
| `inf-nikki-v1` | 是 | 是 | a few crystalline shards；small water-sparkle particles |
| `noir-aiart` | 是 | 是 | a few dark or luminous butterflies；small stained-glass-like fragments |
| `goto-p` | 是 | 是 | a few soft heart-shaped bokeh accents；small ribbon-like flourishes |
| `komori-hikki-style` | 是 | 是 | a few tiny flower petals；sparse luminous dust motes |
| `tinkle` | 否 | 是 | a few blue rose petals；small crystalline shard accents |

## 判定原则

- 可启用：蝴蝶、花瓣、邮票边框、几何符号、星屑等小型装饰，且能与主体身份分开。
- 不启用：兽耳、发饰、固定服装、人物物种、椅子、伞、整片水下/宫殿背景等内容元素。
- 单张参考图只能说明“候选母题”，不能证明作者所有作品都使用它；因此运行时允许完全省略，不能为容纳母题改写主体或构图。
- 同一参考图对应的隐藏旧版本沿用同一审计结果，但后续批量测试仍只处理 `enabled=true` 的画风。

## 代表性 A/B 实测

使用相同棕发紫瞳阅读者、花园窗边场景和 2:3 比例，对 `tid`、`ajicoma` 分别跑了 Gemini 与 GPT-image-2 的母题开/关组。完整图见 `docs/gpt-image-tid-style/style-motif-ab-20260926.html`。

- `tid`：开启后 Gemini 和 GPT 都出现少量蝴蝶/叶片/水滴，主体身份、阅读动作和窗边场景仍保持，风格辨识度有小幅提升。
- `ajicoma`：开启后邮票齿孔框、角落花饰和小型纸片装饰更明确，提升最明显；人物仍是目标角色，没有复制参考图狐娘。
- 模型不提供固定随机种子，服装、构图和曝光差异不能完全归因于母题。因此实现仍把母题限制为“可省略、少量、从属、只在兼容场景使用”，并且不传给 Gemini 重绘。
