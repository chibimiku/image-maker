# 筛图清单 v1

必须输出一个 UTF-8 JSON 对象（无代码围栏），固定 `schema_version: image-maker.style-dataset.v1`。[manifest.schema.json](manifest.schema.json) 是结构模板；完整校验还检查实际目录覆盖、文件哈希、尺寸、读取状态、入选集合与数量。附带 CLI 与 Image Maker 使用相同校验实现。

| 字段 | 内容 |
|---|---|
| style_name | 1–80 个英文、数字、`-` 或 `_`，以英文或数字开头 |
| source_root | 原始图片目录的绝对路径，Windows JSON 反斜杠须转义 |
| recursive | 布尔值；默认 false；决定目录覆盖范围 |
| requested_count | 2–100 的目标张数；默认 12 |
| agent | name（执行 agent）、model（实际识图模型，未知则据实写 unknown）、review_method（如何粗筛和检查原图） |
| selection_summary | 内容多样性、画法一致性和选择取舍，不能只写“选了最高分” |
| shortfall_reason | 入选不足 requested_count 时必填；否则空字符串 |
| selected_order | 全部入选图片的相对路径，顺序固定，不得重复 |
| reference_image | selected_order 中的一张图，用作测试画风参考图 |
| reference_reason | 代表性、细节可读性及内容泄露风险 |
| images | 范围内所有 `.jpg/.jpeg/.png/.webp/.bmp` 图片，含被排除、不可读和未完成审查项 |

每个 images 对象：

- path：相对于 source_root，以 `/` 分隔，不得绝对路径、包含 `..`、反斜杠或冒号；子目录名保留。
- sha256：实际文件字节的 64 位小写 SHA-256；不是缩略图哈希。
- width、height：实际原文件像素尺寸；不能缩略图尺寸；不可读图均为 0。
- readable：脚本确认能完整解码的布尔值。
- inspected：agent 实际看过图为 true。可读图不允许 false 后声称已完成；不可读图可 false。
- decision：selected / rejected / needs_review。needs_review 清单为未完成草稿，不可训练。
- reason：具体可见证据及选入/排除原因，不把角色的固定发色服装当成画法证据。
- scores：可读图片必须填写 completeness、style_consistency、detail_readability、artifact_cleanliness 四项，各 0–10；不可读图 null。不输出 selection_score，程序计算。
- tags：描述性字符串数组，如 `palette:cool`、`framing:full-body`、`background:dark`、`medium:finished-digital`。不要由文件名猜标签。
- duplicate_of：视觉重复被排除项的代表图片相对路径；不是重复时 null。不可指向自身、未知文件，selected 项不得声明 duplicate_of。

selected_order 必须恰好等于所有 decision=selected 项，至少两张，不超过 requested_count；内容完全重复的文件不能同时入选。若筛选不足目标须说明理由。元数据与实际图片不匹配、漏记图片、未知格式、未看完或缺失理由，App 均拒绝训练导入。

使用模板时替换示例内容，真实目录有多少图就记录多少条。模板不是可以直接导入的结果。用户修改目录后需重新枚举或补充审查，而非偷偷删除清单记录以通过检查。
