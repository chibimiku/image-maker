# 图片引用交互、预览布局与重绘比例（2026-10-07）

## 图片引用

按官方 [GPT Image 提示词指南](https://developers.openai.com/cookbook/examples/multimodal/image-gen-models-prompting-guide) 的编号引用方式，正文插入普通文本 `Image N`，没有额外私有 API 标签。

- 用户素材缩略图显示完整 `Image N`，不再用被省略的文件名作为标题；整张卡片、图片和编号均可悬浮查看完整文件名与路径。
- 系统画风图以额外卡片显示在最后，编号 N+1，注明“画风”，不允许拖动或删除；实际用户素材列表不包含这张自动图，后端仍只追加一次。
- 正文有焦点时单击卡片，在当前光标/选区插入 `Image N`。卡片与滚动区域不抢正文焦点。正文无焦点时只预览；拖动不触发插入；双击仍可打开所在目录。
- `utils/image_references.py` 按图片身份映射旧编号→新编号，一次读取旧正文、倒序执行替换，支持 `Image N`（大小写与无空格形式兼容）、`图N`、`圖N`。保持普通数字不动，不会交换1↔2时串号；Qt UTF-16位置转换处理emoji。
- 自动画风图使用独立身份。增添素材时其正文引用顺延；重排用户图时人设/姿势引用随各自图片迁移；更换画风时引用继续指向系统画风角色。重绘模式中保存的生图正文也同步重编号。
- 删除已被引用的图片时将对应引用改为 `[已移除图片引用]`，保留其他有效引用并阻止提交，避免旧编号悄悄指向另一张图。

## 预览布局

旧预览是固定最小220px的 QLabel，并把按上一次控件大小缩好的pixmap作为新的布局尺寸提示；窗口变小或控件增多时没有重新适配，预览最小尺寸与左侧/结果列表挤占空间。

新 `ResizableImagePreview` 保留原始pixmap，在每次resize按当前内容矩形重新缩放；尺寸提示固定，图片不再参与父布局的最小尺寸推导。预览和日志区域与下方结果列表保持各自边界。离线中文UI截图检查通过：`data/test-result/20261007/gpt-reference-ui/numbered-click-preview.png`。

## 本次方图故障证据

原文件全部保留：

| 阶段 | 文件 | 实际尺寸 |
| --- | --- | --- |
| GPT首图 | `gpt-image-2/gptimage2_output_125044_0_41a980.png` | 1536×1024 |
| 第一次Gemini重绘 | `gpt_image_repaint/repaint_gptimage2_output_125044_0_41a980_125117-ef4e9b.jpg` | 2528×1686 |
| 衣装重绘 | `pipeline-steps/125117-af7dfc/wardrobe_125156-2dba87.jpg` | 2048×2048 |

`log/2026-10-07.log` 12:51:18 衣装重绘请求有源图+画风图两张图片，`imageConfig` 只有 `imageSize:2K`，没有 `aspectRatio`；12:51:56返回方图。第一次重绘比例正确，衣装重绘才改变比例。

直接根因是 `utils/post_process.assemble_repaint_request` 计算了 `snapped_aspect_ratio(current)`，但调用 `resolve_repaint_call` 时漏传 `aspect_ratio`，导致计算结果被丢弃、退回磁盘 `auto`。已补传，预检/实际调用都能看到 `3:2`。

同时补齐通用重绘入口：`resolve_repaint_call` 的auto模式按源图实际比例（含EXIF方向）选最接近的模型支持比例，backend逐张计算，混合横竖批次不会共用第一张的比例。显式用户比例仍保留。输出比例偏差超过3%时报错、保留服务器产物、停止下游，不通过拉伸或裁切伪装修复，不新增自动收费重试。UI的auto解释已同步。

## 验证

回归包含真实Qt点击、光标插入、中文/英文/emoji引用交换、系统图顺延、删除引用门禁、窗口缩小不覆盖、混合比例逐张发送、衣装有效请求与实际dispatch一致、错误比例产物保留。

11组相关回归 **340 passed**，py_compile及diff空白检查通过。

尚未重新发起收费生成；下一次生图需正常重启app后观察。没有改写本次原图、方图、日志或发布记录。
