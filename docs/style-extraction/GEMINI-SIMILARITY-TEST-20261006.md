# Gemini 带画风参考图实测（2026-10-06）

本次只生成两张 Gemini 图，没有发出 GPT 生图请求。没有重启或操作现有 app 进程。
测试在独立 offscreen Qt 进程导入 `app.py` 注册的 `SingleGenDebugWidget`，调用实际 `generate_image()` → `ImageGenWorkerThread` → `generate_image_aigc2d`。
仅关闭测试实例的界面状态保存、把输出改到隔离目录；提示词组装、参考图顺序和生成后端沿用真实 GUI 路径。

当前配置：aigc2d / gemini-3-pro-image-preview / 2K / 2:3 / 参考优先，各画风一张；未追加 GPT、重绘或后处理。
两张产物实际均为 1696×2528。画风为已启用的 `puracotte-style-v2` 与 `sakurapion-style`，各使用配置中的参考图。
相同主体是25岁成年女性、栗色长发、绿眼、象牙白衬衫与藏蓝马甲、书店背景；正文位于 `prompts/style-extraction/similarity-validation-20261006.txt`。

完整请求参数、图片、区域提案/叠图、共享计算 JSON 和 HTML 在 `data/test-result/20261006/style-similarity-gemini/`。
HTML 为 `style-similarity-v2.html`，结果为 `comparison.json`。共享引擎 CPU FP32，四个配对的八组指标均成功，同图自检通过。

## 四个配对结果

Gram、AdaIN、LPIPS 越小越近；CSD 和四组本地统计贴近度越大越近。没有综合百分比分数。

| 产物 → 参考 | Gram ↓ | AdaIN ↓ | LPIPS ↓ | CSD ↑ | 明度 ↑ | 边缘 ↑ | 线条 ↑ | 空间 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Puracotte → Puracotte | 2.537698 | 72.405946 | 0.607280 | 0.646576 | 0.821661 | 0.912352 | 0.922658 | 0.796088 |
| Puracotte → Sakurapion | 13.329509 | 140.337624 | 0.667909 | 0.557993 | 0.832966 | 0.946231 | 0.760889 | 0.894943 |
| Sakurapion → Puracotte | 2.901014 | 76.440482 | 0.669887 | 0.547214 | 0.767890 | 0.937625 | 0.954743 | 0.828121 |
| Sakurapion → Sakurapion | 12.913424 | 138.621893 | 0.680712 | 0.538373 | 0.760497 | 0.908226 | 0.743041 | 0.826370 |

Puracotte 产物四个深度指标均更接近自己的参考。Sakurapion 产物的四个深度指标却都更接近 Puracotte 参考；两张产物的脸型、眼睛与发束画法目视也较接近。Puracotte 带出了粉紫光泽、闪点和泡泡，Sakurapion 主要保持较深阴影与较简化的虹膜/发束，参考中的丰富透明层次与细节没有充分迁移。
这两张图只能证明此次样本的结果，不能概括整个画风或模型。内容、明度、配色、姿态与背景不同会影响指标，尤其不能把整图线条贴近度当作发丝连贯性。

## 面部与发丝在线定位核查

当前看图文本模型 gpt-5.6-luna 对四张真实图均返回了候选 JSON，但叠图核查没有通过：

- Puracotte 参考的眼部曲线偏到眼下，发丝路径与头发区域包含饰物、皮肤及衣服。
- 两张产物的面部轮廓延伸到颈部，部分头发多边形夹入脸、领口或衣服；路径不都沿真实细发丝。
- Sakurapion 参考的大致眼位较接近，但轮廓、发丝区域仍混入其他部位。

因此未标记人工确认；暂定数值不能视为有效面部相似度。发丝两项均 unavailable；Puracotte 参考为 three_quarter、产物 frontal，眼部细项因视角不同 not_comparable。Sakurapion 同画风配对有10项 provisional，眼线粗细 unavailable，均不纳入正式均值。详细状态见 JSON/HTML；对应 `*-overlay.png` 保留定位证据。
需要在定位编辑器逐项校正并由用户确认后，才能使用面部细项作正式比较。本次没有用未经校正的自动候选证明画风成功，也没有改动旧确认区域。

## 实测发现的界面修复

单图调试页成功回调使用 `QPixmap`，却未导入；已补充导入，避免成功生成后预览报错。生成文件和完成信号仍已保存，不重复收费生图。`py_compile` 通过，现有单图调试页4项 GUI 回归通过。现有运行进程继续保留，以后正常重启加载修复。
