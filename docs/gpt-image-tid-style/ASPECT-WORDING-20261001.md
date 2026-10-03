# 画幅措辞：横向输入图为什么出成竖图（2026-10-01）

排查对象：`data/20261001/analysis-gpt-image/634dfc91-waterink-style-100536-f70f54`

## 1. 现象

| 环节 | 实际值 | 对错 |
|---|---|---|
| 源图 | `2000x1024` 横向 16:9（水墨横幅，人物偏右、山水在左） | — |
| 实测字段 `aspect_ratio` | `16:9`（`single_analyzer.calculate_closest_aspect_ratio(源图)`） | ✅ |
| Step 1 Vision 描述 | `The composition is arranged as a balanced vertical tableau` | ❌ 幻觉 |
| Step 2 描述精修 | `This vertical illustration … balanced vertical tableau in a 2:3 frame` | ❌ 又加了 2:3 |
| 落盘 `gpt_image_prompt` | `Vertical 2:3 illustration, one woman centered-right, …` | ❌ 继承 |
| 下发 size（request.json / replay body） | `1536x1024` 横向 | ✅ |
| 首图产物 | `1024x1536` 纵向 | ❌ 被文本带跑 |

后续重绘 / 色调校准全部跟随首图（`snapped_aspect_ratio(首图)` → `2:3`，`-style-adjusted.jpg` 1696x2528），
所以整条链「从一开始」就是纵向。

## 2. 文本是哪里写错的

1. `prompts/single-analyzer-system.md` 对画幅没有任何约束 → vision 把横图写成 vertical。
2. `prompts/refine-desc.md` 第 8 条要求「推断最合适的画幅长宽比（竖图推荐 9:16 或 2:3，横图 16:9 或 3:2）」，
   模型据此把 `2:3` 写进正文；而它输出的 `aspect_ratio` 字段随后被代码实测值覆盖 ——
   **字段纠对了，文本没人管**。
3. `prompts/analysis-gpt-prompt-system.md` 第 3 条硬规则是「保持构图、framing、**aspect ratio** 与描述一致」，
   于是假画幅被原样抄进 `gpt_image_prompt`。
4. `single_analyzer._start_gpt_image_thread` 取 `gpt_image_prompt` 当内容锚直接下发；size 走
   `resolve_gpt_first_pass_size`（按**源图**比例，画风图不参与）→ 1536x1024 与文本打架。

## 3. 为什么「文本压过了 size」

全项目 37 条 gpt 首图 request.json 里，请求横向（`1536x1024`）的只有 5 条：

| 任务 | 请求 | 内容锚里的画幅措辞 | 产物 |
|---|---|---|---|
| `5fda1b1d-noir-aiart` | 1536x1024 | `16:9 landscape illustration scene` | 1536x1024 ✅ |
| `4616a226-puracotte` | 1536x1024 | 无画幅词 | 1536x1024 ✅ |
| `56a2eb48-renian` | 1536x1024 | 无画幅词 | 1536x1024 ✅ |
| **`634dfc91-waterink`** | 1536x1024 | `Vertical 2:3 illustration` | **1024x1536** ❌ |
| **`1805bc93-tinkle`** | 1536x1024 | `in a portrait 2:3 composition` | **1024x1536** ❌ |

写着竖幅措辞的两个，正是唯二出成竖图的两次。样本只有 5 条，不足以宣称因果，但方向一致：
**正文里的画幅声明必须和 size 同向，否则会被模型采纳。**

## 4. 改动

- `utils/aspect_wording.py`（新增）：`align_aspect_wording(text, size=/ratio=)` / `aspect_label`。
  只在前 400 字符内替换，且要求「朝向词/比例词」与画幅名词（illustration / scene / frame / composition /
  crop / tableau / portrait …）同现，所以 `large blue waist bow with long vertical ribbons` 这类普通描述不会被误伤。
- `utils/analysis_gen.build_first_pass_request(..., size=…)`：组装前按**本次生效尺寸**改写内容锚；
  payload 记录 `requested_size` / `aspect_alignment`（GUI 与 CLI 都会打印这行日志）。
- `utils/analysis_gpt_prompt.build_gpt_image_prompt(..., aspect_ratio=…)`：落盘时按**源图实测比例**改写，
  别让假画幅留在产物里继续误导下游（GUI 与 `analysis_pipeline` 都传了）。
- 分析 Tab 的 gpt 参数行新增尺寸下拉 `gpt_size_combo`（1024x1536 竖 / 1536x1024 横 / 1024x1024 方）：
  勾着「跟随输入图」时置灰；取消后**以手动档为准**（`resolve_first_pass_size_for_task` —— 手动档既不读源图，
  也不读画风图）。旧的「取消勾选 = 固定 1024x1536」行为保留为下拉默认值。
- `tools/analysis_gpt_run.py`：size 只算一次（`--size auto` → 源图，拿不到才看内容图/分析比例），
  并传进组装点；UI 选项映射跟着换成下拉。
- 源头三处 prompt：`single-analyzer-system.md`（按图片实际宽高描述画幅）、`refine-desc.md` 第 8 条
  （比例只进 `aspect_ratio` 字段，正文禁画幅词）、`analysis-gpt-prompt-system.md` 新增第 8 条（禁画幅词）。

**画风参考图的朝向在任何一支里都不参与**：`data/style-ref/tid.png` 是 1000x1442 的竖图，
回归用例用它验证「竖版画风图 + 横版源图 → 仍然 1536x1024」。

## 5. 验证

```
python tools/analysis_gpt_run.py --json data/20261001/20261001-100536-634dfc91-花影流水の少女.json \
  --style waterink-style --size 1536x1024 --dry-run --output-dir data/test-result/20261001-aspect-check
→ 内容锚画幅按本次尺寸改写: Vertical 2:3 → Horizontal 3:2
```

`--size auto` 走源图（2000x1024 → 1536x1024）时同样改写。回归用例：
`tests/test_analysis_aspect_wording.py`（16 条）+ `tests/test_analysis_channel.py` 的 5 条尺寸/画幅用例；
相关批次 413 条全绿（`test_pyqt6_smoke` / `test_gpt_image2_api` / `test_analysis_*` / `test_post_process` …）。

## 6. 已知未覆盖

- 画风说明（`prompt_gpt` 的 `Composition density` 字段）里可能有 `Vertically packed frame` 这类措辞
  （如 `noir-aiart`）。它属于画风技法描述、位处提示词最前面，本次**没有**对齐；若后续观察到它也能带偏画幅，
  再考虑按同一规则处理画风段。
- `--size 1024x1024` 的方图档没有实拍证据，只有单元测试覆盖。
