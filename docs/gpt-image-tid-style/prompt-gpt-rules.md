# 画风「短版说明」（prompt_gpt）转换规则与维护说明

> 面向 gpt-image 通道（`new.aigc2d` 的 `aigc-2d-gpt` 节点、模型 `gpt-image-2/2.5-*`）。
> 依据与实测数据：`docs/gpt-image-tid-style/`。取用链路：`utils/style_gpt.py` → `utils/styles.py` → 各 Tab。

## 1. 为什么需要第三份说明

`conf/config-styles.json` 里每个画风原本有两份文本：

| 字段 | 用途 | 规模 |
|---|---|---|
| `prompt` | Gemini 通道的完整说明书 | 几千~一万字符 |
| `prompt_compressed` | 参考优先模式的压缩版 | ~1.7k 字符 |
| **`prompt_gpt`** | **gpt-image 通道专用短版（本规则产出）** | **≤680 字符** |

gpt-image 通道把「画风说明 + 主体」拼成**一条** prompt 发给 `/images/edits`。实测（同主体、同参考图，只换文字）：

- 全量说明书（13k）：配色贴合度 **0.12~0.25**，且会把参考图挤掉；
- 压缩版（1.8k）：0.20 上下；
- **结构化字段版（~550 字符）：0.64——全场最贴**，而且不复制参考图角色的发色瞳色。

结论：gpt-image 需要的是**短、结构化、只讲画法**的说明，不是更短的散文。

## 2. 字段契约（唯一的格式规则）

`prompt_gpt` 必须**恰好**是这 8 行，顺序固定，`Field: value` 形式，不允许多余文字：

```
Palette: <主色相、饱和度高/低、明度结构、冷暖倾向>
Lighting: <主光/补光/轮廓光模型、环境光、对比度、是否有 bloom/glow>
Brushwork: <媒介与笔触：水彩晕染、干笔、喷枪、赛璐璐平涂、厚涂>
Edges: <描边粗细与颜色、边缘是锐利/柔和/丢失>
Texture: <表面质感：纸纹、网点、半色调、颗粒、干净平涂>
Composition density: <画面填充度、留白比例、视觉焦点层级>
Detail level: <细节集中在哪、哪里被简化、整体完成度>
Avoid: <渲染层反模式（不该像什么），逗号分隔>
```

约束（`utils/style_gpt.py` 的 `validate_prompt_gpt` 会逐条检查，不合规直接重试）：

1. 8 个字段齐全；除这 8 行外不允许出现别的行；
2. 单字段 ≤160 字符，整份 **150~680 字符**；
3. **不许出现主体/角色词**：`girl/woman/character/hair/eyes/pose/outfit/dress/smile/portrait/少女/发色/瞳色/姿势/服装…`
   （渲染层常见词不误判：`uniform`=均匀光、`skin`=材质、`face`=表面、`lights`=高光 都不拦）；
4. `Avoid` 只写渲染反模式，不写内容禁令；
5. 不许出现 `masterpiece / 8k / best quality / ultra detailed` 这类堆料词。

> 字段值可以带参考图的客观数字（色调占比、留白比例、饱和度水平），但不要抄成像素统计表。

## 3. 转换流程（`tools/convert_styles_gpt.py`）

```
① 取画风条目 → prompt 全文
② 取参考图客观统计（tools/style_ref_stats.py，OpenCV 本地算，秒级）
③ 交给文本模型（conf/config.json 的 base_url/model，key 走 IMAGE_MAKER_TEXT_API_KEY）
   按 prompts/style-gpt-convert-system.md + -user.md 的提示词转成 8 字段
④ 校验 → 不合规重试（最多 3 次，第二次起把报错回灌给模型）
   → 仍超长时用 repair_prompt_gpt_lenient 在字段级收口
⑤ 写回 conf/config-styles.json 的 prompt_gpt 字段（读旧→只加自己的键→写回）
```

常用命令：

```powershell
# 只为还没转过的画风补 prompt_gpt
python tools/convert_styles_gpt.py
# 全部重转（改了规则/换了模型时用）
python tools/convert_styles_gpt.py --force
# 只处理指定画风
python tools/convert_styles_gpt.py --force --only tid,ajicoma
# 只看校验结果
python tools/convert_styles_gpt.py --check
# 只看将要发出的请求
python tools/convert_styles_gpt.py --dry-run --only tid
# 关掉「客观统计」依据（退回纯文字转换，一般不必要）
python tools/convert_styles_gpt.py --force --stats-only=false      # 或 --no-image 走纯文本
# 参考图统计单独跑（也会被 convert_styles_gpt 自动补算）
python tools/style_ref_stats.py --all --json-out cache/temp/style-ref-stats.json
```

## 4. 为什么用「图像统计」而不是把图发给模型

先试过把参考图作为多模态输入（`call_text_model(..., image_path=...)`，模型确实支持 vision），但：

- 单张 1~3MB 的 data URL 会让每次调用 60~300 秒，29 个画风要跑十几分钟甚至超时；
- 而 `tools/style_ref_stats.py` 用 OpenCV 直接在本地算出「亮度/暗部占比/高光占比/留白占比/
  饱和度/色相占比/主色（k-means）/边缘密度/线色亮度」——**比模型目测更准、零延迟**。

所以默认路径是「统计 + 说明书」纯文本转换；`--no-image` 之外的图像输入仅作调试备用。

## 5. 取用优先级（生图时）

`utils/style_gpt.resolve_style_prompt` 与 `utils/styles.build_ref_gen_params(api_type=...)`：

1. api_type 归一化后**含 `gpt`**（`aigc-2d-gpt` / `gpt-image-*`）→ 用 `prompt_gpt`；
2. 缺 `prompt_gpt` → 退回 `prompt_compressed`；再缺 → 退回 `prompt`；
3. 其它通道（`aigc2d` / `autodl` / `openrouter-image` 等）→ 始终用全量 `prompt`。

接入点：`modules/image_generation/prompt_generator.py`（批量提示词生图）、`single_gen_debug_tab.py`（单图调试，
含四种参考图模式）、`image_edit.py`、`char_design.py`，以及 `tools/gpt_image2_gen.py --style <画风>`。

## 6. 维护纪律

- **改规则**就改 `prompts/style-gpt-convert-*.md` 与 `utils/style_gpt.py` 的常量，然后 `--force` 全量重转；
- **改单个画风**：改完它的 `prompt` 后跑 `--force --only <名字>`，别手改 `prompt_gpt`（手改容易漏字段）；
- `conf/config-styles.json` 是 gitignore 的运行时文件，版本化副本在
  `submodules/image-maker-artstyle/config-styles.json`，改完要同步过去再提交；
- 新增画风：先补 `prompt` + `ref_image`，再跑一次转换；没有参考图的画风（如 `noir-art-style`）
  只能纯文字转换，字段里的配色数字会缺失，需要人工补一句。
