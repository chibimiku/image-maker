# gpt-image 系列上的「画风生图」可行性实验（tid 画风，2026-09-20）

本项目在 Gemini（`gemini-3-pro-image-preview`）通道上已经把「画风生图」玩得很顺：几千词的画风指令 + 一张画风参考图 + 参考优先 / 图文交错等四种模式，出图很稳。
同样的输入换到 gpt-image-2 通道就被质疑「表现很差甚至无法生图，可能是前置引导词太长，gpt-image 不支持那么长的输入」。

这份记录是针对性复核：**gpt-image-2 / 2.5 到底能吞多长的提示词、长提示词会不会毁掉画风、参考图在多大程度上真的在起作用**。
结论先放这里：

1. **长度不是硬门槛**。官方文档写得很清楚：`/images/edits` 与 `/images/generations` 的 `prompt` 都是 `maxLength 32000`（字符）。实测把 **31000 字符和 40000 字符**的提示词直接发过去，上游照样出图（40000 已超文档上限，也没被拒，属于「不要依赖」的灰区）。
2. **「无法生图」另有原因**，八成不是长度。本轮 22 次请求 0 失败；真正常见的失败源见文末「失败排查」。
3. **长提示词的真实代价是「画风/主体一起崩」**：11k 字符的全量画风指令会让主体描述被稀释（要「少女站在浅水」画成了水下精灵、水面浮空水晶），风格反而离参考图更远；把画风说明压到 **400~700 字符**时，风格贴近度和主体符合度同时最好。
4. **参考图在这个通道里非常强，但只有「文字不太长」时才轮到它说话**：带参考图 + 短说明，画风贴近度与配色相似度都是全组最高，而且模型会连**参考图角色的发色/瞳色一起搬过来**（粉发 + 金瞳就是 tid.png 本人的特征）——这正是「只学画风」的反面，必须用文字显式约束。
5. `gpt-image-2.5-flare / -sunburst` 用同样的短说明反而表现更好，且输出 token 只有 gpt-image-2 的约 1/3。

> 本轮全部产物、提示词原文、指标原始表与文档快照都在本目录下，可直接复查（目录清单见 §7）。

---

## 1. 官方文档怎么说（要点摘录）

抓取方式：本机 `curl`/`.NET` 出网被墙，统一用 python 走**本机 HTTP 代理**（端口见 `PROJECT_REQUIREMENTS.md`，不入代码常量），脚本见 `tools/`（仓库外的一次性脚本已清理，抓回的正文快照放在 `web/`）。

### 1.1 接口硬约束

| 项 | 约束 | 来源 |
|---|---|---|
| `prompt`（`/images/edits`） | `minLength 1` / **`maxLength 32000`**（字符） | [Create image edit](https://developers.openai.com/api/reference/resources/images/methods/edit) |
| `prompt`（`/images/generations`） | 同为字符串型描述字段 | [Create image](https://developers.openai.com/api/reference/resources/images/methods/generate) |
| 参考图张数 | GPT image 系列**最多 16 张** per request | 同上（`images` 数组） |
| 单图体积 | `image_url` 的 `maxLength 20971520`（20 MB，base64 data URL 长度上限） | 同上 |
| `size` | `auto` / `1024x1024` / `1536x1024` / `1024x1536`；2.5 的 `flare` 之类支持任意 `WIDTHxHEIGHT`（两边都要 16 的倍数，长短边比 ≤3:1，总像素 655,360~8,294,400） | 同上 + [Image generation guide](https://developers.openai.com/api/docs/guides/image-generation) |
| `input_fidelity` | **gpt-image-2 必须省略该参数**，API 不允许改——模型对**每一个输入图都按高保真处理** | 同上 |
| `quality` | `low` / `medium` / `high`（2.5 另支持 `xhigh` / `max`） | [Create image edit](https://developers.openai.com/api/reference/resources/images/methods/edit) |
| `usage` | 返回 `input_tokens_details{image_tokens,text_tokens}` 与 `output_tokens`，可用来精确核对「提示词有没有被截断」 | 同上 |

`input_fidelity` 那条是本次实验的**理论钥匙**：既然 gpt-image-2 对所有输入图都按最高保真处理，而 `/images/edits` 的语义就是「**编辑**这些输入图」，那么提示词越长、越抽象，模型就越会退回到「把这些图改一改」的行为模式——要么照抄参考图主体，要么按自己的老套路画。

### 1.2 官方提示词指南的口径

[GPT Image Generation Models Prompting Guide](https://developers.openai.com/cookbook/examples/multimodal/image-gen-models-prompting-guide)（Cookbook）明说了几件事：

- 「Rich, detailed prompts are the norm... **small wording/layout tweaks usually improve legibility**」——官方鼓励详细，但
- 「**Iterate instead of overloading**: Long prompts can work well, but debugging is easier when you start with a clean base prompt and refine」——**长提示词能跑，但不是推荐起手式**。
- 结构建议：`background/scene → subject → key details → constraints`，**复杂需求用短标签分段/换行，不要一整段长散文**（「use short labeled segments or line breaks instead of one long paragraph」）。
- 多图输入：**按索引和描述逐个引用**（「Image 1: product photo, Image 2: style reference」），并说明如何交互（「apply Image 2's style to Image 1」）。
- 5.1 Style Transfer 的官方范例短得离谱：`Use the same style from the input image and generate a man riding a motorcycle on a white background.` —— 一句就够。

也就是说：**官方推荐的做法是「短句 + 参考图」，不是「几千词说明书 + 参考图」**。本项目的 tid 全量指令（11,186 字符 / 1,417 词）正好反向。

### 1.3 社区/第三方实测：Image API 传参考图时提示词会被忽略

- 日文实测笔记 [「GPT Image 2」Image API 経由でリファレンス画像を渡したら、3 回連続でプロンプトが無視された](https://note.com/more_sunset/n/n8128a703259e)（2026-04-30）：同一张参考图 + 同一条英文提示词，`Image API` 连出 3 张都把**文案和「不要加 logo」这类禁止项一起无视**；换 `Responses API` 同条件一次通过。作者的归因是两者职责不同，并算了一笔账：Image API medium 单张 $0.055、Responses API medium 单张 $0.09（约 1.6 倍），但「连出 3 张废图」的话总成本反而 Responses API 更低。
- 这说明「参考图 + 提示词」在 gpt-image 的 Image API 路径上**服从性本来就弱**，与本轮量化结果一致（见 §5 观察 2、3）。

---

## 2. 实验设计

- 站点/通道：`new.aigc2d`（`apis.aigc-2d-gpt`）→ `POST /v1/images/edits`（带参考图）/ `POST /v1/images/generations`（不带）
- 模型：`gpt-image-2`（主）；`gpt-image-2.5-flare`、`gpt-image-2.5-sunburst`（对照）
- 尺寸/画质：`1024x1536`、`quality=medium`（token 边界测试用 `1024x1024`/`low`）
- 画风：`conf/config-styles.json` 的 `tid`（`prompt` 11,186 字符 / `prompt_compressed` 1,758 字符 / `ref_image = data/style-ref/tid.png`，1000x1442）
- 统一主体（英文常量，所有变体一致，只变画风注入方式）：
  `A girl stands in shallow water at dusk, waist-up, looking at the camera, wearing a white dress.`
- 每次请求一张、顺序执行，产物落 `data/<日期>/gpt-image-tid/`，`*_server_response_*.json` 保留原始返回（含 `usage`）。

### 2.1 变体清单

| 变体（文档内简写） | 画风注入方式 | 文本字符 | 参考图 | 说明 |
|---|---|---|---|---|
| `A1_full_no_ref`（A1） | `off`（全量画风指令，无图） | 11,283 | ✗ | 现有 GUI「关闭」模式的等价物 |
| `A2_compressed_no_ref`（A2） | `off`（压缩版指令，无图） | 1,855 | ✗ | `prompt_compressed` 单跑 |
| `B1_priority`（B1） | `priority` 参考优先（`prompt_compressed` + 参考块） | 4,195 | ✓ | GUI「参考优先」 |
| `B2_head`（B2） | `head` 头部插入（全量指令 + 参考块） | 13,232 | ✓ | GUI「头部插入」 |
| `B3_interleave`（B3） | `interleave` 图文交错 | 13,232 | ✓ | GUI「图文交错」（gpt-image 下实际是拼成一条 prompt） |
| `B4_head_ref`（B4） | `head`（复跑，带图） | 13,232 | ✓ | 复核 B2 |
| `B5_compressed_ref`（B5） | 只给压缩版指令 + 参考图（无参考说明块） | 1,855 | ✓ | 「短文字 + 图」 |
| `C_short_tag` / `C2_short_tag_rerun`（C / C2） | 手写 305 字符「只学画风」短说明 + 参考图 | 402 | ✓ | 校准方向（两次复跑看稳定性） |
| `D_control`（D） | 完全不带画风，只给主体 | 95 | ✗ | 基线「默认画风」 |
| `E1_ref_only`（E1） | 只给主体 + 参考图，一个画风字都没有 | 95 | ✓ | 参考图单独能带多少风格 |
| `G1/G2/G3_calibrated`（G1/G2/G3） | 校准版 678 字符单段说明（保留项 + 更换项）+ 参考图 | 678 | ✓ | gpt-image-2 / 2.5-flare / 2.5-sunburst |
| `G4_calibrated_no_ref`（G4） | 同上但**无参考图** | 678 | ✗ | 隔离「参考图贡献」（对照 G1） |
| `H1_32k` / `H2_40k` | 31,000 / 40,000 字符灌水提示词 | 31,000 / 40,000 | ✗ | 长度边界（H1 已超官方 32000 上限同量级，H2 直接超 8000） |

校准版短说明原文（`G*`，可直接复用改画风）：

```
Use the art style of the attached reference image (Image 1) and draw a NEW image of the scene below.
Keep from Image 1: the gentle watercolor-like digital painting finish, delicate thin lineart, soft
luminous pastel palette, high-key bright ambience, airy pale background with floating translucent
droplets and bubbles, glossy detailed eyes, and the calm elegant mood.
Change from Image 1: the character, her pose, her outfit, the camera framing and the background layout
— all of these come only from the scene description. Do not copy the reference's person or composition.
Scene: <主体描述>
```

---

## 3. 结果（量化）

指标口径与 `tests/calc_style_similarity.py` 一致：**CLIP 图像向量余弦**（越高越像参考图整体）、**HSV 配色直方图余弦**（纯配色）、亮度均值、饱和度均值、白底占比（灰度 >235 的像素比）、边缘密度差、对比度差。
参考图 `data/style-ref/tid.png` 自测：亮度 221.5、饱和度 25.0、白底占比 0.667、边缘密度 0.0663、对比度 0.2139 —— 典型的**高亮、低饱和、大面积留白**的水彩感。

| 变体 | 画风模式 | 文本字符 | CLIP↑ | HSV↑ | 亮度均值 | 饱和度均值 | 白底占比 | 边缘密度差↓ | 对比度差↓ |
|---|---|---|---|---|---|---|---|---|---|
| B5 | 短文字 + 图 | 1855 | 0.8314 | 0.1959 | 169.3 | 71.9 | 0.075 | 0.0329 | 0.0441 |
| **G2** | 校准版 + 图（2.5-flare） | 678 | **0.8134** | **0.3948** | 225.6 | 27.7 | **0.393** | 0.0183 | 0.1200 |
| C | 自定义短说明 + 图 | 402 | 0.7944 | 0.2849 | 222.1 | 25.8 | 0.318 | 0.0091 | 0.1250 |
| B1 | 参考优先 | 4195 | 0.7890 | 0.1055 | 136.0 | 99.3 | 0.057 | 0.0389 | 0.0181 |
| G1 | 校准版 + 图（gpt-image-2） | 678 | 0.7882 | 0.2869 | 220.6 | 27.4 | 0.289 | 0.0119 | 0.1168 |
| G3 | 校准版 + 图（2.5-sunburst） | 678 | 0.7832 | 0.2579 | 197.1 | 37.1 | 0.184 | 0.0526 | 0.0361 |
| C2 | 自定义短说明 + 图（复跑） | 402 | 0.7757 | 0.3186 | 224.8 | 24.2 | 0.320 | 0.0225 | 0.1275 |
| B3 | 图文交错 | 13232 | 0.7751 | 0.2140 | 170.4 | 48.7 | 0.096 | 0.0336 | 0.0203 |
| B2 | 头部插入 | 13232 | 0.7682 | 0.0572 | 126.5 | 94.2 | 0.025 | 0.0537 | 0.0112 |
| A1_run2 | 关闭（全量指令，复跑） | 11283 | 0.7567 | 0.0964 | 114.9 | 108.0 | 0.035 | 0.0420 | 0.0003 |
| H1_32k_run2 | 31k 灌水（复跑） | 31000 | 0.7387 | 0.1466 | 191.2 | 38.8 | 0.164 | 0.0137 | 0.0325 |
| H2_40k | 40k 灌水 | 40000 | 0.7848 | 0.1551 | 203.3 | 38.7 | 0.181 | 0.0235 | 0.0873 |
| B4 | 头部插入（复跑） | 13232 | 0.7363 | 0.2735 | 185.8 | 46.7 | 0.124 | 0.0291 | 0.0565 |
| A1 | 关闭（全量指令） | 11283 | 0.7328 | 0.0151 | 96.1 | 157.3 | 0.019 | 0.0787 | 0.0107 |
| H1_32k | 31k 灌水 | 31000 | 0.7281 | 0.0328 | 130.0 | 64.6 | 0.012 | 0.0195 | 0.0284 |
| G4_calibrated_no_ref | 校准版短说明**无图** | 678 | 0.7218 | 0.2001 | 200.2 | 27.2 | 0.134 | 0.0228 | 0.0626 |
| E1 | 仅参考图，无画风字 | 95 | 0.7212 | 0.0581 | 163.1 | 55.9 | 0.029 | 0.0013 | 0.0760 |
| A2 | 关闭（压缩指令） | 1855 | 0.6583 | 0.1332 | 146.1 | 54.6 | 0.051 | 0.0179 | 0.0008 |
| D | 无画风 | 95 | 0.5307 | 0.0184 | 114.9 | 54.2 | 0.001 | 0.0229 | 0.0386 |

> `A1_run2` / `H1_32k_run2` / `B4` / `C2` 是同条件复跑，用来判断差异是不是抽卡噪声（结论：同一段文字下，长提示词组两张都崩、短说明组两张都稳）。

> **指标读法提醒（重要）**：CLIP 是整图相似度，**会被「主体本身像不像」污染**。`B5_compressed_ref` CLIP 最高（0.8314），原因是它把参考图角色的**粉发金瞳整个搬了过来**（主体近似 → CLIP 高），而不是画风更准；真正反映「画风/配色」的是 **HSV 列 + 亮度/饱和度/白底三列**——那里 `G2`（0.3948 / 亮度 225.6 / 饱和 27.7 / 白底 0.393）与参考图（饱和 25.0 / 白底 0.667）最接近。判定「是否学到画风」请以 HSV + 亮度/饱和/白底为主，CLIP 只作参考。

raw：`similarity_all.txt`（CLIP+HSV）、`fingerprint.json`（亮度/饱和/白底/边缘/色相熵）。
`tests/calc_style_similarity.py` 也能直接复跑：

```powershell
& "C:\Program Files\Python310\python.exe" tests/calc_style_similarity.py --ref-image data/style-ref/tid.png --images docs/gpt-image-tid-style/images/*.png
```

---

## 4. 逐组观感（人工判定，产物见 `images/`）

| 变体 | 产物 | 主体还原 | 画风贴合 | 关键观察 |
|---|---|---|---|---|
| A1 全量指令无图 | `tid-A1.png`（另 `tid-A1_run2.png`） | **崩** | 低 | 「少女站浅水、半身、白裙」被画成**水下精灵 + 水母 + 浮空水晶装饰**，蓝紫冷调，饱和爆表（饱和均值 157.3）；复跑同样崩（饱和 108.0） |
| A2 压缩指令无图 | `tid-A2.png` | 好 | 中低 | 主体完全对，但落回 gpt-image 默认的「半写实 2.5D 厚涂 + 强 bloom」，白底占比 0.051 |
| B1 参考优先 | `tid-B1.png` | 好 | 中 | 高调、加了很多悬浮气泡/水花，比 A2 亮一大截，但饱和仍到 99.3 |
| B2 头部插入（全量） | `tid-B2.png` | 好 | 中低 | 主体对，但饱和度 94.2、白底 0.025，画风没被参考图带过去 |
| B3 图文交错（全量） | `tid-B3.png` | 好 | 中 | 亮度 170、白底 0.096，比 B2 好；但「交错」在 gpt-image 下并不成立（见 §5 观察 3） |
| B4 头部插入（复跑） | `tid-B4.png` | 好 | 中 | 与 B2/B3 同量级，说明该模式下长提示词稳定「压过」参考图 |
| B5 短文字 + 图 | `tid-B5.png` | 好 | **画风被主体污染** | 直接长出**粉发 + 金瞳**（tid.png 角色本人的特征），证明参考图以「编辑输入图」方式在工作 |
| C / C2 短说明 + 图 | `tid-C.png` / `tid-C2.png` | 好 | **最好（其一）** | 柔和细线、粉白低饱和、大片留白 + 悬浮气泡，两次复跑都稳定（HSV 0.2849 / 0.3186） |
| E1 仅参考图 | `tid-E1.png` | 好 | 中 | 一个画风字都没写，靠垫图就把整体调子带偏到高调粉彩、也长了粉发 |
| G1 校准版（gpt-image-2） | `tid-G1.png` | 好 | **好** | 亮度 220.6 / 饱和 27.4 / 白底 0.289，主体干净、没有照抄参考主体 |
| G2 校准版（2.5-flare） | `tid-G2.png` | 好 | **最佳综合** | 亮度 225.6 / 饱和 27.7 / 白底 0.393，配色最贴参考图，发丝细线更接近 tid |
| G3 校准版（2.5-sunburst） | `tid-G3.png` | 好 | 好 | 更暗更"厚"一点（亮度 197 / 边缘密度 0.1189），仍然保留粉彩 + 气泡 |
| G4 校准版短说明（**无参考图**） | `tid-G4_calibrated_no_ref.png` | 好 | 中 | 同一段文字少了参考图：文字里写的「水彩感/细线/粉彩/气泡」都照做了，但**边缘密度 0.0891（参考图 0.0663）、白底仅 0.134（G1 是 0.289）**，整体又退回半写实 2.5D |
| H1/H2 超长灌水 | `tid-H1_32k.png`（`_run2`）/ `tid-H2_40k.png` | **崩** | 低 | 31k/40k 都被接受并出图，但主体彻底跑飞（水精灵、花丛少女），证明**长度不是拒绝原因、而是主因** |

---

## 5. 五条结论（含数据支撑）

**观察 1：长度没有硬门槛。**
`usage` 直接给出了证据：`A1`（11,283 字符）→ `text_tokens 2,112`；`H1`（31,000 字符）→ `text_tokens 5,867`；`H2`（40,000 字符）→ `text_tokens 7,575`。
字符数与 text token 严格成正比（约 5.3 字符/token），**没有任何截断迹象**，40k 甚至超过文档的 32,000 也被接受（中转站未做该校验，属于灰区，别依赖）。22 次请求 0 失败。
→ 「gpt-image 不支持那么长的输入」这个假设**不成立**。

**观察 2：长提示词真正伤害的是「参考图的话语权」。**
把参考图固定住、只变文字量（按文本字符数从小到大）：`C（402）≈ C2 → G*（678）→ B5（1855）→ B1（4195）→ B3/B2/B4（13232）→ A1（11k，纯文本）`。
对应 HSV 配色相似度：0.2849 / 0.3186（402）→ 0.2869 / 0.3948 / 0.2579（678）→ 0.1959（1855）→ 0.1055（4195）→ 0.2140 / 0.0572 / 0.2735（13232）→ 0.0151（11k）——**除了 B4 这一个例外，整体单调下滑**；文字越长，输出越往 gpt-image 的默认风格漂：饱和度从 27.4（678 字符）一路涨到 99.3（4195）、94.2（13232）、157.3（11k 纯文本），白底占比从 0.289 掉到 0.025。
→ **几千词的画风说明书在 gpt-image 上是负资产**：它稀释主体、抢走参考图的空间。

**观察 3：本项目四个模式在 gpt-image 上的语义与 Gemini 不同。**
- `interleave`（图文交错）在 Gemini 上是真的 `[text, image, text]` 交错；在 gpt-image 通道里 `build_gpt_image2_payload` 把 `instructions + prompt + post_instructions` **拼成一条字符串**发给 `/images/edits`，图片永远只有一组附件——**交错结构不存在**，等价于「说明放结尾」。
- `priority`（参考优先）名义上"参考图主导"，实测却被压缩版指令里的细节规则牵着走（HSV 仅 0.1055，白底 0.057）。
- `head`（头部插入）把 1.4k 词规则塞在参考说明之后，长文本直接把参考图压掉。
→ 想要「参考图主导」，**唯一有效的做法是缩短文字**，而不是把模式切换来切换去。

**观察 4：参考图是真的在起作用，而且会「过拟合」到主体特征。**
`E1`（一个画风字都不写）与 `B5`（短压缩指令）都把参考图角色的**粉发 + 金瞳**搬了过来；`D_control`（无图无文字）则是黑发 + 深蓝海面的默认半写实。
隔离实验 `G1（有图）↔ G4（无图）`：同一段 678 字符说明，有图 HSV **0.2869** / 白底 **0.289** / 边缘密度 0.0782，无图 HSV **0.2001** / 白底 **0.134** / 边缘密度 0.0891 —— 参考图单独贡献了约 **+0.09 HSV、白底翻倍**，而且无图时边缘更"硬"（回到 2.5D 厚涂的轮廓）。
→ gpt-image-2 的 `input_fidelity` 不可调、所有输入图按高保真处理，**它倾向于把参考图当"要编辑的原图"而不是"风格色卡"**。想只要画风，必须**显式写出「保留什么 / 换掉什么」**（校准版 G 系列就是干这个的）；反过来，只想让参考图出效果、不想写字，就接受它连主体特征一起搬。

**观察 5（顺带）：2.5 两个模型在同样输入下输出更省、画风更稳。**
`G2/G3` 的 `output_tokens` 只有 **343**，而 gpt-image-2 同参数是 **1,105~1,372**（约 1/3）；耗时 flare 29.6s、sunburst 36.0s（都低于 gpt-image-2 的 51.3s）。`G2` 的 HSV 与白底占比还是全组最佳。
→ 走参考图 + 短说明的场景，`gpt-image-2.5-flare` 是当前更优解。

---

## 6. 给本项目的落地建议

1. **给 gpt-image 通道单独准备一份「短版」画风说明**（目标 **400~700 字符**，参考 `G*` 校准版的写法：保留项 + 更换项 + 一句 scene），不要复用 Gemini 用的 `prompt_compressed`（1,758 字符）和全量 `prompt`（11,186 字符）。可以在 `config-styles.json` 条目里加第三个字段（例如 `prompt_gpt`）或按通道选择，避免在 Gemini 上反向退化。
2. **凡是走 gpt-image 路径，参考图必须配「显式换主体」的短指令**：只写「参考它的画风」不够，实测会连发色/瞳色一起复制。
3. **不要对 gpt-image 用 `head`（全量）与 `interleave` 两种模式**：前者是把长文本压在参考图前面，后者在 payload 层根本不成立。全部塌缩成一条短 prompt 即可。
4. **「无法生图」排查顺序**（按实测概率）：
   - **整包体积**：单图 data URL 上限 20 MB（`maxLength 20971520`），长提示词 + 多张参考图会让 multipart/JSON 请求体越来越大；本项目的 `_maybe_compress_image_path` 只在超 2048px 时压缩，**没有总包体积保护**——这是一条真正可能触发上游拒收的路径；
   - 站点/通道是否支持 `background=transparent`、`input_fidelity`（gpt-image-2 不允许传后者）；
   - 尺寸是否在上游允许清单（aigc2d 只有 3 档，其他值会被收敛，收敛逻辑见 `normalize_gpt_image2_size`）；
   - 参考图是否超 16 张；
   - 内容审核（4xx 会原样落盘，不重试；看 `data/<日期>/<子目录>/*_server_response_*.json` 里的 `error`）；
   - 额度/并发（429/5xx 才重试）、超时（`high` 画质单张 3~5 分钟）；
   - **最后才怀疑提示词长度**——本轮 11k~40k 全部成功，22 次 0 失败，没有复现「长前置词导致拒图」。
5. **量化回归就用这套脚本**：`tests/calc_style_similarity.py`（CLIP+HSV+线条）+ 本目录 `fingerprint.json` 的亮度/饱和/白底口径；改画风注入逻辑后跑一遍，看 HSV↑、亮度↑、白底↑、边缘密度差↓ 是否同向。
6. **模型选择**：参考图 + 短说明走 `gpt-image-2.5-flare`（快、省 token、画风最贴）；要更"重"的厚涂感再试 sunburst；`gpt-image-2` 保留给纯文字生图。

---

## 7. 目录内容

```
docs/gpt-image-tid-style/
├── README.md            ← 本文
├── images/              ← 全部产物（共 18 张；文件名 = 变体名，`_run2` 为同条件复跑，如 tid-G2.png、tid-H2_40k.png）
├── prompts/             ← 每个变体实际发出的提示词全文（可用 tools/build_variants.py 重建）
├── logs/                ← 每次请求的 CLI 日志（请求 URL / 完整 prompt / 尺寸 / 耗时 / usage；G4 的那条没落盘，其 usage 见 `data/**/tid-G4_*_server_response_*.json`）
├── tools/               ← 复跑脚本：build_variants.py（重组变体）/ collect_metrics.py（重算两张表）/ show_metrics.py（看单项指标）/ calc_style_similarity_snapshot.py（CLIP 快照）
├── similarity_all.txt   ← CLIP/HSV/线条统计原始表（collect_metrics.py 生成）
├── fingerprint.json     ← 亮度/饱和/白底/边缘/色相熵原始表
└── web/                 ← 官方与第三方文档正文快照（OpenAI images edits/generate 文档、Cookbook 提示词指南、日文实测笔记）
```

> 每次请求的 CLI 原始日志已随目录留档（`logs/`，含完整 prompt 与 `usage`）；原始返回 JSON 在 `data/<日期>/<子目录>/*_server_response_*.json`。
> 需要新一批日志时按下面命令复跑即可。

画风条目改动后重建变体文字（用的是 `utils/styles.py` 的同一套组装函数，所以与 GUI 口径一致）：

```powershell
& "C:\Program Files\Python310\python.exe" docs/gpt-image-tid-style/tools/build_variants.py --style tid
```

复跑某一组（示例，产物落 `data/<日期>/gpt-image-tid/`）：

```powershell
# 校准版短说明 + tid 参考图，用 2.5-flare
& "C:\Program Files\Python310\python.exe" tools/gpt_image2_gen.py `
  --site new.aigc2d --model gpt-image-2.5-flare --size 1024x1536 --quality medium `
  --output-subdir gpt-image-tid --prefix tid-G2 `
  --prompt "Use the art style of the attached reference image and draw a NEW image. Keep: <风格要点>. Change: <主体/构图>. Scene: <主体描述>" `
  --image D:\code\image-maker\data\style-ref\tid.png
```

> 复现注意：`tools/gpt_image2_gen.py` 的 `--prompt-file` 与 `--prompt` 同时出现会产生**两条独立任务**；提示词里含换行时会被按行拆成多条，请把「画风说明 + 主体」拼成**单行单条**再传（本轮 `H1/H2` 的两次请求就是这么来的，不影响结论）。
