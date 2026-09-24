# gpt-image 画风生图：推荐提示词（v5，Sol 修正版）

> 来源：`gpt-5.6-sol` 两轮诊断（原文 `sol-cases/answer_case_gpt-5.6-sol.md` / `answer_verify_gpt-5.6-sol.md`），
> 实测见实验报告 §4.8。适用于 **gpt-image-2 / 2.5**，配合一张**画风参考图**（`/images/edits`）。

## 0. 参数

| 项 | 值 | 依据 |
|---|---|---|
| 站点/模型 | `new.aigc2d` · `gpt-image-2`（或 `gpt-image-2.5-flare`） | flare 更快更省 |
| 尺寸 | `1024x1536` 竖 / `1024x1024` 方 / `1536x1024` 横 | 主体构图裁到半身，别用全身远景 |
| 画质 | **`medium` 起步**；发丝/蕾丝要求高时 `high` | 实测 high 能减少局部粘连（代价：更慢、更贵、细节更多） |
| 格式 | `png`，`background` 保持不透明 | JPEG 会污染边缘与配色 |
| 提示词长度 | **1500~2000 字符**（v5 = 2000） | 超 2000 后参考图话语权开始下降 |
| 参考图 | 画风样图（必须是**含人物/结构**的完整插画） | 纯背景风格板学不到线稿与发丝组织 |

## 1. v5 全文（≤2000 字符，可直接用）

```
1. Reference
Use the reference only for visual language: luminous anime illustration, clean elegant contours, restrained watercolor texture, airy white space, selective color accents. Do not copy its character, pose, clothes, or objects.

2. Scene
<主体与构图一句话>

3. Composition
Portrait orientation, waist-up framing. Make the head and upper body prominent but leave clear breathing room above the hair and beside both shoulders. Keep the pose readable; avoid tight cropping, oversized bust, or hidden arms.

4. Style
Polished Japanese anime illustration with graceful tapered linework and soft watercolor-like fills. Bright, delicate, and graphic rather than photorealistic. Keep the background spacious and simplified.

5. Light & color
High-key dusk light with pale pink, cool blue, and small warm accents. Preserve midtones and a few compact dark anchors around eyes, hair roots, dress folds, and water contact. White dress, skin, sky, and reflections must remain distinct; no blown highlights, gray veil, or uniform pastel wash.

6. Lines
Use continuous, confident contours for face, jaw, eyes, hair silhouette, shoulders, dress, and waterline. Lines may soften in light but must not dissolve, double, fray, or merge into texture. Prioritize silhouette and facial features over internal decoration.

7. Hair
Build hair as a few large flowing masses, then fewer medium locks, with only sparse fine flyaways. Each lock follows one clear root-to-tip flow. Avoid strand-by-strand striping, tangled loops, parallel grooves, and dense crossings over the face.

8. Detail budget
Three scales only: large silhouette and value masses; selective medium folds, locks, and ripples; very sparse microdetail and sparkles. Keep face and eyes crisp, hair moderately detailed, dress controlled, water and sky simple. Preserve clean negative space; no paper grain everywhere, excessive lace, foam, ripples, or edge clutter.
```

换画风时：把 `4. Style` / `5. Light & color` 两句换成该画风的实际特征（色调、光感、笔触、留白水平），其余段（Composition / Lines / Hair / Detail budget）**照抄不动**——这四段是治「糊线 + 过白」的药，跟具体画风无关。

## 2. 四条硬约束（换句话说就是"别写这些"）

这些是 R5 翻车的直接原因：

| ❌ 别写 | ✅ 改写 |
|---|---|
| `lots of clean white` | `white-dominant background` + `controlled value separation` + `protected midtones` |
| `soft even light` | `soft cool shadows at form turns, roots, folds and overlaps` |
| `Avoid: heavy shading` | 禁 `clipped highlights` / `white-on-white loss of silhouette`，**不要禁掉所有阴影** |
| `delicate fine lineart` | `continuous, confident linework` + `a small number of intentional lines` |
| 水彩当全局 Brushwork | 水彩只出现在 `distant clouds / water reflections / decorative accents` |
| `Edges: soft, occasionally lost`（无范围） | 明确列 crisp 部位（脸/眼/发轮廓/主发束/领口/裙缝/水线），lost edges 只给远景 |

**发丝唯一可执行的写法**：大块 → 中束 → 少量游丝；每束从发旋/发根出发、沿头颅曲率**连通到发梢**、有明确尖端。
**细节预算**（最重要、也最容易被忽略）：大轮廓 / 中景（发束·五官·衣褶·水线）/ 稀疏微料三层，微料只给脸与上身。

## 3. 简短版（`prompt_gpt` 字段版，约 500 字符）

工程上用 `config-styles.json` 的 `prompt_gpt`（8 字段短版）时，至少把上面四条并进字段值：

```
Palette: <画风主色> with controlled value separation; white-dominant background, visible midtones.
Lighting: high-key <画风光感> with protected midtones and restrained highlights; soft cool shadows
at form turns, roots, folds and overlaps; no blown highlights, no uniform pastel wash.
Brushwork: clean digital anime drawing with continuous, confident linework built from a small number
of intentional lines; translucent watercolor texture only in distant background and accents.
Edges: crisp and continuous on face, eyes, hair silhouette, major locks, neckline, dress seams and
waterline; lost edges only in distant background; preserve edge hierarchy.
Texture: subtle grain only in quiet background; keep texture off eyes, facial contours and hair boundaries.
Composition density: low; generous light negative space that stays clean, subject keeps a readable silhouette.
Detail level: three scales only — silhouette and value masses; grouped hair locks, facial features,
dress folds, waterline; sparse microdetail on face and upper torso. Hair as a few large masses with
medium tapered locks and only sparse flyaways, each lock connected root-to-tip.
Avoid: clipped highlights, white-on-white loss of silhouette, broken or sketchy contours, tangled
equal-weight hair lines, hair as an undivided blurry mass, dense micro-detail, thick outlines, background clutter.
```

## 4. 如果你追求极限线稿质量

单次生成有上限。Sol 的判断：严格复刻参考线稿逻辑 + 全新主体 + 浅色画面仍分层 + 每处发束都连贯，四者很难同时满足。两条补救路径：

1. **结构图 → 风格重绘（推荐）**：第一步生成/提供低饱和平涂结构图（`clean connected anime linework, grouped hair locks, clear dress construction, simple value blocking, no watercolor texture, no haze`），第二步用 `edits` 把结构图 + 画风参考图一起喂进去，只加配色与渲染，并要求 `preserve the structure drawing's line connectivity`。`/images/edits` 最多 16 张图，这条路径可行。
2. **局部修复**：对成品裁出发丝/眼睛/衣领分别做高分辨率 `edit`，要求"连接断线、把散线并成有限发束、保脸与构图、不许新增发丝"。

传统超分只能改善像素边缘、不能推断发束拓扑，**不要用它当主要补救手段**。
