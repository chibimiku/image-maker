# 材质表面定向过一遍（2026-10-06）

触发：用户反馈 `data/20261006/2be26f5b_214351-625af8.jpg`（画风 `satou_kuuki-style-v2`，参考图 `data/style-ref/satou_kuuki.jpg`）"皮革太平涂、没有参考图那种光泽"。

## 一、为什么直出会平

三个方面叠加，**不是提示词写漏了 leather**：

1. **模型对材质单词只给中性解释**。风格条目的 `prompt`/`prompt_gpt` 都是画法与配色层面的描述，没有材质光泽口径；`satou_kuuki-style-v2` 的 `gemini_content_field` 还是 `gpt_image_prompt_short`（484 字符），内容锚里只有"粉色系带厚底靴"，没有任何表面响应要求。模型于是给出"粉色皮革"的默认平涂。
2. **参考图没有参与生图**（日志 21:43:22 的请求只有 1 张图 = 画风参考图；源图未附）。参考图上那种亮色漆面皮革（紫红色亮面套装）的光泽结构，模型只能从画风文字里猜到。
3. **保守重绘固件与"加高光"天然冲突**：`repaint-system-conservative-v5.md` 第 5 条与 FORBIDDEN 明确禁止"额外高光、高光扩散、增强对比"。把材质层叠在它后面，实测高光几乎没变化（见下 §四 v1）。

## 二、做法

新增一条**只改表面材质**的定向通道，不碰几何、身份、配色：

- `utils/gpt_image_optimize.py`
  - `MATERIALS` 词汇表（**跨任务通用，只描述材质外观，不认物件**）：`patent_leather`（漆皮/高光皮革）、`matte_leather`、`satin_leather`、`fur`、`lace`、`fabric_satin`、`sheer`（薄纱/丝袜）、`metal`、`wood_varnish`。每项 = 中文名 + 英文材质表述 + 若干条**只描述表面响应**的渲染条款（高光形状/反射带/明暗落差/微观纹理/缝线接触阴影）。条款里只用通用几何词（边缘、形体、折痕、缝线、内侧面），**不出现靴子/裙子/楼梯这类本次任务名词**。
  - `build_material_prompt(material, targets=..., spec=...)`：材质固件 + 该材质条款；未知材质抛 `ValueError`（不静默兜底）。
  - `DEFAULT_MATERIAL_TARGETS`：目标按**材质外观**指认（"image 里已经画成这种材质的每一个表面——不管它是什么物件"），并强制带"其余材质一处都不动"的兜底句。
  - `normalize_material_spec` / `resolve_material_info`：自定义材质规格（`MATERIAL_SPEC_KEYS` = `label` / `material` / `clauses`），键值全是自然语言。
  - `material_names()` / `describe_material()`。
- `prompts/gpt-image-optimize/material-surface-base.md`（新固件）：与 v5 同样的几何/身份/配色锁，但**不禁止**具名表面上的高光与材质结构；正文里写死"目标由**已经画成什么材质**决定，不按物件清单" + "目标之外不得增加光泽/对比/新纹理，也不得丢失已有细节"。
- `tools/repaint_material.py`（新 CLI）：
  ```
  python tools/repaint_material.py --list
  python tools/repaint_material.py --image <产物> --material patent_leather --dry-run
  python tools/repaint_material.py --image <产物> --material patent_leather \
      --reference data/style-ref/satou_kuuki.jpg [--repeat 2]
  # 精确点读某个部位时才需要 --targets（默认不认物件）
  python tools/repaint_material.py --image <产物> --material patent_leather \
      --targets "the panel on the left chest"
  # 新材质不必改代码，直接描述（跨任务）
  python tools/repaint_material.py --image <产物> \
      --material-spec "material=aged bronze; label=做旧青铜; clauses=Add green patina in the recesses only|Keep raised edges brighter"
  ```
  `--reference` 只是**材质参考**：v5 固件的 OPTIONAL MULTI-IMAGE MODE 与材质固件都写了"只借材质处理方式，不搬内容"。产物落 `data/<日期>/gpt_image_repaint-material/`。
  回归锁在 `tests/test_material_repaint.py`（13 项）：内置目标/条款里出现物件名、缺"其余不动"兜底句、缺几何/身份/配色锁、自定义材质规格被静默忽略 —— 任何一条都会变红。

## 三、结果（1 → 3 轮，同一张图）

| 轮次 | 参数 | 观察 |
|---|---|---|
| v1 | `material_222424-d7413b.jpg`（材质层叠在 **v5 保守固件**后面，2 张参考） | 靴头/厚底出现一点湿亮，靴筒仍然平；整体被 v5 的"不许加高光"锁住 → **证明两层直接叠是错的** |
| v2 | `bootgloss_222931-9a2910.jpg`（改用材质固件 + 强化条款 + 指定"整只靴筒"） | 靴头、厚底高光成型，明暗落差变陡；靴筒仍偏哑光 |
| v3 | `shaftgloss_223044-260c0b.jpg`（在 v2 上再点一次，targets 只写"靴筒/系带/脚踝，跟靴头一样的漆皮行为"） | 靴筒出现连续纵向高光带 + 内侧反射带 + 折痕压暗，**漆皮质感成立**；蝴蝶结、系带、绒球、水滴、裙摆、背景、脸部均保持 |

选定产物：`data/20261006/2be26f5b-material-patent_leather.jpg`（= v3 的副本，2700 双联对比见 `data/20261006/material-compare-shaft.png`）。

**结论：单次材质过一遍不够稳，靴筒这种大平面需要两轮（先整体、再点读）**。每一轮都会重新编码整幅画面，几何保持靠固件约束，**不保证像素级不变**。

## 四、边界与未做

- **通用性由用户 2026-10-06 指出后专门整改**：首版把"靴子/裙摆/台阶"写进了默认目标（`the footwear and any garment part...`），换个任务就会指错对象。现在默认目标一律按**材质外观**指认，物件名只允许出现在调用方显式传的 `--targets` 里；`tests/test_material_repaint.py` 用一份物件名词表（靴/鞋/裙/袖/楼梯/栏杆/窗帘/球球/木地板…）扫描全部内置目标与条款，命中即失败。
- 只验证了这一张图、这一个材质（`patent_leather`）。其余材质键（`matte_leather`/`satin_leather`/`fur`/`lace`/`fabric_satin`/`sheer`/`metal`/`wood_varnish`）**只有词汇表与离线预检，没有实图验证**。
- 没有改风格配置（`conf/config-styles.json` 未动）、没有改 v5 重绘固件、没有改 `repaint-system-conservative-*` 的既有行为。
- 没有把"皮革光泽"写进生成阶段：直出仍会平，除非做下面 §五 的根治。
- 本轮共 **3 次付费 Gemini 重绘**（v1/v2/v3，22:22–22:30），另有一张 v1 之前的同名产物；账目见 `log/2026-10-06.log`。

### 测试环境注意（踩过）

跑测试**不要**在终端里设 `PYTHONIOENCODING=utf-8`：它会继承给 `tests/test_style_metrics*.py` 拉起的子进程，而父进程仍按 GBK 解码，4 项 CLI 用例会以 `UnicodeDecodeError` 假失败（`--ignore` 掉也确认过：不带该变量时全绿）。这个变量是我为看中文日志临时加的，不代表代码问题。

## 五、直出根治：**只有 satou_kuuki-style-v2 默认带皮革质感**（已做）

用户 2026-10-06 追加要求：材质默认必须**只管这一个画风**，别的画风一个字都不许加。

### 落点：画风字段 `gemini_surface_clauses`

`generation_clauses` 只进 gpt-image 通道，而用户跑的是 Gemini 直出，所以给画风条目加了一个**新字段** `gemini_surface_clauses`（材质/表面条款，Gemini 直出用）：

- `utils/styles.py::normalize_style_entry` 保留该字段（最多 6 条）；`surface_clauses_for()` 读取；`surface_block_from_clauses()` 渲染成块。
- **拼装点在 `build_ref_gen_params` 的 `post`**（= 所有参考图之后的执行要求）。特意放在那里：
  - 图前会和画风参考图抢"谁是权威"（参考优先模式下参考图说了算），图后才生效；
  - **不加第四个返回值**：`build_ref_gen_params` 有 20 多处调用方按 3 个值解包，改签名会全线炸（这次先踩后修）。
- 块模板 `prompts/gemini-style-surface-authority.md`，块内自带四条边界：只按内容已有的材质生效、整块材质一起处理（不许一半亮一半哑）、内容明确写了别的表面处理时以内容为准、**绝不许产出内容里没有的材质**。
- 排序：`post`（含材质块）→ 内容锁 `gemini-analysis-content-lock.md`，即**材质块在内容锁之前**，测试有断言。
- 日志：`[Gemini 画风] 图后追加 N 条材质/表面条款（satou_kuuki-style-v2 自带…）`。

### 写进 satou 的三条条款

1. 内容里已经是皮革/涂层表面的部位 → 默认高光皮革（连续高光带 + 环境反射带 + 陡明暗落差 + 细粒与缝线接触阴影，不许平塑料纸）。
2. 该画风本就有的湿亮族（皮肤/袜/部分衣片的液滴与湿反光）保持液体拉长，不许变干哑。
3. 同画面里的丝缎/蕾丝/毛绒各保持自己的柔和响应，**不许套皮革高光**；内容里已经画成哑光的材质不扩散处理。

### 验证

- 全库 38 个画风里**只有 `satou_kuuki-style-v2`** 带 `gemini_surface_clauses`（回归会断言这条）。
- `tests/test_style_surface_clauses.py`（11 项）：只 satou 有值、条款不含几何/构图改写、未知画风返回空、`build_ref_gen_params` **仍是 3 个返回值**、图前说明里不出现 `patent leather`、材质块排在内容锁之前、无条款画风的 postamble 与改动前一致、gpt 通道同样拿到该块。
- 相关套件 122 项通过；全量 `pytest tests/ -q --ignore=tests/test_style_ref.py` = **1325 passed / 9 failed**（那 9 项改动前就红，零新增回归）。
- 回滚：`conf/config-styles.json.bak-20261006` 是改前副本；也可以只删 `satou_kuuki-style-v2.gemini_surface_clauses` 这一个键（其余字段未动）。

### 仍未做

1. **参考图参与生图**：Gemini 直出仍只挂画风参考图、不附源图；参考图上的漆皮质感只当"画法"看（这部分靠上面的条款补）。
2. **GUI 入口**：材质过一遍只有 CLI；要进界面应挂「gpt-image-2 专用页」的模式下拉（新增"材质修复"）或重绘参数区，而不是再加常驻行。
3. **只在线验证过 `patent_leather`**：本次 satou 默认条款也没有在线跑过，第一次实图要留意"皮革亮了、丝缎/蕾丝有没有被带亮"。
