# sheya-style「上色不像」定位与 A/B 对照（2026-10-07）

用户反馈：**Gemini 直出时 sheya 的尖锐眼睛已经像了，但参考图那种高饱和度的上色还不像。**
本文只记录结论与入口，完整证据（对齐表、脚本、对照图、请求回放）在
`data/test-result/20261007/sheya-color-audit/README.md`。

## 一、量出来的事实

最近 3 天（10-05/06/07）sheya 一共只有 7 张（10-05 与 10-06 的发布记录里一张都没有，
10-06 那 3 张来自 02:16 的 style-lab）。对着 `data/style-ref/sheya.png` 量：

| | 饱和度均值 | 鲜艳像素(>0.5) | colorfulness | 明度 L | 暗线占比 |
|---|---|---|---|---|---|
| 参考图 | 0.465 | 43.4% | 56.9 | 50.0 | 7.8% |
| app·窗边 ×4 | 0.094–0.150 | ≈0% | 17–20 | 80–82 | 2.9–4.7% |
| app·樱风神社 | 0.342 | 20.6% | 46.2 | 62.4 | 3.0% |
| style-lab ×3 | 0.311–0.348 | 13–15% | 41–43 | 46–58 | 2.3–18.7% |

**输出的饱和度跟着内容走、不跟参考图走**；而且 app 自己的首图审计早就把这条写出来了
（`rendering_language` major，置信 0.88/0.94/0.99，expected「Highly saturated…」
observed「Clean pale…」），**第二张是带着这条原因重画的，仍然 fail**。

## 二、根因

Gemini 直出的 parts 是 `[主次声明+画风规格+内容锚] → 参考图 → [就图执行+内容锁]`
（`modules/others/api_backend.py:2612-2635`），而 `prompts/gemini-analysis-content-lock.md`
第 2 行写着「sole authority for **intrinsic colours**…**this content wins**…
**Apply its line and shading technique** to the requested objects **in their own colours**」。

- `line and shading technique` → 线条/眼睛语法放行（规格里唯一逐条写死的正是眼睛）；
- `in their own colours` + `this content wins` → **上色被判给内容**。

gpt 首图那条链的 `prompts/analysis-style-reference.md:7` 有「内容里的渲染形容词不得覆盖
参考图画法」的护栏，**Gemini 直出这条链没有对等物**。

## 三、A/B 对照（两轮共 19 次真实出图）与落地

六个臂：V0 现状 / V1 画风规格 v2 / V2 规格 + 内容锁改一句 / V3 再加图后「渲染收尾权威」块 /
V4 规格 v3「强一档」/ V5 规格 v4「夸张」。固定 2 个内容锚 + 同一张参考图，走 App 真实组装函数，
每格一张、不跑审计。

| 内容源 | V0 | V1 (v2) | V2 | V3 | **V4 (v3)** | V5 (v4) |
|---|---|---|---|---|---|---|
| shrine（内容有色） | 41.9 / 19.8% | 47.1 / 30.0% | 44.9 | 43.4 | **57.9 / 39.8%** | 49.3 / 26.6% |
| window（高调浅色） | 17.2–24.0 | 18.0 | 15.7 | 18.8 | **21.2** | 17.0 |

参考图基准 56.9 / 43.4%。**V4 在有色内容上打平参考图**（Δfly +1.1，饱和度 0.462 vs 0.465），
读图确认是硬边平涂 + 更饱和的裙子/鸟居/天空，没有变荧光；window 也从 18.0 回到 21.2。
**V5 两格都回落** —— 这一族条款存在峰值，不是越夸张越好。
独立判读（App 首图审计）：shrine 上 V1/V4/V5 各 2 张全 pass（无复制、内容无损）；
window 全 fail，与第一轮同一结论：那格不是文案的问题。

**已落地**：`conf/config-styles.json` → `sheya-style` 的 `prompt` / `prompt_compressed` /
`prompt_gemini` 三份同步换成 **v3-push（1851 字符）**，其余字段一个字节没动；
备份 `conf/config-styles.json.bak-20261007-sheya-color`（最初）与
`conf/config-styles.json.bak-20261007-sheya-v4`（换成 v3 之前）；`tools/sync_styles_to_submodule.py`
同步到画风子模块。**V2 / V3 / V5 实测为负效果，未采用**；`prompts/`、任何 `.py`、
任何全局口径都没改。**改完必须重启 app**。

**未解决**：高调浅色内容到 V4 也只回到 21.2（参考 56.9），审计仍 fail。下一步线索按性价比排序：
审计重画指令换口径（重画正文里嵌的仍是 content wins）→ 内容锚侧的照明措辞（窗边锚里 10 个低饱和词，
属分析链）→ 参考图配技法/配色条。
