# -*- coding: utf-8 -*-
"""生成「昨晚 → 今天凌晨」这一批实验的横向对比 HTML（单文件、图内嵌）。

覆盖 data/20260920（tid 画风 / 真实提示词 / Luna / Sol v5-v6）+ data/20260921（
线条连续性 / 锚图 / 重绘固件四轮 / 工作流）。
每个批次一张图例卡 + 指标表 + 实际提示词（可展开）。
输出：docs/gpt-image-tid-style/overnight-20260920-21.html
"""
import base64
import glob
import html
import io
import json
import os
import re
from datetime import datetime

from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "docs", "gpt-image-tid-style")
OUT = os.path.join(D, "overnight-20260920-21.html")

# ---------------------------------------------------------------- 批次定义
BATCHES = [
    {
        "id": "b1-tid-style",
        "order": 1,
        "title": "① tid 画风 · 四种模式 × 尺寸边界（9-20 02:xx）",
        "dirs": ["data/20260920/gpt-image-tid"],
        "desc": "全量 11k / 压缩 1.8k / 参考优先 / 头部插入 / 图文交错 / 短说明 / 极端长文（31k、40k）。"
                "结论：官方 prompt 上限 32000 字符，实测 11k~40k 全都能出图；长文本的真实代价是「参考图失效 + 稀疏主体崩」。",
        "metrics": "docs/gpt-image-tid-style/similarity_all.txt",
    },
    {
        "id": "b2-verify",
        "order": 2,
        "title": "② 真实提示词验证 · 4 主题 × 3 注入方式（9-20 03:xx）",
        "dirs": ["data/20260921/tid-verify"],
        "desc": "同一批真实主体，对比「短说明+图(V1) / 全量13k+图(V2) / 全量13k纯文本(V3)」。"
                "均值：V1 配色 0.390、V2 0.247、V3 0.119。",
        "metrics": "docs/gpt-image-tid-style/verify_real_prompts.json",
    },
    {
        "id": "b3-luna",
        "order": 3,
        "title": "③ Luna 建议轮 · 8 变体 + 无主体裁剪/风格板（9-20 04:xx）",
        "dirs": ["data/20260921/tid-luna"],
        "desc": "R5 结构化字段版（当时最优）胜出；R2/R4 无主体裁剪能掐掉主体泄漏但会放大留白。",
        "metrics": "docs/gpt-image-tid-style/luna_round.json",
    },
    {
        "id": "b4-sol-v5v6",
        "order": 4,
        "title": "④ Sol 修正 → 最终版 v6 · 跨主体 + 跨画风（9-20 05:xx~06:xx）",
        "dirs": ["data/20260921/tid-sol", "data/20260921/tid-sol-cross", "data/20260921/tid-sol-v5", "data/20260921/tid-sol-v6"],
        "desc": "Sol 指出「参考原画不是水彩为主、是清晰线稿+平涂」，据此改掉 clean white / soft even light / fine lineart 三处写法，"
                "并加发丝分组与三尺度细节预算 → 中灰带 0.04→0.68、高光裁切归零、双手可见。",
        "metrics": "docs/gpt-image-tid-style/render_metrics_v6.json",
    },
    {
        "id": "b5-line",
        "order": 5,
        "title": "⑤ 线条连续性控制变量 · 只换参考图（9-21 02:xx）",
        "dirs": ["data/20260921/line-continuity", "data/20260921/line-continuity-fix",
                 "data/20260921/line-continuity-refprep"],
        "desc": "决定线条连续性的是参考图自身的边缘语言，不是分辨率：say-hana 放大到 1000px 反而更差（84.3/28.3）；"
                "纯文本（123.6/18.74）比给软边参考图更好；文字补救与像素级预处理都无效。",
        "metrics": None,
    },
    {
        "id": "b6-repaint",
        "order": 6,
        "title": "⑥ Gemini 重绘固件四轮迭代（9-21 02:1x~03:2x）",
        "dirs": ["data/20260921/tid-sol-v6-repaint", "data/20260921/tid-sol-v6-repaint-neutral",
                 "data/20260921/repaint-v2", "data/20260921/repaint-v3"],
        "desc": "cel 旧固件会把水彩改写成平涂（say-hana 头部配色 0.22）；中性 v2 保真最好但清线弱；"
                "v3 只改色彩（修好工作流过淡但方差大）；v4 色彩+线条 = 当前默认；v5-dual 用锚图双参考清线最强但保真下降。",
        "metrics": None,
    },
    {
        "id": "b7-anchor",
        "order": 7,
        "title": "⑦ 锚图两阶段（Sol 方案①）v1 vs 改良版（9-21 03:0x~03:2x）",
        "dirs": ["data/20260921/anchor-test", "data/20260921/anchor-v2"],
        "desc": "改良锚图提示词把「锚图与源图配色相似度」从 0.445 提到 0.961；锚图当参考图后 say-hana 段长 99.8→120.5、"
                "noir 98.1→118.6。",
        "metrics": None,
    },
    {
        "id": "b13-hair-region",
        "title": "⑬ 头发专项：局部重绘做成可选区域（脸 / 发 / 裙），三族头发带实测（9-22 01:1x~01:2x）",
        "dirs": ["data/local-repaint"],
        "desc": "用户反馈 say-hana / noir 重绘后头发还是乱——这些组确实走过重绘（v5 固件），但 v5 按 Sol 的分级原则"
                "故意不给绘画型发丝加强连续性。于是把局部重绘做成区域可选（head/hair/face/skirt/upper/full）+ 区域强调句"
                "（hair 条款要求主发束根到尖连续、碎发并入主束、保持可数分组）。头发带指标（上部 45%）："
                "say-hana 段长 111.4→117.9、端点 19.96→17.53；noir 121.1→**147.2**、端点 18.81→**15.01**；tid 230.2→244.3。"
                "顺带修掉 post_process 漏传 use_detail_suffix=False 导致旧 cel 后缀被追加的 bug。",
        "metrics": None,
    },
    {
        "id": "b12-third-pass",
        "order": 12,
        "title": "⑫ 第三件事：结构线叠加 + 局部重绘贴回（noir / say-hana / tid 三族）+ 短锚档字段（9-22 01:0x~01:2x）",
        "dirs": ["data/local-repaint", "data/20260922/analysis-gptimage"],
        "desc": "③-a 结构线叠加（纯本地，不调模型）：段长 +6%（say-hana）/ +17%（noir）/ +22%（tid），"
                "端点密度最多 −16%，配色与饱和度几乎不动 → 接近免费的线条连续性收益，推翻了「tid 没提升空间」的判断。"
                "③-b 局部重绘 + 羽化贴回（裁上半身 62% 放大 1.5 倍重绘再贴回）：段长 +4~7%，可与叠加串联。"
                "① 短锚档 + 参考图角色排除句：饱和度 44→25.7、与 tid 画风相似度 0.107→0.323。",
        "metrics": None,
    },
    {
        "id": "b11-v5-and-wf",
        "order": 11,
        "title": "⑪ v5 重绘固件（Sol 第 7 轮三处保守化）+ 分析产物挂画风参考图实测（9-22 00:0x~00:2x）",
        "dirs": ["data/20260922/repaint-v5", "data/20260922/analysis-gptimage"],
        "desc": "v5 = v4 + 禁止凭先验补全不可见肢体 + 眼部保留式修复 + 主结构/次级笔触分级："
                "修掉了 v4 凭空补手的缺陷（say-hana 腹部），头部 MAE 三例全部改善（3.5→3.4 / 3.7→3.2 / 3.0→2.3），"
                "线条指标小幅回落（段长 −1~8%）→ 已设为默认固件。"
                "W1~W3 = `prompt_gpt`(543) + 分析产物的 `gpt_image_prompt`(1271) = 1816 字符，"
                "挂/不挂 tid 画风参考图的画风相似度几乎一样（0.107 vs 0.130）→ 长文本会压住参考图；"
                "W4 把内容锚压到 ~500 字符后画风相似度翻倍到 0.2731、白底 0.126→0.221。",
        "metrics": None,
    },
    {
        "id": "b9-line-v2",
        "order": 9,
        "title": "⑨ Sol A/B/E 三组验证 · 硬化双参考 / 三参考+结构线 / tid 双参考（今晚 23:4x）",
        "dirs": ["data/20260921/line-v2", "data/20260921/repaint-v3"],
        "desc": "A = say-hana 双参考 + 可验证结构要求；B = 三参考（源图 + 锚图 + 本地 OpenCV 生成的结构线图）；"
                "E = tid 双参考基线。结论：A ≈ v5-dual（提示词硬化无效）；B 线条没更好但保真回升；"
                "E 给 tid 提升线条（段长 221→270）却明显掉保真（配色 0.947→0.871）——印证 Sol 的「水彩厚涂与线条连贯存在原理上限」。",
        "metrics": None,
    },
    {
        "id": "b10-wf-analysis",
        "order": 10,
        "title": "⑩ 分析工作流改造实测 · 分析 prompts → gpt-image-2 → Gemini 重绘（今晚 23:4x~23:5x）",
        "dirs": ["data/20260921/workflow-analysis", "data/20260921/sucai"],
        "desc": "现有产物组合（11k 画风 + 4.4k 描述 = 15.7k 字符）→ **连接被重置**，复现了「太长」；"
                "短画风 + 全量描述 5.0k 成功；压到 1.7k 成功但内容跑偏；挂素材照更保真。"
                "两版再走 Gemini 重绘 v4 固件后段长 +20~27%、端点密度 −22~31%。",
        "metrics": None,
    },
    {
        "id": "b8-workflow",
        "order": 8,
        "title": "⑧ 常用工作流实测 · 素材照 → gpt-image-2(tid) → Gemini 重绘（9-21 02:2x~03:2x）",
        "dirs": ["data/20260921/workflow-tid-photo", "data/20260921/sucai"],
        "desc": "全身构图重绘被中转站内容审核拦，改腰以上一次通过；头部 MAE 14.6、头部配色 0.8896。"
                "顺带修掉一个真 bug：Gemini generateContent 的 contents 缺 role → 400。",
        "metrics": None,
    },
]

# ---------------------------------------------------------------- 指标表（手工整理的权威数字）
TABLES = {
    "b1-tid-style": {
        "caption": "同主体（少女站浅水）只换画风注入方式 · 指标来自 similarity_all.txt / fingerprint.json",
        "cols": ["变体", "注入方式", "文本字符", "HSV配色↑", "亮度", "饱和", "白底", "边缘密度", "主体"],
        "rows": [
            ["B5", "压缩 1.8k + 图", "1855", "0.196", "169", "72", "0.075", "0.099", "好（但搬了粉发金瞳）"],
            ["G2", "校准 678 + 图（2.5-flare）", "678", "0.395", "226", "28", "0.393", "0.085", "好"],
            ["C", "手写短说明 305 + 图", "402", "0.285", "222", "26", "0.318", "0.075", "好"],
            ["B1", "参考优先（压缩+图）", "4195", "0.106", "136", "99", "0.057", "0.105", "好"],
            ["G1", "校准 678 + 图", "678", "0.287", "221", "27", "0.289", "0.078", "好"],
            ["B3", "图文交错（全量 13k）", "13232", "0.214", "170", "49", "0.096", "0.100", "好"],
            ["B2 / B4", "头部插入（全量 13k）", "13232", "0.057 / 0.274", "127 / 186", "94 / 47", "0.025", "0.120", "好"],
            ["A1", "全量 11k 纯文本", "11283", "0.015", "96", "157", "0.019", "0.145", "**崩**（水下精灵）"],
            ["H1 / H2", "31k / 40k 灌水", "31000 / 40000", "0.033 / 0.155", "130 / 203", "65 / 39", "0.012", "0.047", "**崩**"],
            ["G4", "校准 678 无图", "678", "0.200", "200", "27", "0.134", "0.089", "好（退回 2.5D）"],
            ["E1", "仅参考图、无画风字", "95", "0.058", "163", "56", "0.029", "0.068", "好"],
            ["D", "无画风（基线）", "95", "0.018", "115", "54", "0.001", "0.043", "好"],
        ],
    },
    "b2-verify": {
        "caption": "4 个真实主题 · 均值（V1 短说明+图 / V2 全量+图 / V3 全量纯文本）",
        "cols": ["变体", "HSV配色↑", "亮度", "饱和", "白底", "边缘密度差↓", "耗时"],
        "rows": [
            ["**V1 短说明 + 图**", "**0.390**", "**223**", "**27**", "**0.454**", "**0.025**", "36~44s"],
            ["V2 全量 13k + 图", "0.247", "185", "45", "0.172", "0.039", "37~42s"],
            ["V3 全量 13k 纯文本", "0.119", "146", "77", "0.070", "0.085", "63~274s"],
        ],
    },
    "b3-luna": {
        "caption": "Luna 建议轮 · 指标来自 luna_round.json（同一主体）",
        "cols": ["变体", "做法", "HSV配色↑", "亮度", "白底", "边缘密度"],
        "rows": [
            ["B5", "对照：压缩 1.8k + 原图", "0.196", "169", "0.075", "0.099"],
            ["G2", "对照：校准 678 + 原图(2.5-flare)", "0.395", "226", "0.393", "0.085"],
            ["C", "对照：短说明 305 + 原图", "0.285", "222", "0.318", "0.075"],
            ["R5", "**结构化字段版（当时最优）**", "**0.644**", "**239**", "**0.686**", "**0.036**"],
            ["R1", "角色分工版 + 原图", "0.476", "226", "0.439", "0.072"],
            ["R4", "角色分工 + 风格板", "0.469", "228", "0.486", "0.060"],
            ["R2", "角色分工 + crop_white", "0.530", "228", "0.499", "0.062"],
            ["R7", "角色分工 + quality=high", "0.296", "215", "0.222", "0.093"],
            ["G4", "字段版无参考图", "0.200", "200", "0.134", "0.089"],
        ],
    },
    "b4-sol-v5v6": {
        "caption": "Sol 修正 → v6 · 指标来自 render_metrics_v6.json（相对 tid 参考原画）",
        "cols": ["版本", "模型/画质", "中灰带↑", "高光裁切↓", "局部RMS↑", "细碎边缘↓", "骨架段长↑", "端点密度↓", "手"],
        "rows": [
            ["参考原画 tid", "—", "0.152", "0.435*", "34.6", "0.478", "92.4", "14.97", "—"],
            ["R5（旧最优）", "gpt-image-2 medium", "0.041", "0.312", "12.4", "0.778", "78.1", "28.75", "无"],
            ["S1", "Sol 完整版 high", "0.480", "0.037", "28.1", "0.639", "114.3", "23.81", "有"],
            ["V5-A", "Sol 最终版 medium", "0.265", "0.036", "19.8", "0.587", "104.3", "22.03", "无"],
            ["**V6-H**", "**v6 + high**", "**0.681**", "**0.001**", "25.8", "0.566", "**145.5**", "**17.35**", "**双手清晰**"],
            ["V6-M", "v6 + medium", "0.704", "0.0001", "26.7", "0.597", "141.0", "18.49", "双手清晰"],
            ["V6-H(flare)", "v6 + 2.5-flare high", "0.616", "0.000", "26.9", "0.602", "140.4", "18.90", "双手清晰"],
        ],
    },
    "b5-line": {
        "caption": "线条连续性控制变量：同 v6 提示词、同主体，只换参考图",
        "cols": ["参考图", "参考图宽", "骨架段长↑", "端点密度↓", "长线占比↑", "结论"],
        "rows": [
            ["tid 原图", "1000", "**141.5**", "**18.1**", "0.898", "基线（线条连续）"],
            ["say-hana 原图", "832", "99.8", "25.3", "0.837", "柔边 → 输出糊"],
            ["say-hana **放大到 1000px**", "1000", "**84.3**", "**28.3**", "0.790", "**放大反而更差 → 不是分辨率**"],
            ["noir 原图", "816", "98.1", "25.5", "0.822", "柔边 → 输出糊"],
            ["noir 放大到 1000px", "1000", "101.1", "25.8", "0.843", "基本无变化"],
            ["**不给参考图（纯文本）**", "—", "123.6", "18.7", "0.887", "**比给软边图更好 → 是参考图带糊的**"],
            ["ajicoma（1664px，线条清晰）", "1664", "127.7", "18.8", "0.876", "线条清晰就行，跟分辨率无关"],
            ["+ EDGE TRANSLATION 文字段", "—", "114.7 / 108.0", "25.5 / 25.6", "0.856 / 0.844", "文字几乎无效"],
            ["+ 参考图 unsharp/结构线", "—", "—", "—", "—", "像素级预处理也无实质改善"],
        ],
    },
    "b6-repaint": {
        "caption": "Gemini 重绘固件四轮 · 相对各自源图（cel = 旧固件）",
        "cols": ["用例", "固件", "头部MAE↓", "头部配色↑", "头部饱和偏差", "全图配色↑", "骨架段长↑", "端点密度↓", "长线占比↑"],
        "rows": [
            ["tid", "旧 cel", "6.5", "0.9525", "+1.7", "0.9520", "266.1", "8.90", "0.968"],
            ["tid", "v2 中性", "2.3", "0.9237", "+1.4", "0.8792", "212.5", "11.04", "0.954"],
            ["tid", "v3 只改色彩", "2.2", "0.9183", "+0.9", "0.8915", "208.3", "10.86", "0.951"],
            ["tid", "**v4 色彩+线条（默认）**", "3.0", "**0.9652**", "+1.0", "**0.9474**", "**221.0**", "**10.14**", "**0.963**"],
            ["say-hana", "旧 cel", "**63.8**", "**0.2207**", "−40.4", "0.7107", "196.4", "9.65", "0.942"],
            ["say-hana", "v2 中性", "3.4", "0.9660", "±0.0", "0.9814", "116.7", "16.32", "0.868"],
            ["say-hana", "v3 只改色彩", "**73.1**", "0.5738", "+22.3", "0.9869", "122.2", "16.30", "0.874"],
            ["say-hana", "**v4**", "3.5", "**0.9690**", "−0.1", "0.9826", "120.3", "16.18", "**0.878**"],
            ["noir", "旧 cel", "17.8", "0.3598", "−24.4", "0.5237", "184.4", "11.89", "0.935"],
            ["noir", "v2 中性", "3.2", "0.8784", "−1.2", "0.9662", "125.4", "17.50", "0.881"],
            ["noir", "**v4**", "3.7", "0.8589", "−1.9", "0.9674", "**133.6**", "**16.50**", "**0.888**"],
            ["工作流图", "v2 中性", "**56.6**", "0.7560", "**−18.5**", "0.9684", "108.7", "14.98", "0.861"],
            ["工作流图", "**v3 只改色彩**", "**17.8**", "**0.9179**", "−1.9", "0.9730", "101.2", "16.38", "0.852"],
            ["工作流图", "v4", "20.7", "0.6204", "+16.0", "0.9571", "92.3", "17.69", "0.809"],
            ["say-hana", "v5-dual（锚图双参考）", "12.2", "0.9702", "−3.8", "—", "**138.8**", "**15.12**", "**0.898**"],
            ["noir", "v5-dual（锚图双参考）", "13.0", "0.8063", "+4.2", "—", "**173.7**", "**11.42**", "**0.927**"],
        ],
    },
    "b7-anchor": {
        "caption": "锚图两阶段：旧锚图提示词 vs 改良锚图（保留身份/构图/色彩，只增强主要边缘）",
        "cols": ["指标", "旧锚图", "**改良锚图**", "直出（无锚图）", "双参考重绘"],
        "rows": [
            ["锚图与源图配色相似↑", "**0.4452**", "**0.9610**", "—", "—"],
            ["say-hana 锚图/产物 段长↑", "103.1", "110.9", "—", "138.8"],
            ["say-hana 端点密度↓", "22.24", "20.77", "—", "**15.12**"],
            ["say-hana 用锚图出图 段长↑", "120.5", "—", "116.7（v2）", "—"],
            ["noir 用锚图出图 段长↑", "118.6", "—", "125.4（v2）", "**173.7**"],
            ["noir 端点密度↓", "21.37", "—", "17.50（v2）", "**11.42**"],
            ["保真代价（头部 MAE）", "—", "—", "3.4（v2）", "12.2 / 13.0"],
        ],
    },
        "b13-hair-region": {
        "caption": "头发带（上部 45%、左右各留 5%）指标：源图 → v5 重绘 → 头发局部重绘 / 结构线对照",
        "cols": ["族 / 版本", "骨架段长↑", "端点密度↓", "长线占比↑", "细碎边缘↓", "备注"],
        "rows": [
            ["say-hana 源图（未重绘）", "84.5", "27.51", "0.813", "0.539", ""],
            ["say-hana v5 重绘（已重绘）", "111.4", "19.96", "0.851", "0.506", "v5 不强求发丝连续"],
            ["say-hana **+ 头发局部重绘**", "**117.9**", "**17.53**", "**0.869**", "0.490", "区域=头发 + 强调句"],
            ["say-hana + 结构线 0.50", "112.4", "19.52", "0.863", "0.513", "纯本地对照"],
            ["noir 源图（未重绘）", "94.7", "30.13", "0.808", "0.538", ""],
            ["noir v5 重绘（已重绘）", "121.1", "18.81", "0.872", "0.506", ""],
            ["noir **+ 头发局部重绘**", "**147.2**", "**15.01**", "**0.911**", "0.499", "**段长 +22%、端点 −20%**"],
            ["noir + 结构线 0.50", "144.1", "16.95", "0.900", "0.495", ""],
            ["tid 源图（未重绘）", "149.1", "19.12", "0.921", "0.580", ""],
            ["tid v5 重绘（已重绘）", "230.2", "11.12", "0.961", "0.550", ""],
            ["tid **+ 头发局部重绘**", "**244.3**", "**9.58**", "**0.967**", "0.550", ""],
            ["tid + 结构线 0.50", "255.2", "9.31", "0.960", "0.508", "tid 上叠加更强"],
        ],
    },
    "b12-third-pass": {
        "caption": "上：③-a/③-b 在后处理层面对线稿连通性的增益（底图 = v5 产物）；下：① 短锚档与排除句",
        "cols": ["族 / 变体", "骨架段长↑", "端点密度↓", "长线占比↑", "与源图配色↑", "备注"],
        "rows": [
            ["say-hana v5 基线", "118.7", "16.85", "0.870", "0.0406", ""],
            ["say-hana + 结构线 0.35 / **0.50**", "121.8 / **126.1**", "16.58 / **16.10**", "0.877 / 0.883", "0.0408 / 0.0410", "纯本地"],
            ["say-hana 局部重绘贴回", "123.5", "15.74", "0.877", "0.0394", "一次 Gemini"],
            ["say-hana 局部重绘 + 结构线 0.35", "124.1", "16.00", "0.880", "0.0396", "串联"],
            ["noir v5 基线", "122.8", "17.41", "0.876", "0.0078", ""],
            ["noir + 结构线 0.35 / **0.50**", "129.8 / **143.6**", "18.40 / **16.17**", "0.884 / 0.897", "0.0082 / 0.0082", "**段长 +17%**"],
            ["noir 局部重绘贴回 / + 结构线", "126.7 / **132.7**", "17.19 / 17.82", "0.886 / 0.891", "0.0085 / 0.0087", ""],
            ["tid v5 基线", "214.8", "11.12", "0.957", "0.0585", ""],
            ["tid + 结构线 0.35 / **0.50**", "243.4 / **261.3**", "11.07 / **9.37**", "0.962 / 0.963", "0.0581 / 0.0583", "**段长 +22%、端点 −16%**"],
            ["tid 局部重绘贴回 / + 结构线", "230.1 / **241.9**", "10.73 / 10.65", "0.959 / 0.962", "0.0562 / 0.0562", ""],
            ["素材照 W1 完整锚 1816（无排除句）", "81.1", "27.80", "0.777", "0.1065 (tid 相似)", "参考图被压住"],
            ["素材照 **W5 短锚 1325 + 排除句**", "83.4", "28.09", "0.778", "**0.2804 (tid 相似)**", "饱和 29.0"],
            ["素材照 W6 短锚 1025（无排除句）", "87.5", "32.51", "0.803", "0.2237 (tid 相似)", "饱和 32.9"],
            ["素材照 **W7 完整锚 2116 + 排除句**", "81.3", "27.76", "0.773", "**0.3234 (tid 相似)**", "**饱和 25.7 ≈ tid 的 25.0**"],
        ],
    },
    "b11-v5-and-wf": {
        "caption": "上：v4 vs v5 固件（相对源图）；下：分析→gpt-image（带/不带画风参考图）",
        "cols": ["用例", "固件/变体", "骨架段长↑", "端点密度↓", "长线占比↑", "与源图配色↑", "头部MAE↓", "备注"],
        "rows": [
            ["say-hana", "v4", "120.3", "16.18", "0.878", "0.9826", "3.5", "**腹部被凭空加了一只手**"],
            ["say-hana", "**v5**", "118.7", "16.85", "0.870", "**0.9851**", "**3.4**", "手不再凭空出现"],
            ["noir", "v4", "133.6", "16.50", "0.888", "0.9674", "3.7", ""],
            ["noir", "**v5**", "122.8", "17.41", "0.876", "0.9461", "**3.2**", ""],
            ["tid", "v4", "221.0", "10.14", "0.963", "0.9474", "3.0", ""],
            ["tid", "**v5**", "214.8", "11.12", "0.957", "0.9090", "**2.3**", ""],
            ["分析产物现状(P1)", "11k 画风+4.4k 描述 = 15662 字符", "—", "—", "—", "—", "—", "**连接被重置（太长）**"],
            ["素材照 W1", "1816 字符 + **挂 tid 参考图**", "81.1", "27.80", "0.777", "0.1065(tid 相似)", "—", ""],
            ["素材照 W3", "1816 字符 + 不挂图", "87.3", "24.46", "0.800", "0.1297(tid 相似)", "—", "参考图被长文本压住"],
            ["素材照 **W4**", "1042 字符 + 挂 tid 参考图", "**95.2**", "**24.11**", "**0.819**", "**0.2731(tid 相似)**", "—", "**短才算数；但会搬参考图角色特征**"],
            ["W1 + Gemini 重绘", "v4 固件", "100.9", "19.25", "0.840", "0.0743", "—", "重绘仍是最稳增益"],
        ],
    },
    "b9-line-v2": {
        "caption": "Sol A/B/E · 相对各自源图",
        "cols": ["版本", "骨架段长↑", "端点密度↓", "长线占比↑", "与源图配色↑", "头部MAE↓", "头部配色↑", "饱和偏差"],
        "rows": [
            ["say-hana 源图", "88.7", "24.86", "0.803", "1.0000", "0.0", "1.0000", "±0.0"],
            ["say-hana v4（对照）", "120.3", "16.18", "0.878", "0.9826", "3.5", "0.9690", "−0.1"],
            ["say-hana v5-dual（对照）", "138.8", "15.12", "0.898", "0.9702", "12.2", "0.9233", "−3.8"],
            ["say-hana **A 硬化双参考**", "137.8", "15.23", "0.898", "0.9696", "11.9", "0.9291", "−4.8"],
            ["say-hana **B 三参考+结构线**", "118.7", "16.29", "0.874", "**0.9846**", "11.1", "0.9486", "+1.3"],
            ["tid 源图", "140.4", "18.90", "0.913", "1.0000", "0.0", "1.0000", "±0.0"],
            ["tid v4（对照）", "221.0", "10.14", "0.963", "0.9474", "3.0", "0.9652", "+1.0"],
            ["tid **E 双参考**", "**269.6**", "**8.66**", "**0.966**", "**0.8710**", "8.7", "**0.8389**", "−2.6"],
        ],
    },
    "b10-wf-analysis": {
        "caption": "分析→生图→重绘：提示词长度与成败（素材照 = 2026-07-24 12_17_40_1.jpg）",
        "cols": ["变体", "提示词字符", "结果", "骨架段长↑", "端点密度↓", "长线占比↑", "说明"],
        "rows": [
            ["分析产物现状 = P1（11k 画风 + 4.4k 描述）", "**15662**", "**失败**", "—", "—", "—", "**连接被重置（复现「太长」）**"],
            ["P2 短画风 + 全量描述", "5010", "成功", "76.5", "27.87", "0.737", "文本生成，线条弱"],
            ["P3 短画风 + 压缩描述(1.2k)", "1722", "成功", "79.2", "29.41", "0.782", "返回 2 张；内容跑偏（换了场景）"],
            ["P4 短画风 + 全量描述 + 素材照参考图", "5010", "成功", "81.7", "27.09", "0.784", "对照：挂原图后内容保真最好"],
            ["P2 + Gemini 重绘（v4 固件）", "—", "成功", "**95.7**", "**18.82**", "0.814", "段长 +25%、端点 −32%"],
            ["P4 + Gemini 重绘（v4 固件）", "—", "成功", "**98.7**", "**18.68**", "**0.836**", "段长 +21%、端点 −31%"],
        ],
    },
"b8-workflow": {
        "caption": "常用工作流（素材照 → gpt-image-2 tid → Gemini 重绘）",
        "cols": ["步骤", "结果", "耗时", "头部 MAE↓", "头部配色↑", "备注"],
        "rows": [
            ["① gpt-image-2 全身构图 high", "成功", "46.1s", "—", "—", "画风与服装细节都保住"],
            ["② Gemini 重绘（全身）", "**被拦**", "—", "—", "—", "PROHIBITED_CONTENT，1K/JPEG/flash/短提示都无效"],
            ["①' gpt-image-2 腰以上 high", "成功", "36.1s", "—", "—", "改构图后一次通过"],
            ["②' Gemini 重绘（腰以上）", "成功", "79.2s", "**14.6**", "**0.8896**", "头部边缘 0.176→0.189，眼区 0.189"],
            ["顺带修 bug", "api_backend", "—", "—", "—", "Gemini contents 缺 role → 400，已补 role:user"],
        ],
    },
}

PREFIX_LABEL = {
    "tv-c1-v1": "C1 双人百合 · V1 短说明+图", "tv-c1-v2": "C1 双人百合 · V2 全量+图",
    "tv-c1-v3": "C1 双人百合 · V3 全量纯文本",
    "tv-c2-v1": "C2 舞蹈 · V1", "tv-c2-v2": "C2 舞蹈 · V2", "tv-c2-v3": "C2 舞蹈 · V3",
    "tv-c3-v1": "C3 洛可可 · V1", "tv-c3-v2": "C3 洛可可 · V2", "tv-c3-v3": "C3 洛可可 · V3",
    "tv-c4-v1": "C4 洛丽塔 · V1", "tv-c4-v2": "C4 洛丽塔 · V2", "tv-c4-v3": "C4 洛丽塔 · V3",
    "luna-r1": "R1 角色分工+原图", "luna-r2": "R2 crop_white", "luna-r3": "R3 crop_left",
    "luna-r4": "R4 风格板", "luna-r5": "R5 结构化字段版", "luna-r6": "R6 字段+身份排除",
    "luna-r7": "R7 角色分工+high", "luna-r8": "R8 字段版无图",
    "sol-s1": "S1 Sol 完整版 high", "sol-s2": "S2 Sol 完整版 medium", "sol-s3": "S3 精简版 medium",
    "cross-c2": "跨主体 舞蹈", "cross-c3": "跨主体 洛可可", "cross-c5": "跨主体 和服",
    "v5-v5a": "V5-A 浅水", "v5-v5b": "V5-B 舞蹈", "v5-v5c": "V5-C 和服",
    "v6-v6h": "V6-H 浅水(high)", "v6-v6m": "V6-M 浅水(medium)",
    "v6x-say_hana": "V6 跨画风 say-hana", "v6x-noir": "V6 跨画风 noir", "v6x-ajicoma": "V6 跨画风 ajicoma",
    "sg-tid": "prompt_gpt · tid", "sg-sayhana": "prompt_gpt · say-hana", "sg-noir": "prompt_gpt · noir-aiart",
    "lc-a": "A tid 参考图", "lc-b": "B say-hana 原图", "lc-c": "C say-hana 放大1000px",
    "lc-d": "D noir 原图", "lc-e": "E noir 放大1000px", "lc-f": "F 无参考图(纯文本)", "lc-g": "G ajicoma",
    "h-h1": "H1 文字段 say-hana", "h-h2": "H2 文字段 noir", "h-h3": "H3 say-hana high", "h-h4": "H4 noir high",
    "rp-e1": "E1 unsharp 参考图", "rp-e2": "E2 结构线 参考图",
    "v2-tid": "v2-tid", "v2-sayhana": "v2-sayhana", "v2-noir": "v2-noir", "v2-wf": "v2-wf",
    "v3-tid": "v3 色彩 tid", "v3-sayhana": "v3 色彩 say-hana", "v3-noir": "v3 色彩 noir", "v3-wf": "v3 色彩 工作流",
    "v4-tid": "v4 色彩+线条 tid", "v4-sayhana": "v4 色彩+线条 say-hana", "v4-noir": "v4 色彩+线条 noir",
    "v4-wf": "v4 色彩+线条 工作流",
    "v5-dual-sayhana": "v5-dual say-hana", "v5-dual-noir": "v5-dual noir",
    "anchor-sayhana": "旧锚图 say-hana", "anchor-noir": "旧锚图 noir",
    "anchor2-sayhana": "改良锚图 say-hana", "anchor2-noir": "改良锚图 noir",
    "ref-sayhana": "用旧锚图出图 say-hana", "ref-noir": "用旧锚图出图 noir",
    "wf-tid": "工作流 全身 生图", "wf-tid2": "工作流 腰以上 生图",
    "wf-tid-repaint": "工作流 全身 重绘(被拦)", "wf-tid2-repaint": "工作流 腰以上 重绘",
    "rp-01": "Gemini 重绘 cel · tid", "rp-02": "Gemini 重绘 cel · noir", "rp-03": "Gemini 重绘 cel · say-hana",
    "neutral_tid": "Gemini 重绘 中性v1 · tid", "neutral_sayhana": "Gemini 重绘 中性v1 · say-hana",
    "v6-a-sayhana": "A 硬化双参考 say-hana", "v6-b-sayhana": "B 三参考+结构线 say-hana",
    "v6-e-tid": "E 双参考 tid", "structline-sayhana": "本地结构线图", "anchor2-tid": "改良锚图 tid",
    "hair-sayhana": "say-hana 头发局部重绘（旧，带 cel 后缀）", "hair-noir": "noir 头发局部重绘（旧，带 cel 后缀）",
    "hair-tid": "tid 头发局部重绘（旧，带 cel 后缀）", "hair2-sayhana": "say-hana 头发局部重绘（已修）",
    "hair2-noir": "noir 头发局部重绘（已修）", "hair2-tid": "tid 头发局部重绘（已修）",
    "overlay-sayhana-020": "say-hana + 结构线 0.20", "overlay-sayhana-035": "say-hana + 结构线 0.35",
    "overlay-sayhana-050": "say-hana + 结构线 0.50", "overlay-noir-020": "noir + 结构线 0.20",
    "overlay-noir-035": "noir + 结构线 0.35", "overlay-noir-050": "noir + 结构线 0.50",
    "overlay-tid-020": "tid + 结构线 0.20", "overlay-tid-035": "tid + 结构线 0.35",
    "overlay-tid-050": "tid + 结构线 0.50", "local-sayhana": "say-hana 局部重绘贴回",
    "local-noir": "noir 局部重绘贴回", "local-tid": "tid 局部重绘贴回",
    "combo-sayhana": "say-hana 局部重绘+结构线", "combo-noir": "noir 局部重绘+结构线",
    "combo-tid": "tid 局部重绘+结构线",
    "w5": "W5 短锚+排除句", "w6": "W6 短锚无排除句", "w7": "W7 完整锚+排除句",
    "v5-sayhana": "v5 固件 say-hana", "v5-noir": "v5 固件 noir", "v5-tid": "v5 固件 tid",
    "w1": "W1 挂画风参考图(1.8k)", "w2": "W2 挂画风参考图 medium", "w3": "W3 不挂图(1.8k)",
    "w4": "W4 短内容锚+挂图(1.0k)", "w1-repaint": "W1 + Gemini 重绘",
    "wf-p1": "P1 全量15.7k（失败）", "wf-p2": "P2 短画风+全量描述", "wf-p3": "P3 短画风+压缩描述",
    "wf-p4": "P4 +素材照参考图", "wf-p2-repaint": "P2 + Gemini 重绘", "wf-p4-repaint": "P4 + Gemini 重绘",
    "original": "输入素材照",
}


def img_uri(path, max_edge=760, quality=80):
    if not os.path.isfile(path):
        return None
    try:
        if os.path.getsize(path) < 1024:
            return None
        im = Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    w, h = im.size
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def key_of(basename):
    if "_output" in basename:
        return basename.split("_output")[0]
    return os.path.splitext(basename)[0]


def label_of(key):
    return PREFIX_LABEL.get(key, key)


def prompt_of(name, key):
    prompt_dir = os.path.join(BASE, "data", "gpt-image-tid", "prompts")
    log_dir = os.path.join(BASE, "data", "gpt-image-tid", "logs")
    if name == "b1-tid-style":
        cands = glob.glob(os.path.join(prompt_dir, f"{key}*.txt"))
    elif name == "b2-verify":
        c = key.split("-")[1].upper() if "-" in key else key.upper()
        cands = glob.glob(os.path.join(log_dir, f"verify_{c}_*.log"))
    else:
        cands = (glob.glob(os.path.join(log_dir, f"*{key}*.log"))
                 + glob.glob(os.path.join(prompt_dir, f"{key}*.txt")))
    for p in sorted(cands, key=os.path.getsize, reverse=True):
        text = open(p, encoding="utf-8", errors="replace").read()
        m = re.search(r'"prompt":\s*"((?:[^"\\]|\\.)*)"', text, re.S)
        if m:
            try:
                body = json.loads('"' + m.group(1) + '"').strip()
                if len(body) > 80:
                    return body
            except Exception:  # noqa: BLE001
                pass
        if "---" in text:
            body = text.split("---")[0].strip()
            if len(body) > 80 and "PROMPT_CHARS" not in body.splitlines()[0]:
                return body
        if len(text.strip()) > 80 and p.endswith(".txt"):
            return text.strip()
    return ""


def table_html(spec):
    if not spec:
        return ""
    head = "".join(f"<th>{html.escape(c)}</th>" for c in spec["cols"])
    body = []
    for row in spec["rows"]:
        cells = []
        for c in row:
            txt = str(c)
            if txt.startswith("**") and txt.endswith("**"):
                cells.append(f"<td class='hl'>{html.escape(txt.strip('*'))}</td>")
            else:
                cells.append(f"<td>{html.escape(txt).replace('**', '')}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<p class='cap'>{html.escape(spec['caption'])}</p>"
            f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


sections = []
total_imgs = 0
for b in sorted(BATCHES, key=lambda x: x.get('order', 99)):
    cards = []
    for rel in b["dirs"]:
        folder = os.path.join(BASE, rel.replace("/", os.sep))
        if not os.path.isdir(folder):
            continue
        files = []
        for pat in ("*output*.png", "*output*.jpg", "*.jpg", "*.png"):
            files += glob.glob(os.path.join(folder, pat))
        seen = set()
        files = [f for f in sorted(set(files), key=lambda p: os.path.getmtime(p))
                 if not (f in seen or seen.add(f))]
        for path in files:
            base = os.path.basename(path)
            if base.startswith("input_preview"):
                key = "original"
            else:
                key = key_of(base)
            uri = img_uri(path)
            if not uri:
                continue
            total_imgs += 1
            prompt = prompt_of(b["id"], key)
            details = ""
            if prompt:
                details = (f"<details><summary>提示词（{len(prompt)} 字符）</summary>"
                           f"<pre>{html.escape(prompt)}</pre></details>")
            cards.append(f"""
        <figure class="card">
          <img loading="lazy" src="{uri}" alt="{html.escape(label_of(key))}">
          <figcaption><b>{html.escape(label_of(key))}</b>
            <span class="fname">{html.escape(base)}</span>
            <span class="dir">{html.escape(rel)}</span>
            {details}
          </figcaption>
        </figure>""")
    if not cards:
        continue
    sections.append(f"""
  <section id="{b['id']}">
    <h2>{html.escape(b['title'])}</h2>
    <p class="desc">{html.escape(b['desc'])}</p>
    {table_html(TABLES.get(b['id']))}
    <details class="galwrap" open><summary>展开/收起这一批的 {len(cards)} 张图</summary>
    <div class="grid">{''.join(cards)}</div></details>
  </section>""")

toc = "\n".join(
    f'<li><a href="#{b["id"]}">{html.escape(b["title"])}</a></li>'
    for b in sorted(BATCHES, key=lambda x: x.get('order', 99)) if any(f'id="{b["id"]}"' in s for s in sections)
)

doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>gpt-image 画风生图 · 昨晚→凌晨 全批次对比</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 14px/1.65 system-ui,"Segoe UI",sans-serif; margin:0; padding:24px 30px 90px; background:#fafafa; color:#222; }}
 h1 {{ font-size:23px; margin:0 0 4px; }}
 h2 {{ font-size:17px; margin:34px 0 4px; padding-top:12px; border-top:1px solid #ddd; }}
 .desc {{ margin:2px 0 8px; color:#555; }}
 .cap {{ margin:6px 0 6px; color:#666; font-size:12.5px; }}
 .tw {{ overflow-x:auto; margin-bottom:10px; }}
 table {{ border-collapse:collapse; font-size:12.5px; }}
 th,td {{ border:1px solid #ddd; padding:3px 9px; text-align:left; white-space:nowrap; }}
 th {{ background:#f0f0f0; }}
 td.hl {{ background:#fff6d8; font-weight:600; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:12px; }}
 .card {{ margin:0; background:#fff; border:1px solid #e3e3e3; border-radius:8px; overflow:hidden; }}
 .card img {{ width:100%; display:block; }}
 .card figcaption {{ padding:7px 9px; font-size:12px; }}
 .fname,.dir {{ display:block; color:#888; font-size:10.5px; word-break:break-all; }}
 details {{ margin-top:6px; }}
 pre {{ white-space:pre-wrap; word-break:break-word; background:#f4f4f4; padding:7px; border-radius:6px; max-height:240px; overflow:auto; font-size:11px; }}
 ul.toc {{ columns:2; }}
 .kpi {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:10px 14px; margin:10px 0 0; }}
 .kpi b {{ color:#a05a00; }}
 @media (prefers-color-scheme: dark) {{
   body {{ background:#1b1b1b; color:#e9e9e9; }}
   .card,.kpi {{ background:#262626; border-color:#3a3a3a; }}
   th {{ background:#333; }} td.hl {{ background:#4a4326; }}
   pre {{ background:#333; }}
 }}
</style></head><body>
<h1>gpt-image 画风生图 · 昨晚 → 今天凌晨 全批次对比</h1>
<p class="desc">覆盖 data/20260920 与 data/20260921 两天的 {total_imgs} 张产物、8 个实验批次。表格是每个批次的权威指标，图片可按批次展开。</p>
<div class="kpi">
 <b>一句话结论</b>：① 官方 prompt 上限 32000 字符，11k~40k 都能出图，但<b>长文本会把参考图挤掉</b>（配色贴合 0.39→0.12）；
 ② <b>画风生图最优 = 600~2000 字符的短说明 + 参考图</b>，配合 v6 的「明度保护 + 线稿连续 + 发丝分组 + 细节预算」；
 ③ <b>线条连续性由参考图自身的边缘语言决定</b>，与分辨率无关（放大柔边图反而更差，纯文本反而更好）；
 ④ <b>Gemini 重绘默认固件已换成「保守修复 + 色彩匹配 + 线条连通」</b>（写死 cel 的旧固件会把水彩改写成平涂：say-hana 头部配色 0.22）；
 ⑤ 双参考（源图 + 改良锚图）清线最强但保真下降，适合线稿优先场景。
</div>
<h2 style="border:none;padding:0">目录</h2>
<ul class="toc">{toc}</ul>
{''.join(sections)}
<p class="desc" style="margin-top:40px">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
脚本 <code>tools/make_overnight_report.py</code> · 实验报告 <code>docs/gpt-image-tid-style/README.md</code></p>
</body></html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"written {OUT}  {os.path.getsize(OUT) / 1024 / 1024:.1f} MB  {total_imgs} imgs  {len(sections)} sections")
