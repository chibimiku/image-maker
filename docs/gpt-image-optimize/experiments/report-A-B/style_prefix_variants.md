# 任务 B · 前置画风 prompt 变体（内容部分逐字不动）

内容部分固定为下面这一段，**一字不改**（与 v3 测试完全一致）：

```text
a girl with long pastel pink hair and large brown eyes sitting on a thick mossy tree branch above a calm lake on a sunny afternoon, wearing a pink and cream lace frilled sweet Lolita one-piece dress with a big ribbon headdress, dappled golden sunlight, blue sky, pastel colors, clean cel shading, crisp continuous line art, sharp fine detail, medium full body shot.
```

只替换**前置画风部分**，形成 5 个变体。所有变体只针对「线条连续性与细节清晰度」这一问题设计不同归因假设。

## B0 · 基线（无前置，= v3 原样）
```text
Anime illustration style, {CONTENT}
```

## B1 · 前置：明确"平涂动画赛璐璐 + 等宽线稿"
假设：画风指令没有钉死渲染方式，模型才自由发挥成半厚涂。
```text
Traditional Japanese anime cel animation illustration, flat cel shading, bold even-weight continuous outlines, limited pastel palette, {CONTENT}
```

## B2 · 前置：明确"矢量级线稿 + 限制细节预算"
假设：细节预算没有被限制，模型用渐变代替线条。
```text
Clean vector-like anime line art, flat colours, minimal shading, limited palette, simple background rendering, {CONTENT}
```

## B3 · 前置：交给画师语境（少用渲染术语）
假设：术语堆砌反而让模型去做半厚涂，用工作语境更稳。
```text
An anime illustration from a high-quality animation production, drawn by a professional animation key animator with a fine pen, {CONTENT}
```

## B4 · 前置：把"线条连续"写成结果指标
假设：改成对成品的可检查要求，比"风格名词"更有效。
```text
Anime illustration with flawless unbroken line work, no blurry areas anywhere in the frame, every detail sharply rendered, {CONTENT}
```

## B5 · 前置 + 后置双重钉死（对照组）
假设：只在尾部约束（v3 的做法）不够，需要前后夹击。
```text
Anime illustration with clean crisp cel-shaded line art, {CONTENT} Every outline must be continuous and closed, hair strands and lace drawn as separate countable shapes, flat base colours with one hard-edged shadow per material, crisp edge definition in the background as well, no soft gradients, no blurry areas, no colour bleeding.
```
