#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把大模型视觉打分结果追加进「出图对比报告」HTML（在 </body> 前插入一节）。"""
import html
import io
import os
import re


def _section(result):
    rows = ["<tr><th>候选</th><th class='num'>总分</th><th class='num'>线条笔触</th>"
            "<th class='num'>五官头发</th><th class='num'>上色光影</th><th class='num'>材质层次</th>"
            "<th class='num'>背景处理</th><th class='num'>分项加权</th><th>判定</th><th>依据 / 最大差距</th></tr>"]
    for item in result.get("assessments") or []:
        score = item.get("style_score")
        verdict = "通过" if score >= 80 else ("存疑" if score >= 60 else "不通过")
        color = {"通过": "good", "存疑": "warn", "不通过": "bad"}[verdict]
        rows.append(
            "<tr><td>{id}</td><td class='num'><b>{s:.0f}</b></td><td class='num'>{a:.0f}</td>"
            "<td class='num'>{b:.0f}</td><td class='num'>{c:.0f}</td><td class='num'>{d:.0f}</td>"
            "<td class='num'>{e:.0f}</td><td class='num'>{w}</td><td class='{cls}'>{v}</td>"
            "<td>{r}<br><span style='color:#6b6b73'>最大差距：{g}</span></td></tr>".format(
                id=html.escape(str(item.get("id"))), s=score, a=item.get("linework", 0),
                b=item.get("face_hair", 0), c=item.get("shading", 0), d=item.get("texture", 0),
                e=item.get("background", 0), w=item.get("facet_weighted", "-"), cls=color, v=verdict,
                r=html.escape(str(item.get("reason") or "")), g=html.escape(str(item.get("biggest_gap") or ""))))
    note = html.escape(str(result.get("overall_note") or ""))
    return ("<h2>六、大模型视觉打分（0-100）</h2>"
            f"<p class='legend'>参考集共性：{note}<br>"
            "由文字/视觉分析接口一次发多张图评出（画风为主、内容不计）；"
            "与上面的本地指标并列参考，不互相替代。</p>"
            f"<table>{''.join(rows)}</table>")


def patch_report(path, result):
    with io.open(path, encoding="utf-8") as handle:
        doc = handle.read()
    doc = re.sub(r"<h2>六、大模型视觉打分（0-100）</h2>.*?(?=</body>)", "", doc, flags=re.S)
    doc = doc.replace("</body>", _section(result) + "</body>")
    with io.open(path, "w", encoding="utf-8") as handle:
        handle.write(doc)
    return path
