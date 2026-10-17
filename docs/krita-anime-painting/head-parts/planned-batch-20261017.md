# 整体计划后的首批执行记录

用户授权按整体目标逐项执行，无需每项完成都等待确认。总体路线与检查标准见 [练习计划](../PRACTICE-PLAN.md)。本轮已执行结构修正、眼睛技术修复、正面细节整合，并保留过程与失败版本。

## 实际改动及自检

| 练习 | 实际变化 | 识图结果 |
|---|---|---|
| 顶部结构 | 改掉完全规则的圆团外轮廓，增加收起发量的起伏与底部接合轮廓 | 差异轻微，顶部仍偏圆团；不能称盘发结构已解决 |
| 左后发末端 | 一处长束增加不等长分叉，更新所有相关接收投影和遮挡轮廓 | 原尺寸看见小束开口，但整体仍有平滑长带感 |
| 虹膜边界 | 从定位脸中去除越过眼睑的椭圆，改为独立虹膜层及眼裂蒙版 | 首遍误用后发分区扣除，导致两眼空白；实际看图发现后改为仅用脸前刘海遮挡，复看已正常 |
| 正面虹膜 | 在当前方向重新绘制蓝灰虹膜、细放射纹、下缘小反光与克制高光 | 与正面眼裂协调；没有贴入原三分之四头像的像素，旧宝石眼稿仍保留 |
| 五官整合 | 尖细眼睑和睫毛、浅眉底与短眉丝、较柔鼻部和小嘴形、肤色渐变、轻微冷暖脸部平面 | 比占位脸协调；表情仍中性，耳部、头颈与上身未完成 |

眼睛的失败首遍保留为 `03-frontal-gem-eyes.png`，其虹膜被错误遮挡。修正为 `04-frontal-gem-eyes.png`；最终整合为 `planned-head-reviewed.png`。不能把首遍无虹膜的输出写成成功。

## 仍未通过的项目

头发外轮廓和内部束线仍过于光滑；顶部固定方式与右侧收拢区域缺乏明确空间结构；五官和头发的完成度仍不均衡；衣服仍为定位块面。这是头部整合练习，不是完整插画交付。

下一批继续根部、耳部、颈肩与衣料大形；不得用继续平均加丝代替造型。表情保留独立 Topic，当前中性版本不因“表情学习”而被随意覆盖。

## 保存与复核

- [当前整图 PNG](../../../data/test-result/20261017/krita-planned-head-batch/painting/planned-head-reviewed.png) / [分层 KRA](../../../data/test-result/20261017/krita-planned-head-batch/painting/planned-head-reviewed.kra)
- [头部原尺寸局部](../../../data/test-result/20261017/krita-planned-head-batch/painting/head-detail-reviewed.png)
- [眼睛局部](../../../data/test-result/20261017/krita-planned-head-batch/painting/eyes-detail-reviewed.png) / [左后发末端](../../../data/test-result/20261017/krita-planned-head-batch/painting/left-tip-detail-reviewed.png)
- [本批目标与限制](../../../data/test-result/20261017/krita-planned-head-batch/batch-plan.json)
- [完整可复算快照](../../../data/test-result/20261017/krita-planned-head-batch/final-scene-snapshot.json) / [复算与文件检查](../../../data/test-result/20261017/krita-planned-head-batch/verification.json)

2000×3000，12 张阶段与局部 PNG 完整解码，阶段 KRA 的 ZIP 结构可读；三个新增研究脚本编译通过。快照在 Krita 新文档实际重绘，与逐步修改版 RGB 像素一致。以上证明断点和文件可用，不证明艺术质量已通过。

全部为当前本机 Krita 的程序曲线和图层研究；未发生生成式图像请求、下载、安装或生产配置变更。没有后台定时任务。
