# 独立 Topic：动漫人物表情

2026-10-17。用户指出“眼睛漂亮，但表情呆滞”，要求独立学习、不干扰头发专项。此目录、要素 JSON 和 Krita 练习画布单独保存；表情试验未应用到头发主稿。

## 已阅读的资料

- [Magda Proski：How to Draw Expressive Faces，Art Rocket](https://www.clipstudio.net/how-to-draw/archives/157239)。已读公开正文：表情依赖眉、眼周、脸颊与嘴的协同变化；眉头与眉尾可以分别移动；张嘴时还应考虑下颌。动漫简化应优先保留这些形状关系。文中某些情绪与特征的绝对对应，不作为普遍定律。
- [Raquel Arellano：How to draw a mouth and lips，Art Rocket](https://www.clipstudio.net/how-to-draw/archives/161618)。已读公开正文：嘴有体积和深度；嘴角与上下唇随表情改变，转头时保持它们贴合面部方向。采用结构思路，不把文中嘴大小、性别、性格之间的举例当固定规则。
- [Liz Staley：Beginner’s Guide To Expressions，CLIP STUDIO TIPS](https://tips.clip-studio.com/en-us/articles/4362)。仅读到搜索摘要中的同眼嘴、不同眉形对照；正文两次超时，列为待读，未称完整学完。

本轮阅读以公开文字为主，未声称看过视频或逐张看过网页图片。以下是本项目自己的练习和主观判断，不是作者对本作品的评价。

## 当前角色的呆滞感

识图判断：虹膜细节丰富，但眉、下眼睑和嘴角均接近中性；直视镜头和接近对称的五官强化了静止感。增加高光或腮红未必能改变情绪。中性可以成立，关键是它是否符合画面情境。

## 控制变量练习

先固定头部包络、眼距、鼻位、发型与虹膜配色，再改变表情。六格保持相同轻微转头和 -5° 倾斜，目前是浅层三分之四示意，不称完整三维透视。

| 练习 | 改动 | 自检 |
|---|---|---|
| 中性 | 眉眼嘴放松 | 作为对照 |
| 只改嘴角 | 其它五官不变 | 单换笑嘴的作用 |
| 联动微笑 | 嘴角、下眼睑、眉与脸颊 | 是否形成一致神态 |
| 惊讶 | 眉上移、眼裂张开、小幅张嘴 | 是否同向变化 |
| 担忧／难过 | 眉头上提、嘴角下落 | 眉头方向是否画反 |
| 专注 | 眉头稍压、眼裂略收、嘴收紧 | 是否容易读成不悦 |

数值只属于此练习，不是表情标准比例。先准备 [要素 JSON](../../../data/test-result/20261017/krita-expression-topic/elements.json)，然后画构造、五官，再识图修正。

## 实际自检与修正

1. 草稿：六格中心线与五官位置一致，适合控制变量；头型偏长，后续需练更接近正式角色的脸型。
2. 首遍：中性与单改嘴角可辨；联动微笑把下眼睑收得太多，出现困倦／眯眼感，填色笑嘴也偏厚。
3. 修改：放松下眼睑，把厚笑嘴改为轻的闭嘴弧线；两种微笑使用同嘴形，便于比较眉眼和脸颊的影响。
4. 复看：微笑更柔和；惊讶、难过容易区分；“专注”仍像不悦。轻微笑与中性的差别仍较小，尚不能称解决原人物呆滞感。

下一轮先练“温柔微笑／困倦”“专注／不悦”等接近的神态，再练正面、三分之四、低头。检查时遮住标签再判断，不能依据自己写的文字宣布成功。通过独立练习后，再决定是否迁移到人物。

## 文件

- [修正版 PNG](../../../data/test-result/20261017/krita-expression-topic/painting/expression-reviewed.png) / [分层 KRA](../../../data/test-result/20261017/krita-expression-topic/painting/expression-reviewed.kra)
- [首遍](../../../data/test-result/20261017/krita-expression-topic/painting/expression-comparison.png) / [定位草稿](../../../data/test-result/20261017/krita-expression-topic/painting/expression-construction.png)
- [练习脚本](../tools/author_expression_topic.py)

图形由程序规划，在 Krita 中分层绘制与检查，不是数位笔自由手绘；此六格没有替代正式头像宝石眼的精细绘制。
