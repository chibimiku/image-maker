# 坐姿完整插画第四轮：逐工序停下检查

本轮沿用已学习的构造、五官、线稿和配色知识，重新设计坐姿与几何，从 Krita 空白文档开始绘制《A quiet page》。没有导入旧作品或参考图片像素。画面为蓝发少女坐在椅上，双手扶住腿上的书。

## 本轮采用的检查方式

用户强调工序并要求每步自查后，实际改为七段执行。第一段创建空白画布，之后六段通过 `krita_studio.py append` 继续**同一文档**；每段保存 KRA 和 PNG，工具返回后先查看图片再进入下一段。发现问题则用 `revise` 修改原有图层、导出复看。

`append` 检查当前文档的准确路径、尺寸及新增层名冲突，避免把后续工序画进其它打开的文档。`svg_erase_masks` 用实际透明度扣除前景遮挡，让后补的发丝保留正确前后关系。

| 工序 | 自查重点 | 结果及修改 |
|---|---|---|
| 1 坐标与草稿 | 头部朝向，肩腰、髋膝踝、坐面关系 | 初稿缺关节及承重点，补上关节圆、腿部骨架和坐面接触线，保存 `02a-joints-reviewed.png` |
| 2 独立线稿 | 轮廓、闭合、遮挡、粗细 | 初版只是全轮廓叠加，出现穿透线；改为逐物体遮挡后的线稿，删除粗白色高光的错误描边；复看 `phase-2.png` |
| 3 大色块 | 暖皮肤、冷蓝头发、深椅子的分离 | 独立检查 `phase-3.png`；此时不加最终发丝。白色遮挡用的线稿不能盖住底色，所以检查底色时隐藏该层 |
| 4 体积与衣褶 | 颈部投影、身体侧面、坐姿衣褶走向 | 加身体两侧与腿上布料投影；复看 `phase-4.png`。衣褶仍偏图案化，不能据此称真实布料模拟 |
| 5 五官 | 同一头部朝向、杏眼、小瞳孔、短鼻、小嘴 | 眉下重复线过重、双唇突出；减淡重复线并简化嘴部；修改后导出 `phase-5-face-reviewed.png` 复看 |
| 6 头发与装饰 | 发根、发束、细丝及遮挡 | 初次统一把后发丝放在顶层，导致穿脸和穿衣；为后发/前发分别扣除前景遮挡，重新复看 `phase-6.png` |
| 7 氛围与整图 | 缩略、灰度、镜像、手部放大 | 补柔光与前景；手部放大发现手腕被书边隔断，分两次修正连接后重看 `hands-detail.png` |

每步检查不是要求无条件改一遍；发现具体错误才修改。还没解决的问题要记录，不能用“检查过”代替“画得好”。

## 文件

- [画面元素 JSON](../../data/test-result/20261016/krita-seated-practice-v4/elements.json)
- [最终分层 KRA](../../data/test-result/20261016/krita-seated-practice-v4/checked-stages/azure-quiet-page.kra)
- [最终 PNG](../../data/test-result/20261016/krita-seated-practice-v4/checked-stages/azure-quiet-page.png)
- [修正后的草稿](../../data/test-result/20261016/krita-seated-practice-v4/checked-stages/02a-joints-reviewed.png)
- [独立线稿](../../data/test-result/20261016/krita-seated-practice-v4/checked-stages/phase-2.png)
- [底色](../../data/test-result/20261016/krita-seated-practice-v4/checked-stages/phase-3.png)
- [验证记录](../../data/test-result/20261016/krita-seated-practice-v4/verification.json)
- [构造脚本](tools/author_seated_practice.py)

最终使用 `checked-stages/`。`painting/` 与 `workflow-final/` 保留为早期尝试，后者存在发丝穿透错误，不作为交付图。七份 `phase-*-result.json` 和修正记录保存实际执行情况；最后的通用 `render-record.json` 只对应最后一次修改，不能代替整段工序记录。

## 验证及实际限制

2400×3360，最终 KRA 共 31 个图层节点（包括初始白底节点），保留并隐藏草稿、坐标、独立线稿和底色检查层。27 张阶段/局部/检查 PNG 可解码；初始空白 KRA 合成图为纯白。脚本编译、范围内 diff-check 及实际新建/追加/修改 Krita 操作完成。

本轮有 13 组发束、1035 条程序构造的细丝路径，另有原生 Pencil-2 / Gpen 笔触。多数形体由曲线路径在 Krita 中栅格化，这些数量不是画质分数，也不能称为全部手工笔刷绘制。

头部使用浅曲面透视近似，椅子使用统一针孔相机投影。头部和椅子未作严格场景相机标定；全身解剖、手指关节、衣褶体积与头发自然度仍不足。手腕连接比初版清楚，但连接线及手掌形体仍显机械。最终是工序练习成稿，尚未达到参考图的自然程度。

本轮没有新增联网检索或付费生图请求，也没有安装资源、改模型配置、重启应用或提交 Git。
