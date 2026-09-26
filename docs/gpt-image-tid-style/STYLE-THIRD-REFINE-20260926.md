# 第三次画风定向修订（2026-09-26）

本轮固定使用 `data/20260921/sucai/2026-07-24 12_17_40_1.jpg`，针对 sheya、say-hana-v4、inf-nikki-v1、iris-mix-style 和 satou_kuuki-style-v2 做定向修订。联网产物和请求记录全部保存在 `data/test-result/20260926/analysis-gpt-image/style-third-refine/`，横向图见 `style-third-refine-20260926.html`。

## 结论

- **sheya-style**：取消模型闭环后的全局通道增益，改为在提示词中把源图定义为固有色真值。蓝色只允许成为局部投影、轮廓光或几何环境光；保留第二轮已经认可的窄而警觉的眼型。补充禁止骰子、立方体、水面、蓝花、黑衣和短发等参考图内容。第一次新首图请求被审核拦截，重试成功；新 GPT 首图与完整画风重绘都没有整幅蓝滤镜，暖肤色、棕发、象牙色服装和木质背景得以保留。
- **say-hana-v4**：眼睛从低矮尖锐的细缝改为有明显纵向开口的中大圆润杏眼，使用宽上眼睑、短睫毛组、大而清晰的虹膜，并禁止长外眼线和狐狸眼。关闭额外五官修订，避免第一次完整重绘之后再次破坏周围线条；要求脸、发块、肩线和服装轮廓使用长而连贯的主线。
- **inf-nikki-v1**：场景、身体、服装、头发体积、材质和光照继续保持 UE5/PBR，面部单独改成二维动漫设计。明确禁止毛孔、真实眼睑褶皱、法令纹和真人演员感。新首图及 Gemini 重绘都保住了三维环境，同时面部不再接近真人。
- **iris-mix-style**：保留 1:2.5 至 1:3 的 Q 版比例，同时锁定坐在华丽椅子上的动作、腿部姿势、手靠近脸的手势，以及书房/图书馆中的书架、桌柜、地球仪、画框、灯和蜡烛。新首图和重绘均保留了椅子与室内叙事。
- **satou_kuuki-style-v2**：腿、丝袜和服装边缘增加液滴、细流和宽湿反光；配色要求同时出现大面积深蓝和深红/牛血红区域，以黑梅色和近黑色稳定暗部，紫色不再独占。加入藤蔓、漆裂、烟雾大理石纹和暗色花纹等“堕落”主题纹样。液体限定为雨水、魔法漆或凝露。该画风允许服装和主题配色变化，因此新增 `skip_identity_refine=true`：身份审计仍保存，但不让通用修订把深色设计恢复为象牙色原衣装。

## Sheya 新首图重试

- GPT 首图：`data/test-result/20260926/analysis-gpt-image/style-third-refine/sheya-style-fresh-retry/20260921-234549-2026-07-_output_114843_0_b8a609.png`
- Gemini 完整画风重绘：`data/test-result/20260926/analysis-gpt-image/style-third-refine/sheya-style-fresh-retry/20260921-234549-2026-07-_output_114843_0_b8a609-114853-b3c0bf-final-rp_114924-bca62e.jpg`
- 最终指标：亮度 129.3、饱和度 60.7、平均连通段长 79.1、端点密度 32.75、长线占比 0.781。

这次结果说明 Sheya 的蓝白问题主要应在首图和完整重绘提示中约束局部色彩关系，不适合在末尾使用整图 RGB 增益反向校色。后者会把所有表面一起推色，仍会呈现滤镜感。

## 实现

`skip_identity_refine` 已贯通 GUI 和无头 CLI：`build_first_pass_request` 把画风字段传入请求；GUI 与 `tools/analysis_gpt_run.py` 都会落盘审计，但在该字段为真时跳过最多两轮身份回改。同步工具也会把此字段写入版本化画风配置。

画风配置以 `submodules/image-maker-artstyle/config-styles.json` 为版本化真值，并同步到本机 `conf/config-styles.json`。
