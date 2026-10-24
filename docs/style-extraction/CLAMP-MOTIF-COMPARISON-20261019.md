# CLAMP 可选花卉母题：收费A/B对照（2026-10-19）

完成6图＋3批视觉评审，6/6图为不同哈希，6/6候选评审覆盖，无未完成请求。默认母题继续关闭。

## 结果（同通道画风量表分0–10）

| 通道 | 关闭 | 开启 |
|---|---:|---:|
| Gemini直出 | 5.983 | 5.933 |
| GPT首图 | 5.133 | 5.133 |
| GPT→Gemini重绘 | 5.600 | 5.667 |

模型全部主体门禁通过，但不是完美人体、文字或设计保真证明。开启组没有出现明确的边缘花卉弧线；Gemini主要仍是指定盆栽，GPT增加的藤蔓、植物画和花盆属于场景内装饰。两组脸部仍偏现代动漫水彩。开启重绘更平面清楚，但两组GPT输入不同，无法把它归因母题；单独采样也会改变眼睛、光线、姿势、构图。

本结果只说明“当前可省略母题块＋当前书店主体”没有建立有效性，不否定所有花卉构图。未来若另测明确的装饰海报版式，属于新的设计干预，不能把它与本次仅开关母题的结果混为一谈。

## 控制条件

采用生产配置Gemini8＋GPT3＋重绘7组合，文本正文与附件一致，2:3。Gemini gemini-3-pro-image-preview/2K，GPT gpt-image-2.5-sunburst/max，重绘gemini-3.1-flash-image/2K/none参考模式。未改模型、画风或默认母题。

A/B Gemini提示词只有母题块增减；GPT实际prompt也只有该块增减，非prompt的说明元数据motif_prompt相应变化，其余请求元数据一致。正文两组均允许由母题指令提供边缘装饰；保留“不要樱花、花纱、封闭出版边框”等原约束，实验不要求这些内容。重绘使用相同7轮条款，不直接注入母题，只处理各自首图。

15个冻结输入的最终hash一致；每条件每通道1个独立样本，无paired seed、无重复稳定性/因果证明。原10图＋原作脸部裁剪2图作为同批参考，各通道2候选1批，沿用12维5组等权和主体门禁；另记录装饰可见性、来源联想、核心脸法与内容干扰。分数不是相似度百分比，不跨通道作绝对比较。

## 运行与失败防护

复用tools/style_extract.py --controlled-tests --spec prompts/style-tests/clamp-motif-comparison-20261019.json。--dry-run只核对不收费。真实模式每个阶段发送前原子落盘，已成功的图只复用；失败/中断阶段不自动再次发送。重绘用strict=True，解码后再检查源图hash，防止失败回退GPT原图却记成成功。

同样为原GUI/迭代测试_generate_test_image启用strict=True，今后重绘流水线失败记录为空失败通道，不再以回退源图参加重绘评选。实际本次两次重绘均有新产物账本且与输入不同，未触发回退。

42项针对性测试通过（含新增成功复用/失败不重发/拒绝源图回退）；扩展首次35过3错，其中2项为unittest沙箱TEMP原子改名权限问题，改用工作区TEMP后37过/1排除；排除项为已有GUI恢复续训timeout_seconds=120但断言>=600，不在本次修理。3个Python文件编译和diff-check通过，不称全量或真人桌面验收。部分测试退出有pyreadline析构警告，成功测试退出码0。

## 交付

- delivery.html：自包含6图、局部、分数与完整请求文字。
- comparison-six.jpg / face-comparison.jpg：左关闭右开启，按三通道排列。
- summary.json：所有分数、评审观察与结论；local-visual-review.json为独立本地看图记录。
- comparisons/：三组逐维评审、原始响应、输入hash及补充母题观察。
- controlled-plan.json / controlled-state.json / controlled-run.log：实际计划、冻结输入、逐工序输出和断点。
- motif_off/ 与 motif_on/：原图、GPT原响应、重绘账本和工序记录。
- completion-verification.json：6图解码/唯一hash、冻结输入一致、2次真实重绘与3评审批次。

本次新增收费图片6张及视觉评审3批；GPT保留原后台暂时性重试配置，未手动重复成功工序。不推断未校准max固定每图成本。未修改生产默认/公共配置/提交或重启App。

交付目录：`data\test-result\20261019\clamp-motif-comparison`。
