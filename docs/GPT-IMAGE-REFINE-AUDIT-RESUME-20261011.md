# Refine接口失败断点续跑与产物大图

任务 `gptimage2_output_150825_0_03ee51-150956-2f857c-final-ink.png` 对应 `data/20261011/gpt_image_repaint/refine-7226f353fcc3-refine.json`，v9 hybrid。诊断成功、针对性跳过，Pro背景图已返回；background-audit的文本API遇到TLS EOF，状态fallback，背景未获审计通过，衣装和末步整体尚未执行。最后对回退输入执行加墨，不能据final-ink文件名判定refine完成。

## UI与恢复行为

- 新增“从断点继续”：自动选失败步骤。候选已返回、审计接口失败且没有审计结论时，复用候选，只补跑原审计，通过后执行后续步骤。审计system/user/images及候选prompt/options从原记录复用。旧失败记录不改，建立带parent_record的新子链，不标记人工override。
- 保留“重画该步骤并继续”和“采用该候选并继续”，后者仍属于明确的人工决定。正常fail审计不能当作接口失败自动复审；复审fail/异常仍保留输入并停止，无自动重试。
- 修复显式加载同一记录时阶段列表不更新、正在执行的同一记录转fallback后选择停留在旧步骤的问题。载入图片保留用户选中的产物，明确绑定记录；继续前验证源图hash、输入和历史参数，预检错误不会创建worker或锁死UI。
- 断点按钮拆为两行，避免一行过宽。状态区域增高，审计接口失败及原因可见。
- 点击产物列表条目或右侧预览可打开大图窗口；包括适应窗口、100%、放大、缩小、可选中/复制的绝对路径。双击条目打开目录仍保留，阶段候选预览的大图使用候选实际路径。

重启App加载新代码后，在GPT出图页点击“载入refine记录/产物”，选择本任务final-ink图或refine JSON，再点击“从断点继续”。该任务会先复审已有背景图，避免重新收费生成背景。

CLI同一入口支持：

```powershell
python tools/gpt_image2_gen.py --refine-record data/20261011/gptimage2_output_150825_0_03ee51-150956-2f857c-final-ink.png --refine-resume-audit --dry-run
```

去掉dry-run为显式执行。仍不重跑GPT源图或诊断，也不自动重复本地色调/加墨。最终需要在新refine结果上追加本地工序时应明确操作。

## 验证和本任务续跑

六套350项离线回归通过；追加CLI预检/续跑/失败退出码验证后，两套31项通过。覆盖失败审计候选复用、原审计请求复用、pass后仅衣装/整体两次出图、复审fail/超时停止、原记录不改及UI点击大图/绝对路径/同记录状态切换。四个运行模块语法编译及diff检查通过，不等同于真实桌面交互验收。

本任务dry-run定位到background断点，后续为background审计→clothing→whole。首次线上命令在具体外传授权前被自动审批拒绝，未派发。随后用户明确回复“授权本次上传和付费续跑”，同意源图、背景候选和后续中间图向new.aigc2d.com外传及3图/5文本总上限。

线上续跑：原背景候选复审fail0.94，右侧蓝色玻璃窗被改成棕色木窗；从原输入重新Pro背景返回 `refine-8cd758e5e908-background_173446-6ba637.jpg`，内容审计pass0.92但头发返回uncertain/null。定位到模板第2段要求hybrid背景/衣装返回实际头发检查，第12段却要求“other stages”返回not-a-hair-decision，修正第12段例外；未放宽保真字段或接受uncertain。

增加对旧hybrid模板引起的“不进行头发检查”回答的显式复审：只适用于背景/衣装、其他全部保真与有效性字段已通过、hair_state=uncertain/null且证据明确为not-a-hair-decision的已返回候选，使用当前修正范围的system，仍复用原user/images。新子链记录scope_clarification，旧记录保持，若实际头发未通过仍停止。额外核心/状态/CLI153项离线通过。

复审新背景pass0.94/hair clear，继续Flash衣装，返回 `refine-e8ea00c9431c-clothing_173805-47c801.jpg`，审计fail0.93：虽服装清理有效且人物/物体/材料保留，但胸前新增交叉系带、裙摆新增密集卷纹与装饰带，违反设计范围。最新子链 `refine-e8ea00c9431c-refine.json` status=stopped/stopped_at=clothing，保留已通过背景，最后whole未发出。

用户额外授权1图1审计，累计上限4图6文本。从已通过背景重做衣装，显式重画会通过文件模板追加该步骤上次审计的unacceptable_changes，避免重复设计漂移；仍只上传可信输入，不上传被拒候选，不放宽审计。该行为及后续范围隔离回归通过，核心/状态/CLI共154项通过。新衣装 `refine-51870cef0ec9-clothing_174208-6f8d2a.jpg` 仍因新增胸前系带fail0.98，不能称重试提示已可靠解决漂移。

用户明确选择采用这次衣装候选，子链 `refine-205800f2ccea-refine.json` 记录manual_override，旧fail保持。末步Flash整体/头发/光照及透明布料、蕾丝材质响应完成出图 `refine-205800f2ccea-whole_174509-19b213.jpg`，最终审计180秒ReadTimeout，未得到结论；status=fallback，selected保留人工采用的衣装输入。UI可只补whole审计，无需重复图片。已请求额外1次文本授权，待用户决定。

用户再明确授权1次文本审计，累计上限4图7文本。只复用末步候选补审，未再生图；子链 `refine-b91302a165e8-refine.json` 最终pass0.93/hair clear，status=finished。背景为模型通过，衣装为用户人工采用（父链manual_override，原fail不改），末步为模型通过，不能称全链每一步自动pass。

在新通过的whole原始产物上只补原任务一次本地ink（默认target_sep=10），输出 `data/20261011/refine-205800f2ccea-whole_174509-19b213-175002-df7748-final-ink.png`。新manifest位于 `data/20261011/pipeline-steps/refine-03ee51-resumed-final/`，不在旧final-ink上叠加，不重跑其它工序。模型审计针对whole原始JPEG，本地PNG不是再次模型审计结果。

合计4图4返回、7文本6返回1超时（超时收费未知），全部预留已结算，无在途/自动重试；原源图/记录及旧产物保留，原记录hash核查一致。未重跑GPT首图/诊断、未重启App。代理全图观察末步头发更连续，衣装与背景基本沿用人工输入；不等于批量稳定性或用户成品验收。证据 `data/test-result/20261011/refine-03ee51-resume/verification.json`、budget及离屏中文大图/控制区截图。
