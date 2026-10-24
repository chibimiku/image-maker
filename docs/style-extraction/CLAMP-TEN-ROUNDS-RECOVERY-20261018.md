# CLAMP 第8–10轮与终审恢复收尾

> **2026-10-19 溯源纠正与采用**：第10轮重绘TLS失败后返回GPT输入，不是成功重绘；此前12新图/4重绘成功说法撤回，实际11新图/3重绘成功，39评分记录含38张不同图。剔除回退项后，采用Gemini8＋GPT3＋真实重绘并列中的7，已写入clamp-sakura-chobits-original。原评分不篡改，详见本目录provenance-correction-20261019.json及docs/style-extraction/CLAMP-MOTIFS-AND-ADOPTION-20261019.md。

已完成新增3轮、每轮4次差异检查、10批局部细化、合并、全量终审和12张三路测试图。原会话因上下文压缩网络错误中断于第8轮启动前；本会话读取原授权和第7轮状态后继续。未重跑前7轮成功产物。

## 最后同批复评

每通道5张：旧选择＋第8、9、10轮＋本次终审。沿用12维、5组等权及主体门禁；每批同一10张原作加两张原作脸部裁剪。三批全部返回，15/15覆盖。为保证5个版本真正同批，仅在评审进程内将批量大小设为5并记入输入哈希，GUI默认仍为4。

| 通道 | 模型排序候选 | 限制 |
|---|---|---|
| Gemini直出 | 第8轮 | 脸型和眼睛均仅5/10；代理看图仍认为鼻唇偏写实，旧第3轮保留 |
| GPT首图 | 保留第3轮 | 旧版与4个新版五版完全同分，不能称唯一最佳或质量完全相同 |
| GPT→Gemini重绘 | 第10轮 | 比第7轮/终审画风分仅高0.05，保留近似备选，无稳定胜出证明 |

总体仍偏现代动漫水彩，未形成稳定的CLAMP眼型、墨线节奏和脸部抽象优势。模型排序不是用户验收；所有选择仅为候选，没有写入正式画风预设。每个包保留其实际测试的同轮GPT底图与条款，未混装不同轮次。

## 常规评分与模型参数

全历史常规评分39/39：原18张＋第6/7轮及终审9张＋本次12张。原评分和本次特征聚焦复评分分开保存；不能把两种评审指引下的小分差当作画质涨跌，更不能作为相似度百分比。未跑深度指标，未训练模型权重。

文本模型gpt-5.6-luna；直出gemini-3-pro-image-preview/2K，GPT gpt-image-2.5-sunburst/max/1024×1536，重绘gemini-3.1-flash-image/2K。沿用固定成年原创角色、原参考附件、2:3与原重绘参考模式none。10图原始样本、旧状态、固定正文、指导文件、公共配置及重绘配置哈希全部一致。

## 中断、请求与费用

初次沙箱连接失败未返回新图，独立目录保留；自动审批因当前消息外传授权不够具体而拒绝，用户在本会话明确授权同一10张原作、旧/新产物与必要局部上传至https://new.aigc2d.com并承担费用后执行。

第9轮GPT首图首次代理连接被断开，按现有后台配置重试一次后返回；失败尝试费用未知。12个图片工序均成功，4份GPT响应及4份重绘账本可查，没有手动重复已成功图片。没有固定每张价格或已校准max费用声明。

最后三个同批评分请求最初被供应商403 insufficient_quota拒绝，用户确认充值并要求继续后，只补这3次评审，均返回，未新增生图。各通道quota-rejection-before-recharge.json保留原拒绝证据。当前无未完成请求。

## 修复与检查

- 英文主体禁词改为整词匹配，避免manga误命中man、purposeful误命中pose；中文规则保留，实际主体词仍拦截。
- 汇总可读取父轮完整排名，解决连续续跑只读取最近观察目录导致早期轮次漏评的问题；旧final正确映射至baseline阶段。
- 40项针对性回归通过；3个Python文件编译及相关模块导入通过；diff-check通过（ignore-submodules）。初次默认沙箱临时目录的3个os.replace权限失败，通过工作区basetemp重跑解决；未更改业务原子写实现。测试退出有pyreadline析构警告，成功测试进程退出码0。
- 本次全部12张解码及唯一哈希检查通过，新增局部批次中一条0.78置信警告保留。没有全套测试或真人桌面验收结论。
- 未修改运行时画风配置、提交或重启App。词汇校验代码在运行中的GUI里需重启才加载；本次CLI已用新代码。

## 文件

- delivery.html：15张旧/新完整对照、两套评分、实际提示词包与限制。
- all-stage-comparison.html / all-stage-ranking.json：全历史39张常规评分。
- final-channel-selections.json：最终候选、GPT五版并列与完整测试包。
- selected/：每通道导出的候选提示词和旧选择备份；new-candidates/：本次四阶段完整包。
- new-12-contact.jpg / face-comparison.jpg / selected-comparison.jpg：新图、旧新头部上半身、候选三路联系表。裁剪只作观察，非等头尺寸科学指标。
- observations/：每阶段常规评分；cross-channel-comparison/：同批评分、原始响应、输入哈希和余额拒绝历史。
- completion-verification.json：12图尺寸/哈希、39/39常规、15/15复评、输入冻结与失败说明。
- recovery-provenance.json / local-visual-observations.json：断点来源、授权恢复与逐轮本地观察。
- clamp-sakura-chobits-original_style_iter_result.json：可在App导入的完整10轮状态；不自动采用正式预设。

交付目录：`data\20261018\style-extraction\clamp-sakura-chobits-original\run-rounds-8-10-20261018-r1`。
