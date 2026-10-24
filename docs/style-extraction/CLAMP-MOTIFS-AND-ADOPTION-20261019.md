# CLAMP 画风采用与装饰母题研究（2026-10-19）

## 已采用配置

新增启用 `clamp-sakura-chobits-original`。运行时 `conf/config-styles.json` 与画风子模块各新增一条；其他38条保持逐字段相同。参考图复制到 `submodules/image-maker-artstyle/references/clamp-sakura-chobits-original.webp`，与原Chobits参考逐字节同hash。

- Gemini完整说明/priority专用说明：第8轮。
- GPT八字段：第3轮（五版同分保留旧版）。
- 重绘条款：第7轮（在真实返回的重绘候选中，与第8轮/终审并列最高）。
- 重绘参考模式none，额外五官修订关闭，未修改默认模型或公共配置。
- 用户授权按通道组合；该跨轮整链尚未付费实测。现有运行时会让重绘条款同时影响GPT首图，因此不能称组合后的GPT首图与第3轮实际请求完全相同。

## 对上一轮结论的纠正

第10轮重绘API实际TLS失败，账本new_files为空；流水线回退到GPT输入，误记为重绘成功并参加评分。因此它的重绘最高分不是有效重绘证据，已剔除。真实新增11张图；12工序记录中1项为源图回退，全历史39个候选记录包含38张不同图片，成功重绘3次而非4次。原模型响应/原评分留档，不把模型门禁伪改为fail；单独记录产物溯源排除。详见实验目录 `provenance-correction-20261019.json`。

## 决定

**没有足够证据把一种固定邮票边框、固定法阵或固定花纹设为小樱＋Chobits的通用标记。** 现有样本中重复的书封框首先是同版本的出版包装，不能仅凭它反复出现就当作原作画法。也未找到可核验的作者说明，支持给两部作品一律加同一种边框。

**证据最充分的可选母题是细线植物花卉＋疏朗花瓣。** 在用户需要装饰性插画、且场景允许时，可用少量花朵、细叶枝和散瓣形成不封闭的侧边弧线，保留大片空隙。形状/位置是本项目的转化方案，不是作者访谈规定的固定公式；它只能增加装饰语汇，尚无实测证明会提高角色还原度。

已存为此风格的 `motif_clauses`，**默认 `motif_enabled=false`**，可在App画风母题选项启用。不开启时不追加此条款。条款限制装饰位于画面边缘、从属于人物，不改衣装和场景，不添加文字、卡牌、法阵或角色配件；也不照搬参考图的出版边框。

次要候选：蕾丝/缎带/多层褶边可作为正文允许的衣装设计方向，但不得自动给普通服装增加饰物。星月、卡牌、魔法阵、羽翼可加强某部作品联想，却容易变成小樱或其它作品的内容识别，不作为本混合风格的默认补丁。未据此添加新收费工序或新提示词改写请求。

## 可核验资料及其范围

1. [讲谈社《复刻版小樱插画集3》](https://www.kodansha.co.jp/comic/products/0000386933) 与 [CLAMP官方复刻公告](https://clamp-net.com/archives/58878)。出版社确认复刻包含当年插图解说、访谈等；本轮只核查公开书目信息，没有读取未公开全书。
2. [《Card Captor Sakura Memorial Book》2001访谈英译](https://chibiyuuto.com/clamp-interviews/clamp-interview-card-captor-sakura-memorial-book-february2001/)。说明小樱的细线、曲线和轻盈画面方向；猫井谈查花卉辞典、尽量避免一章内重复同一种花。可支持“花卉作为有意选择的表现语汇”，不能推出固定花框、固定花种或以花代替人物画法。
3. [《CLAMP No Eshigoto North Side》2002访谈英译](https://chibiyuuto.com/clamp-interviews/clamp-interview-clamp-no-eshigoto-north-side-august2002/)。谈到小樱偏彩色墨水，Chobits更多使用gouache；同时谈整体书籍设计。不应把两者压成单一水彩纹理，也不能将某版书封外框自动认作通用装饰。
4. [《CLAMP No Kiseki》7卷Chobits访谈英译](https://chibiyuuto.com/clamp-interviews/clamp-interview-clamp-no-kiseki-vol-7-chobits-march2005/)。谈圆珠笔模拟带铅笔感的线条，并提到喜欢绘制褶边衣装。支持作品间技法与衣装兴趣的差异，不支持给所有角色统一添加蕾丝。
5. [国立新美术馆CLAMP展](https://www.nact.jp/exhibition_special/2024/clamp/) 与 [官方COLOR SHIRO画集](https://www.clamp-ex-goods.jp/1455/)。官方展览将画材与表现技法的变化作为观察方向；公开画集说明是书目/展示范围证据，不代表本轮已阅读全文。
6. [讲谈社《Your eyes only ちぃフォトグラフィクス》书目](https://www.kodansha.co.jp/comic/products/0000011354)。搜索可核查书目存在，正文抓取403；未据此声称读过内页。

第2–4项是标明原出版物及译者的粉丝译文，非CLAMP官方英文发布；未取得日文原稿，中文结论为保守转述。没有把搜索不到的“CLAMP受穆夏影响”当事实，也没有以二手书评替代访谈。

另复看本项目已下载的原作封面联系表：Chobits4有植物花朵，1/2偏空气和水的流线；小樱有多样花卉、星月等。此为本地有限样本视觉观察，不是统计共性或全书结论。

## 验证与边界

用生产取用函数核对Gemini/GPT字段、重绘条款、参考图可读性、母题关闭/开启行为，以及两份配置仅新增此条目。仅本地验证，0新增收费请求；未自动重启App。相关证据在 `data/test-result/20261019/clamp-style-adoption/`。

画风子模块本地提交：`e7729df`（仅配置新增条目与原参考图）；主仓库其他工作未提交，未推送远程。

## 后续收费对照已完成

2026-10-19按用户授权增加6张图片与3批视觉评审，三路混合配置在固定主体上实际出图；当前花卉母题未形成明确有效边饰或稳定辨识度提升，默认继续关闭。见[完整A/B实验](CLAMP-MOTIF-COMPARISON-20261019.md)。这不等于整个生产GUI/分析/审计链已验证。
