# 三组扩大 refine 实验完成

用户授权的单人金属、单人布料、双人三组实验已完成，比较人物→背景→配饰→光照的两步末段、同背景起点的一步修复＋调光，以及在两步最终图上再叠文字描述＋tinkle 画风参考。

交付 `data/test-result/20261008/refine-expanded-three/comparison.html`，共 21 主流程格，原图/脸部与衣装对照/实际 prompts/审计/失败记录均内嵌。

原会话中断时缺 B 光照、三组画风层及四项审计。本次补齐，初次本机代理权限失败三次后使用独立 r1 标签重试。全轮 25 图片尝试/22 返回（含一张比例错误 GPT），16 文本尝试/15 返回/1 历史超时。全部 25 图片预留槽与16文本槽结算，无未结算槽；JSONL 缺一个 reserved 行，由现存独占 slot 21 与 settled 行对账，未篡改原账本。

视觉观察：B 两步较少改胸前设计，合并版出现新增卷纹；A/C 无一致明显优胜。参考层对 B 的透明衣料和光泽局部有收益，伴蕾丝/折层重排；A/C 收益有限，未证明完整稳定画风迁移。末步调光审计 A/B pass、C fail，前序多个 fail 保留；独立实验继续不等于生产门禁通过，无最终画风审计或用户验收。

Image 1 编辑底图，Image 2 tinkle 原参考；实际后端按该顺序附图，与 [Gemini 官方多图示例](https://ai.google.dev/gemini-api/docs/image-generation#4-advanced-composition-combining-multiple-images)核对。模型未迁移。原图/prompts hash 保持；公共配置在中断期间改变，原计划保留并记恢复差异，不能声称整个历史配置相同。

完整证据与限制见实验目录 README、summary、completion-check 和 final-visual-review。未修改生产默认、未 commit、未重启。
