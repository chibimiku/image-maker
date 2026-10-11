# 首图审计实用容差（2026-10-11）

用户确认 cccc11c1 首轮队列失败后手动重跑一次，共两轮各两张，并要求放宽检查、复审已有生成图。

生产首图审计保留重大内容错误、明显固有色变化、人体错误、参考内容复制和不兼容画法门禁。轻微表情开合、次要辫子/饰物、轮廓强弱、平滑程度及光点/纹理不足允许带警告通过。同属兼容二维动漫画法的差异不要求精确复制画师。

画风重大问题须声明 style_failure_kind=incompatible_medium 或 incompatible_rendering_family；其余画风问题降为 minor。fail 标签没有达到 0.65 门槛的重大证据时归 uncertain、不重画、不封锁。开嘴微笑/闭嘴微笑分级另有代码护栏。复制参考内容仍拦截，缺结论与接口超时沿用原行为。warnings/policy_version 可查，日志报告非拦截限制。

## 复审结果

复用两轮原候选审计的实际 content prompt，并按生产回调追加相同任务 metadata 的完整内容；候选→tid 参考图顺序，项目现有 gpt-5.6-luna/new.aigc2d.com 文本审计，180 秒，每张单次，不生成新图。

| 图片 | 最终结果 |
|---|---|
| cccc11c1_135817-first-e8795c-a1_135858-fdbebb.jpg | pass，辫子和画风限制记警告 |
| cccc11c1_135817-first-e8795c.jpg | pass，画风限制记警告 |
| cccc11c1_141125-first-3c15ea-a1_141201-fc18eb.jpg | fail，深棕/黑瞳变浅灰紫；嘴部开合经最终代码降为 minor |
| cccc11c1_141125-first-3c15ea.jpg | pass，用户指定的最终图 |

首版4次真实文本审计2pass/2fail，暴露模型将轮廓/平滑拆成重大问题。收紧严重画风边界后4次文本复审3pass/1fail；最后增加微笑开合代码护栏，只离线重新规范化本轮原始模型回复，结果仍3pass/1fail，不另发请求。45项离线测试通过（原37+新8），编译及相关diff-check通过。另一次通配符unittest发现pytest状态测试不能导入，当前Python无pytest，不声称那组状态测试或全套通过。

总计0次新生图，8次实际文本复审全部返回；前置4次本机WinError10013连接失败另存，合计12次文本尝试，均有预留/结算，无自动重试。每组预算4次，读原账本slot/event对账；send_budget.state现有实现不把文本settled槽位计入settlements映射，本次没有改写该账本或预算工具。最终4预留4成功结算。源图、参考与metadata收尾hash一致。

证据：data/test-result/20261011/first-image-audit-practical/；network-r1保留第一版，final包含最终完整请求、原始模型回复及final-decisions.json。没有改写历史审计/队列判定、图片或配置，没有提交或重启。Python判定层需重启app加载，新任务使用新版规则；历史任务不自动重跑。仅4图案例验证，不能证明跨任务稳定性。
