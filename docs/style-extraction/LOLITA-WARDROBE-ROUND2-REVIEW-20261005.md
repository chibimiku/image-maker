# Round2 复核与 Round3 依据

日期：2026-10-05。本次核对32张图、samples/scores/summary/ledger和runner，未在线调用，未改生产逻辑或原实验记录。

32个样本ID唯一，图片全部存在且可解码，prompt全文hash均一致。账本记录生图32、文字2、带图18、模型列表1。v3在GPT中避免了C03/C04强制变成Lolita；C06三张v3图比v2明显收紧，一张Gemini仍露出较多上身。C01/C02换装方向有可见回归结果。

## 评价与报告问题

1. **保护评价缺少条件。** 16条评分请求只传case/action/framing，没有content和protected。发色、瞳色、胸针与衣色的具体要求没有传给评分员，保护pass/fail/uncertain不能算完整身份锚验证。
2. **C04-gemini-r2失败依据超出输入。** 原内容只要求red school uniform with a gold star brooch，没有白衬衫或格纹裙，也没有源图。评分把对照B的白衬衫/格纹裙变成A必须保留的条件。目视A仍是红上衣、红百褶裙和星形胸针，不能据缺白衬衫/格纹直接判换装失败；精确原剪裁仍无法验证，不能因此改判全项pass。
3. **硬样本统计有误。** summary.scoreboard.v3_hard实际是9 pass、2 failed、1 unproven，报告写10通过、2失败。全16组12 pass、2 failed、2 unproven与JSON一致，但受前两项缺输入影响，不能照搬作采用门槛。
4. **就绪门禁未遵守。** review ready=false、prepare_ok=false仍继续生成。审查误把冻结控制的已知问题当候选阻塞，确实是协议缺陷；不等于执行者获准忽略失败。Round3须明确控制/候选身份，且候选失败必须拦所有--generate/--run入口。
5. **其他偏差。** A/B实际11/5，未实现约8/8平衡。解析使用gpt-5.6-luna而非指定DeepSeek端点，应明确模型归属。C06校准把B描述成dress/skirt也略过度，不能算细部精准识别证明。

方向支持省去不适用衣装块与强化裁剪，但当前评分不足以确认所有保护锚。不要再花调用追问原评分，下一轮先让评价收到完整真实条件。

## 人物扩展

前两轮主要是棕长发绿眼成年女性。Round3用六名原创成年女性：原人物桥接、短银发眼镜、深肤卷发、成熟黑发、蓝短发眼镜、紫色束发。分别测保留、只补全、Gothic改款和严格近景。年龄、肤色、发型、眼镜、体型不得随衣装变化。

人物组合是探索，不把效果归因于某单一属性；没有源图仍只验证文字锚，不称同脸身份或原剪裁精确保留。人物卡在prompts/wardrobe/round3/characters.json，计划在WARDROBE-ROUND3-CHARACTERS-AND-PLAN-20261005.md。尚未在线执行。
