---
name: style-dataset-curator
description: 从用户给定的插画目录中用视觉审查挑选画风提取训练集，产出 Image Maker 可校验导入的筛图 JSON 清单。适用于百张作品选取代表样本、去重复和避免角色内容偏置；不自动训练、生成图片或写入画风配置。
---

# 画风训练图筛选

将用户给定目录中的图片审查后选成一个清晰、同画法而内容有变化的训练集。交付 UTF-8 `selection-manifest.json`，格式固定为 `image-maker.style-dataset.v1`，供 Image Maker「多图画风提取 → 导入 Agent 筛图清单」读取。必须先读 [references/manifest-contract.md](references/manifest-contract.md)，可从 [assets/selection-manifest.template.json](assets/selection-manifest.template.json) 理解字段。不同识图 agent 都使用同一契约，不要求特定模型。

## 输入与准备

需要源目录、英文画风名、目标张数；未指定张数时选 12 张。默认只扫描该目录第一层；只有用户要求包含子目录时才开启 recursive。输出放在用户指定目录，或当前工作区独立输出目录，保留源图。

使用随 skill 附带的脚本生成**真实元数据**和分批联系表，不让视觉模型猜文件路径、尺寸或哈希：

```powershell
& 'C:\Program Files\Python310\python.exe' '<skill目录>\scripts\curate.py' inventory '<源目录>' --style-name '<英文名>' --count 12 --output '<输出目录>\selection-draft.json' --sheets '<输出目录>\contact-sheets'
```

其他环境使用已安装 Python 3.10+ 和 Pillow。脚本无需 Image Maker 项目、联网或模型 API。不可用时通过环境提供的文件与图片工具完成相同枚举、哈希和审查，不能编造这些字段。

## 视觉审查与选集

1. 查看所有联系表，再打开入选候选及有疑义图片的原图检查脸、发束、线条、衣料和背景。联系表仅适合粗筛，不能凭文件名、时间或低分辨率小图决定最终选集。遇到模型无法识图或材料不全时保留 needs_review，不声称完成。
2. 每张可读图片记录 `inspected=true`、四项 0–10 分、具体理由和描述性 tags。分项为成稿完整度、与目标画风的画法一致性、细节可读性、压缩/遮挡/水印等对观察的干扰程度（artifact_cleanliness 越高越干净）。锚点：0 不可用、2 很弱、4 有明显不足、6 可用但有限、8 良好、10 优秀。分数是视觉判断，不是相似度百分比。
3. 排除草稿、重复裁切、设定拼图、严重模糊、遮挡画法细节的版本；人物数量、画幅、深浅配色本身不是排除依据。用户明确要研究的 Q 版、草稿或特定时期画法优先于默认规则。目录包含不同阶段/媒介/画法时，围绕目标分组，不把风格不同的作品硬混为同一画风。
4. 对完全重复和视觉近重复只保留一个代表，排除项用 `duplicate_of` 指向代表的相对路径。相同角色但明显不同作品可以保留；不同角色但同幅图裁切不能算多样性。
5. 在高一致性、高可读性候选中覆盖可用的冷暖配色、明暗背景、近景/半身/全身、构图和角色变化，避免重复角色/服装/场景被误认为必需画风。不要为了凑类别引入异画法。固定辅助分 = 成稿×30% + 一致性×30% + 细节×25% + 洁净度×15%；脚本和 App 计算，agent 不提供总分。选集还考虑去重和覆盖，因此不是机械取最高分。
6. 填写 `selected_order`（决定 App 导入顺序），选一张细节清晰且具有代表性的入选图作为 `reference_image`，并说明身份/服装等强内容泄露风险。无法找到完全中性的参考图时据实说明，不把风险伪装成已消除。

## 交付与校验

保持生成的 path、sha256、width、height、readable 不变，补齐 agent 信息、逐图决定、理由、分数和标签，以及选择总结、参考图理由。所有可读图必须实际查看；不可读图 rejected、inspected=false、scores=null，并说明损坏。不能确认的可读图 needs_review，说明原因；这样的草稿不可导入训练。

目标张数不是强制凑数。优质样本不足时选至少两张，填写 shortfall_reason；不足两张或有未完成审查项时，交付草稿并明确不能训练。

```powershell
& 'C:\Program Files\Python310\python.exe' '<skill目录>\scripts\curate.py' validate '<输出目录>\selection-manifest.json'
```

跨机器移动素材可加 `--source-root '<本机源目录>'`；相对路径与图片内容须不变。校验失败应修正清单或补审，不关闭哈希、目录覆盖和查看状态检查。最终报告给出选集总结、推荐参考图、数量不足或风格混杂的限制、JSON 路径和校验结果。不要仅给 Markdown 名单或旧式 `selected_images` 数组。

到筛图清单交付为止。用户在 App 预览理由并确认导入后，程序复制入选图到独立目录。此 skill 不自行调用付费生图/训练 API，不自动修改画风配置。
