# 共享双图 / 多参考画风指标（2026-10-06）

统一入口 `utils.style_similarity.compare_images(inputs, requested, progress, cache_directory)`。
`image_manifest(candidates, references)` 接收路径并冻结 SHA-256。
双图页使用画风提取同一个 `create_worker` 子进程及 `compute` 适配器，CLI `--similarity` 直接调用同一个计算入口。
计算只读本地图片和权重，不调用文本或生图 API，不修改图片。

## 固化的八组指标

| key | 计算方法 | 趋势 / 解释 |
|---|---|---|
| gram | VGG-19 五层 `G=FFᵀ/(CHW)`，各层 `sum((Ga-Gb)^2)/4`，层权重均为 1 | 非负距离，越小越接近；`gatys-layer-sum/v2` |
| adain | 深度特征各层通道均值 / 标准差向量 L2 距离之和 | 非负距离，越小越接近 |
| lpips | 官方 LPIPS-AlexNet v0.1 校准网络 | 感知距离，越小越接近；受主体 / 姿势影响 |
| csd | 官方 CSD ViT-L 风格描述子 L2 归一化后的余弦 | 越大越接近；不能叫人类校准百分比 |
| tone | P90−P10、局部 RMS、中灰带占比、高光裁切率 | 组内统计贴近度 0–1，越大越接近 |
| edges | σ=2/1/0 的 Canny 边缘占比，按三个尺度总量归一化 | 组内统计贴近度 0–1 |
| lines | 长线 / 碎线占比、每千骨架像素端点密度、平均连通段长度 | 组内统计贴近度 0–1；并非越长越好 |
| space | 最大低纹理亮区、总低纹理亮区、中间调主体占比 | 组内统计贴近度 0–1；构图混杂最大 |

深度模型预处理 / 输出层 / 权重来源在 `utils/style_metrics/config.py` 和共享 `metadata.py`。
本地原始统计从历史 `tests/style_render_metrics.py` 移到 `utils/style_render_metrics.py`；旧实验入口及旧 `closeness` 公式保留。
新比较明确版本 `style-similarity/2`、`symmetric-render/1`：EXIF 转正、RGB、长边统一 512px bicubic 等比缩放，使用固定形态学骨架算法。
各组的实际分量键由 `LOCAL_KEYS` 固定，等权平均 `1−abs(a−b)/(abs(a)+abs(b))`，双方为零取 1。
这是对称比较；旧的相对参考值公式不对称，零参考值也不能得到同图满分，所以不复用其最终旧总分。
JSON `metric_contract` 固化键、公式、尺寸及算法；逐对 `detail` 保留双方原始统计、分量贴近度、深度分层数值。

八组是计算指标，不是 `prompt_gpt` 的八个描述字段。视觉模型的 12 项评分、主体门禁仍独立保留。
当前只迭代提示词，没有扩散加噪目标、时间步与模型预测，不能计算真实 LoRA 训练 loss。
不把距离机械转成百分比，不合成没有标注校准依据的八项总分。

## 使用

UI：生图页下新增「画风相似度」，选择待比较图和参考图、设备，点击计算；支持取消和 JSON 导出。
窗口显示八组数值，详情含设备、精度、公式、hash、自检与错误。
多图画风提取：「自动深度指标」/ 手动指标按钮继续使用原流程；所有轮次及终审的每张候选图分别与每张提取源图计算。
汇总显示均值、中位数、最小 / 最大及有效配对数；任意配对缺失时该指标的均值 / 中位数为空，不以少量成功结果冒充全参考集。
每项独立同图自检：距离目标 0，CSD 目标 1；失败不纳入正常统计。
缓存按输入 hash、公式代码、依赖版本、权重 hash、设备与精度隔离；只复用完整缓存，计算期间输入变化拒绝写回。

```powershell
python tools/style_metrics_verify.py --similarity A.png B.png --device cpu --out cache/temp/similarity/result.json
# 一张候选对多张参考，逐张比较，不先平均图片：
python tools/style_metrics_verify.py --similarity A.png B.png --reference C.png --reference D.png --device auto-npu
# 对已有画风提取所有轮次 / 终审与全源图计算：
python tools/style_metrics_verify.py --comparison-state path/to/state.json --device auto-npu
python -m unittest discover -s tests -p test_style_similarity.py
```

双图 CLI 完整成功退出 0，partial 或失败退出 1，保留 JSON / HTML。
显式 NPU 不可用不伪称 CPU 为 NPU；`auto-npu` 优先探针可用的 NPU，再 CUDA，再 CPU。
NPU FP16 与 CPU/CUDA FP32 不混用数值。

## 验证

可靠性实验执行协议见 [DeepSeek 严格受控实验](DEEPSEEK-SIMILARITY-CONTROLLED-EXPERIMENT-20261006.md)。该协议尚未执行完毕，包含技术失败重试、原作品留出、干扰对照、定位误差和独立盲评，不把计算正确等同画风效度。

测试覆盖同图 / 对称 / 差异、损坏图片、hash 变化、缺失不计零、全部候选 × 全部参考、提取与独立计算数值一致、CLI 同入口、UI 冻结参数 / 子进程 / 八项展示。
真实双图验证素材及报告在 `cache/temp/style-similarity-validation/`，不写入生产日期目录。
本轮不重启正在运行的 app；新 CLI 立即使用更新，GUI 待空闲时正常重启加载。


## 面部与发丝扩展（schema 2）

新增统一 `utils/style_face_metrics.py`，在每个 candidate/reference 配对下保存 `face_features`，同一候选的全部参考汇总为 `face_summary`。
主八组的 `status` 与独立 `face_status` 分开：整图完成不冒充面部定位完成。

| 指标 | 固定测量分量 |
|---|---|
| 发丝细腻程度 | **标出的发丝**横向明度剖面的半对比度线宽中位数 / P10 / P90，以及线宽变异；均除以脸宽，至少三条可读发丝 |
| 仅发丝连贯性 | 沿发丝路径等间距采样的可见覆盖、最长连续段、断口占比、支持状态切换频率 |
| 眼睛亮度 | 左右眼开口内部平均灰度与 P90，除以255；不采眉毛或全头区域 |
| 上下眼睑间距 | 眼角局部坐标系下上 / 下眼睑 41 点重采样，平均与最大间距除以该眼宽 |
| 睫毛画法统计 | 标出睫毛的条数、相对眼宽的线宽 / 长度、线宽变化、相对眼弦的方向、可测时根部至尖端粗细比 |
| 眼睛左右宽度 | 每只眼内外眼角的弦长 / 脸宽，不比较原始像素宽度 |
| 两眼之间宽度 | 可见内眼角间距 / 脸宽，及间距 / 双眼平均宽度 |
| 眼睛弧度 | 上 / 下眼睑相对眼角弦线的拱高，以及41点曲线的归一化二阶差分幅度 |
| 虹膜占比 | 可见虹膜标注与眼开口相交后的面积 / 眼开口面积 |
| 眼部高光 | **虹膜内**灰度≥240的候选高光面积占比、连通区域数与最大块占比，排除眼白 |
| 上下眼线粗细 | 可见眼睑线横向明度剖面线宽 / 眼宽，分别保留左右上 / 下眼线 |
| 眼角倾斜 | 眼角弦线相对双眼中心轴的角度（消除整体头部旋转） |
| 眼鼻口比例 | 眼中心至可见鼻中心、鼻至可见口中心的垂直距离 / 脸宽 |

这些是局部统计的**贴近程度**，不是“越细越美”“睫毛越多越好”，也不能概括全部绘画语言。
眼睛神态、透视、虹膜固有色、表情和照明仍可能混杂。当前只按视角类别检查可比性，没有精确3D视角校正。
发丝不借用整图 Canny / 骨架连通分数：采样中心和整个横向带都必须在标出的头发内部区域；越界就不给数值。
固定采样步长约脸宽/384，半宽为测量单位的2.5%，21个横向采样点，可见对比度阈值12/255。
输入眼宽不足16px、脸宽不足64px、闭眼 / 遮挡、眼睑回折 / 交叉、坐标非法、有效分量不一致均不补出指标。
没有区域定位时保持 unavailable；没有发丝路径时不以衣服/脸/背景边缘代替。
各项对称分量贴近度等权平均，不合并整图与面部总分。

### 自动定位与人工校正

`utils/style_regions.py` 通过当前配置的看图文本模型生成候选，提示词独立在 `prompts/style-extraction/face-regions-v1.md`。
模型不能声称人工确认：保存时强制 `confirmed=false`。多人物无法唯一定位时需人工指定；不猜“最大脸”。
定位按图片 SHA-256 存在 `cache/style-regions/<sha>.json`；历史修改归档到其 `history/`。不向输入图所在的外部目录写文件。
候选缓存按图片、端点 hash、模型和提示词 hash，已经确认的定位优先保留；不重复收费请求。
区域文件带 `version=style-regions/1`、`image_sha256`、`pose`、`face_outline`、左右眼曲线/虹膜/睫毛、头发区域/发丝路径、可选鼻口点。

新双图页默认打开「自动面部定位（文本 API）」；可以关闭，仅计算已保存定位 / 整图指标。
「面部 / 发丝定位…」打开叠加曲线编辑器：自动候选、逐点画曲线、撤销点、清除此项、眼睛张合和视角、JSON精细编辑、人工确认。
定位请求期间锁定编辑，保存修改后原结果失效，需要重新计算。
画风提取的指标页有同样定位入口，可选择源图或某一轮候选；「自动面部定位」默认关，避免历史任务自动追加文本请求，手动开启可一次定位全部源图与候选。
计算子进程用临时环境变量接收UI当前端点配置，不在 argv / 区域文件中写 API key。

自动候选计算标记 `provisional`，可查看暂定数值，但正式均值与中位数只接受 `ok`；覆盖不全保持空值。
按两图视角类别检查可比性，未知/不一致的眼部几何不作正式比较。原有主体/内容复制门禁保持独立。

```powershell
# 自动提案（会调用当前文本分析端点）；候选不算人工确认：
python tools/style_metrics_verify.py --similarity A.png B.png --locate-regions --device cpu
# 导入已经检查的定位（JSON必须匹配本次输入图片hash）：
python tools/style_metrics_verify.py --similarity A.png B.png --regions A-regions.json --regions B-regions.json --device cpu
# 对全部提取图 / 每轮候选做候选定位与统一测量：
python tools/style_metrics_verify.py --comparison-state path/to/state.json --locate-regions --device auto-npu
```

### 本轮证据

真实CPU双图：Gram/AdaIN/LPIPS/CSD与本地四组全部 `ok`，报告 `cache/temp/style-similarity-validation/result-full.json`（八组首版报告，不含后加面部定位）。
面部可控图样验证：同图十三项均为1、换顺序对称、发丝断口降低连续支持、区域外黑线不影响发丝数值、暗化虹膜影响眼亮度而不改眼宽、未确认候选不计正式均值、闭眼/视角/越界/低分辨率拦截、自动提案不能声称人工确认、统一引擎配对与直接测量一致。
2026-10-06 已对两张参考图及两张真实 Gemini 产物运行自动定位并核查叠图：返回候选，但精度未通过，存在眼睑偏位及发丝区域混入脸/衣服。保持 provisional / unavailable，需人工校正确认；不能宣称模型能稳定定位每根发丝。八组全图指标的四个配对全部成功，实测记录见 [Gemini 带参考图比对](GEMINI-SIMILARITY-TEST-20261006.md)。
相关完整 pytest 138 passed / 5 skipped（慢速跨设备测试未开启）；最后的发丝采样带约束、编辑器缩放/叠加等补充复跑42项相关测试全部通过。没有安装依赖或重启 app。

区域几何快照额外嵌入 `region_geometry`；读取完整缓存与发布报告前再次核验区域hash，防止计算过程中被人工校正后写回旧定位结果。
