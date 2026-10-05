# myc0t0xin 启用记录（2026-10-05）

用户确认后启用 B 组合，画风名称为 `myc0t0xin`。

| 配置字段 | 来源 / 设置 |
|---|---|
| `prompt` | 原运行 round5 的 `gemini_full_prompt`，逐字复制 |
| `prompt_gpt` | 原运行 round2 的 `gpt_image_prompt`，584 字符，八字段校验通过 |
| `repaint_clauses` | 原运行 round2 的 7 条文字条款，逐条复制 |
| `enabled` | `true` |
| `motif_enabled` / `motif_clauses` | `false` / 空列表 |
| `repaint_reference_mode` | `none`，重绘只使用 GPT 首图和文字条款 |
| `ref_image` | `submodules/image-maker-artstyle/references/myc0t0xin.jpg` |

参考图取自已验证的 `76033649_p0.jpg`，原字节复制到画风子模块，随配置一起版本化。首次生成仍可按界面参考模式使用这张图，重绘不重新发送画风图。

配置写入运行时 `conf/config-styles.json`，并同步本条目到子模块 `config-styles.json`。只修改 `myc0t0xin` 键，其他配置保持原值；子模块中其他未提交的画风修改保留。

画风独立仓库本地提交：`2c0af60`（`feat(styles): enable validated myc0t0xin B combination`），只包含本条配置和参考图。主仓库目前忽略 `submodules/`、未跟踪该目录的 Git 指针，因此通过本记录保存画风提交号，主仓库只提交启用记录与独立样本依据，不包含其他工作中的代码。

依据：[独立样本补测](myc0t0xin-independent-samples-20261005.md)。20 张唯一图片、240 对 NPU / FP16 深度比较全部完成。B 在两种主体下 Gemini 视觉均值更高，但只有每主体每路两次样本，深度指标未统一支持 B 更优；不能宣称普遍领先或参考内容复制已彻底解决。C 组合因复制参考内容被排除。

启用前确认：字段与源轮次一致、GPT 八字段合法、参考图存在且哈希一致、启用后的画风列表包含 `myc0t0xin`。重启 app 后从画风下拉选择 `myc0t0xin`。
