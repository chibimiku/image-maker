# 2026-10-06 画风版本迭代 → git 提交对照

| 提交 | 阶段 | 涉及画风的版本变化 | 对应快照 |
|---|---|---|---|
| `7644b15` | 迭代一：ccc canh 首轮修复 | ccccanh `prompt_gemini` 5250（=全文）→ 1889/v5；新增 `gemini_content_field=gpt_image_prompt_short`；`utils/styles.py` 修复 `gemini_content_field` 被丢弃 | `20261006-0140-iter1-before.json`（改前）、`20261006-0153-iter15-ccccanh-v5.json`（改后） |
| `c16bd3e` | 迭代二：批量加锚 + 5 画风重写 | 15 个画风加锚（ajicoma / cute-lingerie-wardrobe / dall-e-v2 / fuzichoco-v2 / inf-nikki-v1 / iris-mix-style / myc0t0xin / noir-art-style / puracotte-style-v2 / renian / satou_kuuki-style-v2 / say-hana-v4 / tid / tinkle-style / waterink-style）；millon-knots 5632→1975、puracotte-style 820→1755、goto-p 1031→1529、komori-hikki-style 1091→1725、noir-aiart 1181→1725 | `20261006-0237-iter2-ccccanh-plus-batch.json`（改后）、`20261006-1200-current.json`（现值，含 v7） |
| `ad57364` | 迭代三：评估工具链与报告 | 不改画风；新增共享相似度入口、面部指标、双图 UI、5 个报告页 | — |

## 回滚方法

```powershell
# 1) 从 git 取任一阶段快照
git show 7644b15:conf/style-history/20261006-0140-iter1-before.json > conf/config-styles.json

# 2) 或从本地备份复制（data/ 下的实验备份）
Copy-Item "data/test-result/ccccanh-fix-20261006-0155/config-styles.BEFORE.json" conf/config-styles.json

# 3) 同步子模块副本
python tools/sync_styles_to_submodule.py
```

改完 `conf/config-styles.json` **需要重启 app** 才在 GUI 生效。

## 本地备份完整清单

| 路径 | 阶段 |
|---|---|
| `conf/style-history/20261006-0140-iter1-before.json` | 迭代一之前（git 已提交） |
| `conf/style-history/20261006-0153-iter15-ccccanh-v5.json` | ccc canh 第一轮修复后（git 已提交） |
| `conf/style-history/20261006-0237-iter2-ccccanh-plus-batch.json` | 批量改动后（git 已提交） |
| `conf/style-history/20261006-1200-current.json` | 当前现值（git 已提交） |
| `data/test-result/ccccanh-fix-20261006-0155/config-styles.BEFORE.json` | 今天最早基线 |
| `data/test-result/ccccanh-fix-20261006-0155/config-styles.BEFORE-apply.json` | 应用前 |
| `data/test-result/ccccanh-fix-20261006-0155/config.BEFORE.json` | `conf/config.json` 备份 |
| `data/test-result/ccccanh-fix-20261006-0155/submodule-config-styles.BEFORE.json` | 子模块同步前 |
| `data/test-result/ccccanh-fix-20261006-0155/submodule-config-styles.BEFORE-sync.json` | 子模块同步前（第二份） |
| `data/test-result/style-lab/backups/config-styles.before-lab-choice.json` | v5 选定前 |
| `data/test-result/style-lab/backups/config-styles.before-batch.json` | 批量改动前 |
| `data/test-result/style-lab/backups/submodule-config-styles.before-second-sync.json` | 子模块第二次同步前 |
| `submodules/image-maker-artstyle/config-styles.json.bak` | 子模块工具自动备份 |
