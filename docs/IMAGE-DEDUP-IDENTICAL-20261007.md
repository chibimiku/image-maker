# 图片去重 `mode="identical"`（侯选 + 发布副本 / 浏览器重复副本）

用户 2026-10-07 反馈 `data/20261007` 顶层"一次制品经常出两个 jpg，看着一样"。核查结论：
**两张字节完全相同**，来源是发布名副本与它同轮的 `-aN` 候选；**与今天新加的"重新请求/审计重画"无关**。
本文件记规则与工具用法；完整证据见 `data/test-result/20261007/dedup-identical-20261007/README.md`。

## 两张同图是怎么来的

```
生成一张（第 N 轮） → api_backend 落盘 ...-aN_HHMMSS-token.jpg     ← 候选（审计台账指着它）
                    → promote_candidate() 再 copy2 成发布名 ...-first-token.jpg
                      （utils/first_image_audit.py:269 docstring / :321 shutil.copy2）
```

`single_analyzer._run_with_audit()` 把 `candidate_dir` 与 `publish_dir` 传成**同一个目录**
（`modules/image_analysis/single_analyzer.py:1847-1848`），所以候选与发布副本并排躺在日期目录里。
判据：两张的 `mtime` 精确到**微秒**相同（`copy2` 保元数据）；只出过一张图的任务（`stopped_reason=passed`）
也照样两张；43 个任务 58 个 `(任务,轮次)` 里没有任何一轮落两张、也没有两轮同字节。

**为什么 2026-10-07 才显形**：10:47 之前发布名后缀写死 `.png`（`single_analyzer.py:1311`），
挂在 JPEG 字节上算"假后缀"，那次去重把**发布名**删了；后缀纠正上线后发布副本变成合规 `.jpg`，
去重不再认领它 —— 今天留下的 9 张发布名副本全部 ≥ 12:59，唯一更早的 `bac7b4cc_012940` 是真 PNG。

## 为什么必须收敛（不只是占地方）

`publish_server.find_image_for_json()` 要求"同 hash 前缀的图片**恰好一张**"，否则返回 `None`
（`publish_server.py:233`）→ 拖投稿 JSON 入队时报"无法从结果 JSON 确定对应原图"。
实测 `data/20261007` 的 45 个投稿 JSON：反查失败 **19 → 去重后 16**。
剩下的 13 个仍是"同前缀多张"，因为**多轮任务里被审计打回的 `-a1` 候选还在原地**（既有行为）。

## 规则（`identical` 模式）

1. 按 `(目录, SHA-256)` 分组，只删**字节完全相同**、且同组还有幸存者的那份；
2. 幸存者次序：**发布队列在册（`uploading`/`uploaded`/`pending` 都算）** > 后缀没撒谎 >
   名字里带 `-first-` > **名字短** > 名字；
3. 一组里两份都在发布库里（`add_to_queue` 按 MD5 拦着，理论不该出现）→ **整组一张都不删**；
4. 实删前重新校验：**不在发布队列里**、文件还在、哈希没变、幸存者还在且哈希一致；
5. `misnamed`（默认，只删假后缀）与 `all`（连真后缀重复也收敛、**不看**发布队列）行为一字未改。

## 命令行

```powershell
python tools/image_dedup.py --dir data/20261007 --mode identical            # 预演（默认不删）
python tools/image_dedup.py --dir data/20261007 --mode identical --apply `
       --json data/test-result/20261007/dedup-identical-20261007/dedup-report.json
python tools/image_dedup.py --dir data/20261007 --mode identical --no-publish-db   # 不读队列，只看名字长短
```

2026-10-07 实跑：10 组 / 删 10 张 / 释放 27.53 MB；复扫同字节分组 0；
`data/publish/publish_queue.db` 前后 SHA-256 不变；`tools/repair_image_paths.py --report …` 预演 0 待改。

## 2026-10-07 追加：发布链改成「选中即 move 到发布名」（用户选方案 A）

用户原话：**「A 选中即 move 到发布名 这个方案吧，留着完全一样的没有意义」** —— 已实施：

- `utils.first_image_audit.promote_candidate()` 改成**改名**交付：`os.replace`（同卷原子改名、
  mtime 不变）→ 跨卷退 `shutil.move` → 都不成才退回复制；新增 `keep_copy=True` 保留旧复制语义；
  目标已存在且 SHA-256 相同 → 把候选那份重复**收掉**（只留一张），内容不同仍走 `unique_path` 另存两留；
- `generate_with_audit`：被选中那一轮的台账 `candidates[].file` 跟着指向交付文件，
  原候选名记在 `moved_from` / `selected_moved_from`（**不留悬空路径**）；
- 结果：一次过审 **只剩发布名一张**（原来是两张同字节）；重画后 = 被拒 `-a1` + 发布名，两者字节不同。

证据 / 可复跑探针 / 回归：`data/test-result/20261007/promote-move-20261007/README.md`。
**GUI 里是旧模块，必须重启 app 才按新行为落盘。**

## 2026-10-07 追加 2：历史目录里的「被拒候选」摘出 hash 前缀（用户要求「手动改掉」）

用户：*「多轮任务里被拒的 -a1 还在发布目录 … 这个已经存在的数据帮我手动改掉吧」*。
不删文件，把**没被交付**的历史候选改名成 `rejected-<原名>`：

- 判定三查（任一命中就不动）：名字是 `-aN` 候选？**在发布队列里**？是某份台账的 `selected_file`？
  另外每组至少留一张匹配前缀的（防止把整组搬空，反而一张都匹配不到）。
- 配套 `utils/rejected_candidates.py` + `tools/rename_rejected_candidates.py`
  （`--apply` 才动，默认预演，输出 `renames` 清单）。
- `tools/repair_image_paths.py` 扩展：认 `renames` 映射，并新增修 `candidates[].file`
  （旧路径已不在盘、清单给出在盘新路径时才改）。

20261007 实跑：**12 个文件改名 + 22 条台账路径改写**（12 条改名 + 10 条先前去重删掉的），
投稿 JSON 反查 **30/46 → 35/46**、无变坏、顶层 79 张一张没少、发布队列 DB 一个字节没动。
仍未解析的 11 个：8 个是"同 hash 有 2 张合法交付/已发图"（同一 hash 的任务重跑过，或两张都进过队列），
3 个是"该任务本来就没出图"。全库只读预演显示其它日期目录只剩 `20261006` 还有 1 个同类候选，未动。

证据：`data/test-result/20261007/rejected-rename/README.md`。

## 仍未做

- `publish_server.find_image_for_json` 的"同 hash 前缀恰好一张"对**多个合法交付/已发文件**依旧
  （同一 hash 的任务重跑、或两张都进过发布队列）返回 None —— 要彻底解决只能上方案 B（候选落子目录）
  或替用户指定"这个 hash 的代表图"，本轮都不做。
- 既有的一例：`17ba74d1_012837-first-1b682a-audit-first-image-audit.json` 指向的图整个 `data` 树里都不存在
  （该任务没留下产物），`repair_image_paths.py` 每次都会跳过并报告。
