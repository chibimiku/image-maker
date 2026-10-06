#!/usr/bin/env python
"""画风「按字段合并」验证 CLI（myc0t0xin 跨轮字段组合）。

不重跑分析轮次：从既有训练结果 JSON 里按字段复制指定轮次的内容，重新校验后跑三路测试生图
（Gemini 直出 / GPT-image-2 首图 / GPT 首图经 Gemini 文字模式重绘），再复用 app 的视觉复核与
本地深度指标链路比较。

用法示例::

    # 1) 无联网预检：字段来源/hash、生效提示词、输入图顺序、模型与尺寸全部落盘
    python tools/style_merge_validate.py precheck --source <训练结果.json> --out <输出根目录>

    # 2) 实跑三路 × 2 次
    python tools/style_merge_validate.py run --source <训练结果.json> --out <输出根目录> --repetitions 2

    # 3) 视觉复核（12 项量表 + 四项主体门禁）
    python tools/style_merge_validate.py compare --out <输出根目录> --subject-label primary

    # 4) 本地深度指标（Gram / AdaIN / LPIPS / CSD，NPU 优先）
    python tools/style_merge_validate.py deep --out <输出根目录> --subject-label primary

    # 5) 交付物：app 可读结果 JSON、候选包、字段来源、脱敏请求审计、建议条目
    python tools/style_merge_validate.py report --out <输出根目录> --subject-label primary

每个子命令都可用 `--subject-label <标签>` 指定主体分组（第二阶段用不同主体时另起一组，
不与第一阶段的自动比较集合混在一起）。
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.image_analysis.style_merge_validation import (  # noqa: E402
    PROBE_CHANNELS,
    MergeValidationTask,
    ReviewCandidateTask,
    ValidationError,
    atomic_json,
    build_candidate_package,
    build_precheck,
    build_precheck_document,
    build_review_package,
    build_review_state,
    build_state,
    channel_digests,
    cost_report,
    file_hash,
    read_json,
    request_digest,
    review_state_basename,
    review_state_path,
    run_attempt,
    run_deep_metrics,
    save_state,
    state_path,
    write_deliverables,
)


def _parse_candidates(values, keys=""):
    """`--candidate KEY=PATH[:package_key]` → 候选列表。

    路径里的盘符冒号不能当分隔符，所以只按**最后一段** `:` 后面的内容判断是否是包键
    （包键形如 `round_5`、`final`，不含路径分隔符）。
    """
    keys = [item.strip() for item in str(keys or "").split(",") if item.strip()]
    if keys and len(keys) != len(values or []):
        raise ValidationError(f"--candidate-keys 有 {len(keys)} 个，--candidate 有 {len(values or [])} 个，数量必须一致")
    items = []
    for index, raw in enumerate(values or []):
        text = str(raw)
        if "=" not in text:
            raise ValidationError(f"--candidate 需要 KEY=PATH 形式，收到 {text!r}")
        key, rest = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValidationError(f"--candidate 的包名不能为空：{text!r}")
        package_key = ""
        head, sep, tail = rest.rpartition(":")
        if sep and tail and os.path.sep not in tail and "/" not in tail and "." not in tail:
            rest, package_key = head, tail.strip()
        if keys:
            package_key = keys[index]
        label, _, label_text = key.partition("|")
        items.append({"key": label.strip() or f"candidate{index + 1}",
                      "label": (label_text.strip() or label.strip()),
                      "path": os.path.abspath(rest.strip()),
                      "package_key": package_key})
    seen = {}
    for item in items:
        if item["key"] in seen:
            raise ValidationError(f"候选包名重复：{item['key']}（包名会进候选 ID 与目录，必须唯一）")
        seen[item["key"]] = True
    return items


def _task(args, package_keys=("A", "B", "C")):
    overrides = {}
    if getattr(args, "gpt_model", ""):
        overrides["gpt_model"] = args.gpt_model
    if getattr(args, "gemini_model", ""):
        overrides["gemini_model"] = args.gemini_model
    common = dict(
        output_root=args.out,
        subject=args.subject or "",
        repetitions=args.repetitions,
        reuse_cache=not args.no_cache,
        subject_label=args.subject_label,
        stage=args.stage,
        model_overrides=overrides,
    )
    keys = tuple(key.strip() for key in
                 (str(getattr(args, "packages", "") or "").split(",")) if key.strip())
    candidates = _parse_candidates(getattr(args, "candidate", None) or [],
                                   getattr(args, "candidate_keys", "") or "")
    if candidates:
        task = ReviewCandidateTask(
            candidates=candidates,
            reference=getattr(args, "reference", "") or "",
            dry_run=bool(getattr(args, "dry_run", False)),
            **common,
        )
        task.state_name = review_state_basename(getattr(args, "state_name", "") or "ccccanh-review")
        if keys:
            unknown = [key for key in keys if key not in task.package_keys]
            if unknown:
                raise ValidationError(f"--packages 里的 {unknown} 不在 --candidate 里：{list(task.package_keys)}")
            task.package_keys = keys
        return task
    if not keys:
        keys = package_keys
    return MergeValidationTask(source=args.source, dry_run=bool(getattr(args, "dry_run", False)),
                               package_keys=keys, **common)


def _packages(task, keys):
    entries = {}
    for key in keys:
        if isinstance(task, ReviewCandidateTask):
            item = next(entry for entry in task.candidates if entry["key"] == key)
            entry = build_review_package(task, item)
        else:
            entry = build_candidate_package(task, key)
        entries[key] = entry
        print(f"[候选包 {key}] {entry['label']}｜字段来源={entry['field_sources']}")
        print(f"          gpt 短版合法={entry['normalization']['gpt_image_prompt_valid']}"
              f" errors={entry['normalization']['gpt_image_prompt_errors']}")
        if entry.get("motif_decision"):
            print(f"          母题：{entry['motif_decision']}")
    return entries


def _verify_source(task):
    if isinstance(task, ReviewCandidateTask):
        for item in task.candidates:
            state = task.candidate_state(item)
            info = {"path": os.path.abspath(str(item["path"])), "sha256": file_hash(str(item["path"])),
                    "bytes": os.path.getsize(str(item["path"]))}
            origin = _candidate_origin(task, item)
            print(f"[候选 {item['key']}] {info['path']}")
            print(f"         sha256={info['sha256']}  bytes={info['bytes']}  "
                  f"形态={origin['shape']}  取包位置={origin['record_path']}")
        if not os.path.isfile(task.source):
            raise ValidationError(f"兜底来源 JSON 不存在：{task.source}")
        print(f"[兜底来源] {task.source}")
        images = task.dataset_images()
        print(f"[源图] {len(images)} 张（逐张哈希写入 precheck.json；张数取自候选 JSON 记录，不写死）")
        subject = task.subject_prompt()
        print(f"[测试主体] {subject!r}")
        print(f"[画风参考图] {task.style_reference()}")
        return task.candidate_state(task.candidates[0]), images, subject, {}
    if not os.path.isfile(task.source):
        raise ValidationError(f"源结果 JSON 不存在：{task.source}")
    info = {"path": task.source, "sha256": file_hash(task.source), "bytes": os.path.getsize(task.source)}
    print(f"[源结果] {info['path']}")
    print(f"         sha256={info['sha256']}  bytes={info['bytes']}")
    state = task.source_state()
    images = task.dataset_images()
    print(f"[参考图] {len(images)} 张（逐张哈希写入 precheck.json）")
    subject = task.subject_prompt()
    print(f"[测试主体] {subject!r}")
    print(f"[画风参考图] {task.style_reference()}")
    return state, images, subject, info


def _candidate_origin(task, item):
    from modules.image_analysis.style_merge_validation import _candidate_variant_source
    return _candidate_variant_source(task, item)


def _build_entry(task, key):
    if isinstance(task, ReviewCandidateTask):
        item = next(entry for entry in task.candidates if entry["key"] == key)
        return build_review_package(task, item)
    return build_candidate_package(task, key)


def _load_entries(task, keys, precheck_path):
    """读预检结果；**必须重新核算通道哈希**，防止拿旧口径的预检文件做复用判定。

    真实事故（2026-10-05）：早期版本的通道哈希把「来源说明」也算进去了，于是包 B/C 的
    三条通道都被判成与包 A 逐字一致、直接复用了 A 的产物。改用只含传输内容的口径后，
    旧 precheck.json 里的哈希就不再成立 —— 宁可停下来要求重跑预检，也不能用错哈希复用。
    """
    if not os.path.isfile(precheck_path):
        raise ValidationError(f"缺少预检结果 {precheck_path}；请先跑 precheck（无联网）")
    stored = read_json(precheck_path)
    entries = {}
    for key in keys:
        entry = _build_entry(task, key)
        package_precheck = (stored.get("packages") or {}).get(key)
        if not package_precheck:
            raise ValidationError(f"预检结果里没有候选包 {key}")
        entry["precheck"] = package_precheck
        expected = channel_digests(package_precheck["channels"])
        actual = package_precheck.get("channel_digests") or {}
        if actual != expected:
            raise ValidationError(
                f"预检结果 {precheck_path} 的通道哈希与当前口径不一致（包 {key}）。"
                "旧口径会把不同的请求误判成相同而复用错误产物；请重跑 precheck 与 run。")
        entry["precheck"]["channel_digests"] = actual
        entry["precheck"]["request_digest"] = request_digest(package_precheck["channels"])
        entries[key] = entry
    return entries, stored


def command_precheck(args):
    task = _task(args)
    _, images, subject, _ = _verify_source(task)
    entries = _packages(task, task.package_keys)
    precheck = build_precheck_document(task, entries, images, subject, progress=print)
    target = os.path.join(task.output_root, args.subject_label, "precheck.json")
    atomic_json(target, precheck)
    print(f"[预检完成] {target}（未发送任何联网请求）")
    return 0


def command_run(args):
    packages = str(args.packages or "").strip()
    keys = tuple(key.strip() for key in packages.split(",") if key.strip()) or None
    task = _task(args, package_keys=keys or ("A", "B", "C"))
    _, images, subject, _ = _verify_source(task)
    precheck_path = os.path.join(task.output_root, args.subject_label, "precheck.json")
    if args.skip_precheck or isinstance(task, ReviewCandidateTask):
        # 复核模式**总是现场重算**：主体不同 → 组装出的提示词不同 → 通道哈希不同，
        # 沿用上一个主体的 precheck.json 会把别的场景当成同一个请求。
        entries = {}
        for key in task.package_keys:
            entry = _build_entry(task, key)
            entry["precheck"] = build_precheck(task, key, entry, subject, task.style_reference(), images,
                                               progress=print)
            entries[key] = entry
        if args.skip_precheck:
            print("[预检] 现场重算（--skip-precheck）：每主题的请求哈希不同，不能沿用上一主题的预检文件")
        else:
            print("[预检] 复核模式现场重算：主体不同，提示词与通道哈希必须按本主体重算")
        atomic_json(precheck_path, build_precheck_document(task, entries, images, subject))
    else:
        entries, _ = _load_entries(task, task.package_keys, precheck_path)
    records = {}
    for key in task.package_keys:
        entry = entries[key]
        for attempt in range(1, task.repetitions + 1):
            records[(key, attempt)] = run_attempt(task, key, entry, attempt, subject,
                                                 task.style_reference(), images, progress=print,
                                                 records=records)
    if isinstance(task, ReviewCandidateTask):
        state = build_review_state(task, entries, records)
        path = review_state_path(task)
        atomic_json(path, state)
    else:
        state = build_state(task, entries, records)
        path = save_state(task, state)
    print(f"[生图完成] 状态文件 {path}")
    print(f"[花费估算] {cost_report(entries, task)['total_usd']} USD（本地公式估算）")
    return 0


def _task_for_state(args):
    """比较/深度/报告阶段只需要「输出根目录 + 主体标签 + 状态文件名」，不需要重读候选字段。"""
    keys = tuple(key.strip() for key in str(getattr(args, "packages", "") or "A,B,C").split(",")
                 if key.strip())
    task = _task(args, package_keys=keys)
    if not isinstance(task, ReviewCandidateTask):
        directory = os.path.join(task.output_root, task.subject_label)
        explicit = str(getattr(args, "state_name", "") or "")
        if explicit:
            path = os.path.join(directory, review_state_basename(explicit))
        else:
            from glob import glob
            paths = sorted(glob(os.path.join(directory, "*_style_iter_result.json")))
            if len(paths) > 1:
                raise ValidationError("多个状态文件，请用 --state-name 明确指定")
            path = paths[0] if paths else state_path(task)
        if os.path.isfile(path):
            state = read_json(path)
            if state.get("review_validation"):
                pre = read_json(os.path.join(directory, "precheck.json"))
                task = ReviewCandidateTask(
                    candidates=pre["candidates"], reference=pre.get("reference_source") or "",
                    output_root=task.output_root, subject=pre["subject"],
                    subject_label=task.subject_label,
                    repetitions=max((int(r.get("attempt", 1)) for r in state.get("test_images", {}).values()), default=1))
                task.state_name = os.path.basename(path)
    return task


def _state_file(task):
    if isinstance(task, ReviewCandidateTask):
        return review_state_path(task)
    return state_path(task)


def command_compare(args):
    task = _task_for_state(args)
    path = _state_file(task)
    if not os.path.isfile(path):
        raise ValidationError(f"状态文件不存在：{path}；请先跑 run")
    from modules.others.api_backend import apply_secret_env_overrides, load_config
    from modules.image_analysis.style_comparison import compare_state_in_process
    config_data = apply_secret_env_overrides(load_config())
    base_url = str(config_data.get("base_url") or "")
    model = str(config_data.get("model") or "")
    key = str(config_data.get("api_key") or "")
    if not (base_url and model and key):
        raise ValidationError("文本视觉模型未配置（conf/config.json 顶层 base_url / model / key）")
    result = compare_state_in_process(path, (base_url, key, model), timeout=600, progress=print,
                                      cancelled=lambda: False)
    print(f"[视觉复核] {result.get('selection_status')}；best={result.get('best_id')}；"
          f"报告 {result.get('report_path')}")
    return 0


def command_deep(args):
    task = _task_for_state(args)
    path = _state_file(task)
    if not os.path.isfile(path):
        raise ValidationError(f"状态文件不存在：{path}")
    result = run_deep_metrics(task, path, device=args.device, progress=print)
    print(f"[深度指标] status={result.get('status')}；设备={result.get('backend', {}).get('actual')}"
          f"/{result.get('backend', {}).get('precision')}；报告 {result.get('report_path')}")
    return 0


def command_report(args):
    keys = tuple(key.strip() for key in str(getattr(args, "packages", "") or "A,B,C").split(",")
                 if key.strip())
    task = _task_for_state(args)
    subject_dir = os.path.dirname(_state_file(task))
    precheck_path = os.path.join(subject_dir, "precheck.json")
    entries, precheck = _load_entries(task, task.package_keys, precheck_path)
    state = read_json(_state_file(task))
    comparison = state.get("automatic_comparison") or {}
    deep = state.get("deep_feature_comparison") or {}
    attempts = {}
    for key in task.package_keys:
        for attempt in range(1, task.repetitions + 1):
            marker = os.path.join(subject_dir, "generated", key, f"attempt-{attempt}", "attempt.json")
            if os.path.isfile(marker):
                attempts[(key, attempt)] = read_json(marker)
    written = write_deliverables(task, entries, precheck, state, comparison, deep, attempts,
                                proposed_package=getattr(args, "proposed_package", None))
    from modules.image_analysis.style_merge_validation import (build_contact_sheet,
                                                               write_conclusion_markdown,
                                                               write_validation_report)
    sheet = build_contact_sheet(task, attempts)
    written["report"] = write_validation_report(task, entries, precheck, state, comparison, deep,
                                                attempts, contact_sheet=sheet)
    written["conclusion"] = write_conclusion_markdown(task, entries, precheck, state, comparison,
                                                      deep, attempts)
    if sheet:
        written["contact_sheet"] = sheet
    print("[交付物]")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="画风候选验证：myc0t0xin 跨轮字段合并（A/B/C）与按运行记录候选 JSON 取包的复核对照")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("precheck", command_precheck, "无联网预检：字段来源/hash、生效提示词、输入图顺序与参数"),
        ("run", command_run, "实跑三路测试生图（每包每次独立目录，成功产物缓存复用）"),
        ("compare", command_compare, "视觉复核：12 项量表 + 四项主体门禁（复用 app 同一段代码）"),
        ("deep", command_deep, "本地深度指标：Gram / AdaIN / LPIPS / CSD（NPU 优先）"),
        ("report", command_report, "交付物：结果 JSON、候选包、请求审计、报告与建议条目"),
    ):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("--source", default=os.path.join(
            "data", "20261004", "style-extraction", "myc0t0xin", "run-20261004-225929-156628",
            "myc0t0xin_style_iter_result.json"),
            help="源训练结果 JSON；复核模式下改作候选没记录数据集时的**兜底来源**")
        item.add_argument("--out", required=True, help="输出根目录（绝对路径或仓库相对路径）")
        item.add_argument("--subject-label", default="primary", help="主体分组标签（默认 primary）")
        item.add_argument("--stage", default="stage1", help="阶段标签（stage1 / stage2）")
        item.add_argument("--repetitions", type=int, default=2, help="每路重复次数（默认 2）")
        item.add_argument("--subject", default="", help="固定测试主体（默认沿用源运行记录）")
        item.add_argument("--no-cache", action="store_true", help="忽略已有成功产物缓存")
        item.add_argument("--gpt-model", default="",
                          help="覆盖 GPT 首图模型（默认读 apis.aigc-2d-gpt.model；覆盖值进请求审计）")
        item.add_argument("--gemini-model", default="",
                          help="覆盖 Gemini 模型（默认读 apis.aigc2d.model）")
        # 复核模式：按运行记录候选 JSON 取包（可只给部分字段，包名不写就按项数补 A/B/C）
        item.add_argument("--candidate", action="append", metavar="KEY=PATH[:package_key]",
                          help="候选包：`包名=候选JSON路径`（可选 `:round_5` 指定取哪一轮的包）；可重复")
        item.add_argument("--reference", default="",
                          help="复核模式的兜底来源 JSON（候选没记录数据集/筛图来源时用）")
        item.add_argument("--candidate-keys", default="",
                          help="逗号分隔的取包位置，顺序与 --candidate 一致（如 round_5,final）")
        item.add_argument("--state-name", default="",
                          help="状态文件名前缀（默认 cccccanh-review）；复核模式用")
        if name == "deep":
            item.add_argument("--device", default="auto-npu",
                              choices=["auto-npu", "npu", "cuda", "cpu", "auto"])
        if name == "report":
            item.add_argument("--proposed-package",
                              help="明确指定待观察的候选草稿；不启用，保留自动评分原结果")
        if name == "run":
            item.add_argument("--skip-precheck", action="store_true",
                              help="不用上一主题的 precheck.json，现场重算请求哈希（第二阶段换主体时用）")
        if name in ("run", "compare", "deep", "report"):
            item.add_argument("--packages", default="",
                              help="涉及哪些候选包；不填时：合并模式 A,B,C / 复核模式按 --candidate 顺序")
        item.set_defaults(handler=handler)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except ValidationError as exc:
        print(f"[停止] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[中断] 已有产物与预检记录保留", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
