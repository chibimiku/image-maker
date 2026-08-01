import os
import io
import json
import base64
import random
import datetime
from PIL import Image
from openai import OpenAI
from utils.task_runtime import append_log_line, set_task_status
from utils.prompt_loader import read_prompt_file, find_missing_prompt_files
from utils.styles import normalize_style_entry
from modules.others.api_backend import generate_image_whatai, generate_image_aigc2d

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTextEdit, QListWidget, QListWidgetItem, QFileDialog,
                             QMessageBox, QLabel, QAbstractItemView, QSpinBox, QGroupBox,
                             QFormLayout, QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap


def compress_and_encode_image(image_source, max_dim=2048):
    try:
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = image_source

        if img.mode != 'RGB':
            img = img.convert('RGB')

        original_width, original_height = img.size

        if max(original_width, original_height) > max_dim:
            scaling_factor = max_dim / max(original_width, original_height)
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=100)
        base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return "image/jpeg", base64_string

    except Exception as e:
        print(f"处理图片时发生错误: {e}")
        return None, None


STYLE_ITER_COMMON_PROMPT_FILE = "style-iter-common.md"
STYLE_ITER_REFINE_PROMPT_FILE = "style-iter-refine.md"
STYLE_ITER_FINAL_REVIEW_PROMPT_FILE = "style-iter-final-review.md"
STYLE_ITER_LOCAL_EXTRACT_PROMPT_FILE = "style-iter-local-extract.md"
STYLE_ITER_LOCAL_MERGE_PROMPT_FILE = "style-iter-local-merge.md"


def get_style_analyzer_missing_prompt_files():
    return find_missing_prompt_files([
        STYLE_ITER_COMMON_PROMPT_FILE,
        STYLE_ITER_REFINE_PROMPT_FILE,
        STYLE_ITER_FINAL_REVIEW_PROMPT_FILE,
        STYLE_ITER_LOCAL_EXTRACT_PROMPT_FILE,
        STYLE_ITER_LOCAL_MERGE_PROMPT_FILE,
    ])


class StyleIterCancelledError(Exception):
    pass


def _crop_image_regions(image_path, output_dir, crop_count=4):
    """从图片中按比例裁剪多个区域，返回裁剪后的文件路径列表。"""
    crops = []
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size

        # Define proportional crop regions for anime character art
        # All regions are horizontally centered with vertical slicing
        regions = [
            # (name, left_ratio, top_ratio, right_ratio, bottom_ratio)
            ("head_hair",    0.15, 0.00, 0.85, 0.42),   # Head & hair region
            ("face_closeup", 0.25, 0.08, 0.75, 0.38),   # Face close-up (tighter)
            ("upper_body",   0.10, 0.28, 0.90, 0.65),   # Upper body / torso
            ("lower_body",   0.10, 0.55, 0.90, 1.00),   # Lower body / legs / skirt
            ("detail_center",0.20, 0.15, 0.80, 0.85),   # Overall detail crop
        ]

        # Only use the requested number of regions
        used_regions = regions[:max(1, min(crop_count, len(regions)))]

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Sanitize base name for file system
        safe_base = "".join(c for c in base_name if c.isalnum() or c in "._- ").strip() or "crop"

        for region_name, l, t, r, b in used_regions:
            left = int(w * l)
            top = int(h * t)
            right = int(w * r)
            bottom = int(h * b)

            # Ensure valid crop dimensions
            if right <= left or bottom <= top:
                continue

            crop_img = img.crop((left, top, right, bottom))

            # Save to output directory
            crop_filename = f"{safe_base}_{region_name}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            crop_img.save(crop_path, "JPEG", quality=95)
            crops.append(crop_path)

    except Exception as e:
        print(f"裁剪图片区域时出错 {image_path}: {e}")

    return crops


class StyleIterativeWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    save_signal = pyqtSignal(str, str)
    finish_signal = pyqtSignal(str, str, str)

    def __init__(self, image_paths, api_key, base_url, model_name,
                 total_rounds=3, images_per_round=2,
                 existing_state=None, output_dir="", timeout_seconds=120,
                 enable_test_gen=False, test_prompt="",
                 img_api_type="", img_instructions="", img_aspect_ratio="1:1",
                 file_prefix="", seed_prompts=""):
        super().__init__()
        self.image_paths = list(image_paths)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.total_rounds = int(total_rounds)
        self.images_per_round = int(images_per_round)
        self.existing_state = existing_state
        self.output_dir = str(output_dir or "").strip()
        self.timeout_seconds = int(timeout_seconds)
        self._cancel_requested = False
        # Test generation params
        self.enable_test_gen = bool(enable_test_gen)
        self.test_prompt = str(test_prompt or "").strip()
        self.img_api_type = str(img_api_type or "").strip()
        self.img_instructions = str(img_instructions or "").strip()
        self.img_aspect_ratio = str(img_aspect_ratio or "1:1").strip() or "1:1"
        self.file_prefix = str(file_prefix).strip()
        self.seed_prompts = str(seed_prompts or "").strip()

    def request_cancel(self):
        self._cancel_requested = True
        self.requestInterruption()

    def _check_cancel(self):
        if self._cancel_requested or self.isInterruptionRequested():
            raise StyleIterCancelledError()

    def _build_state(self, step_count, iterations):
        return {
            "version": "1.0",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": {
                "image_count": len(self.image_paths),
                "images": [os.path.abspath(p) for p in self.image_paths],
            },
            "parameters": {
                "total_rounds": self.total_rounds,
                "images_per_round": self.images_per_round,
            },
            "iterations": iterations,
            "final_art_style_prompts": "",
            "file_prefix": self.file_prefix,
        }

    def _save_state(self, state, output_path=None):
        state["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if output_path is None:
            output_path = os.path.join(self.output_dir, "style_iter_result.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return output_path

    def _build_iteration_history_text(self, iterations):
        if not iterations:
            return "No prior iterations."
        lines = []
        for it in iterations:
            it_type = it.get("type", "unknown")
            it_round = it.get("round", "?")
            step = it.get("step", "?")
            if it_type == "commonality_extraction":
                lines.append(f"--- Iteration Step {step} (Round {it_round}, Commonality Extraction) ---")
                lines.append(it.get("art_style_prompts", ""))
                lines.append("")
            elif it_type == "refinement_check":
                lines.append(f"--- Iteration Step {step} (Round {it_round}, Check {it.get('check_index', '?')}) ---")
                lines.append(f"Image: {os.path.basename(it.get('image', ''))}")
                lines.append(f"Prompts Before: {it.get('prompts_before', '')}")
                lines.append(f"Differences Found: {it.get('differences_analysis', '')}")
                lines.append(f"Prompts After: {it.get('prompts_after', '')}")
                lines.append("")
            elif it_type == "final_review":
                lines.append(f"--- Iteration Step {step} (Final Review, {it_round}) ---")
                lines.append(f"Prompts Before: {it.get('prompts_before', '')}")
                lines.append(f"Review Analysis: {it.get('final_review_analysis', '')}")
                lines.append(f"Prompts After: {it.get('prompts_after', '')}")
                lines.append("")
            elif it_type == "imported_prompts":
                lines.append(f"--- Iteration Step {step} (Seed, Imported Prompts) ---")
                lines.append(it.get("art_style_prompts", ""))
                lines.append("")
            elif it_type == "local_extract":
                lines.append(f"--- Iteration Step {step} (Local Refine Extract, {it_round}, Batch {it.get('batch_index', '?')}) ---")
                lines.append(f"Crop Types: {it.get('crop_types_analyzed', '')}")
                lines.append(f"Findings: {it.get('localized_style_findings', '')}")
                lines.append(f"Confidence: {it.get('confidence', '?')}")
                lines.append("")
            elif it_type == "local_merge":
                lines.append(f"--- Iteration Step {step} (Local Refine Merge, {it_round}) ---")
                lines.append(f"Prompts Before: {it.get('prompts_before', '')}")
                lines.append(f"Merge Analysis: {it.get('merge_analysis', '')}")
                lines.append(f"Prompts After: {it.get('prompts_after', '')}")
                lines.append("")
        return "\n".join(lines)

    def _get_current_prompts(self, iterations):
        for it in reversed(iterations):
            if it.get("type") == "final_review":
                return it.get("prompts_after", "")
            if it.get("type") == "local_merge":
                return it.get("prompts_after", "")
            if it.get("type") == "refinement_check":
                return it.get("prompts_after", "")
            if it.get("type") == "commonality_extraction":
                return it.get("art_style_prompts", "")
            if it.get("type") == "imported_prompts":
                return it.get("art_style_prompts", "")
        return ""

    def _generate_test_image(self, current_prompts, round_num, output_path):
        """每轮结束后生成一张测试图片：艺术风格 prompts + 用户测试 prompt。"""
        self._check_cancel()
        gen_prompt = f"{current_prompts}\n\n{self.test_prompt}"
        self.log_signal.emit(f"[Round {round_num}] 测试生图: 正在调用 {self.img_api_type} 生成测试图片...")
        self.progress_signal.emit(f"Round {round_num}/{self.total_rounds} — 生成测试图片中…")

        try:
            if self.img_api_type == "aigc2d":
                result = generate_image_aigc2d(
                    prompt=gen_prompt,
                    model="",
                    aspect_ratio=self.img_aspect_ratio,
                    instructions=self.img_instructions,
                    api_type=self.img_api_type,
                    file_prefix=f"style_test_r{round_num}",
                    cancel_check=lambda: self._cancel_requested or self.isInterruptionRequested(),
                )
            else:
                result = generate_image_whatai(
                    prompt=gen_prompt,
                    model="",
                    aspect_ratio=self.img_aspect_ratio,
                    instructions=self.img_instructions,
                    api_type=self.img_api_type,
                    file_prefix=f"style_test_r{round_num}",
                    cancel_check=lambda: self._cancel_requested or self.isInterruptionRequested(),
                )

            self._check_cancel()

            if isinstance(result, dict):
                saved_files = result.get("saved_files", []) or []
            elif isinstance(result, list):
                saved_files = result
            else:
                saved_files = []

            if saved_files:
                self.log_signal.emit(f"  测试图片已生成: {len(saved_files)} 张")
                for f in saved_files:
                    self.log_signal.emit(f"    {f}")
                return saved_files
            else:
                self.log_signal.emit(f"  警告：测试生图未返回图片文件。")
                return []
        except StyleIterCancelledError:
            raise
        except Exception as e:
            self.log_signal.emit(f"  测试生图失败: {e}")
            return []

    def _step_commonality_extraction(self, client, iterations, current_round):
        self._check_cancel()
        self.log_signal.emit(f"[Round {current_round}] Phase 1: 提取图片共性，生成艺术风格 prompts...")
        self.progress_signal.emit(f"Round {current_round}/{self.total_rounds} — 共性提取中…")

        content_list = [{"type": "text", "text": read_prompt_file(STYLE_ITER_COMMON_PROMPT_FILE).strip()}]

        # In rounds 2+, include previous iteration history as context
        if current_round > 1 and iterations:
            history_text = self._build_iteration_history_text(iterations)
            context_text = (
                "\n\nCONTEXT FROM PREVIOUS ITERATIONS:\n"
                f"The following is the iteration history from previous rounds. "
                f"Use this to build upon what has been learned, not start from scratch.\n\n"
                f"{history_text}"
            )
            content_list[0]["text"] += context_text

        for index, path in enumerate(self.image_paths, start=1):
            self._check_cancel()
            self.log_signal.emit(f"  [{index}/{len(self.image_paths)}] 编码图片: {os.path.basename(path)}")
            mime_type, base64_image = compress_and_encode_image(path)
            if base64_image:
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high"
                    }
                })

        self._check_cancel()
        self.log_signal.emit("  正在向 LLM 发送共性提取请求，请耐心等待...")

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content_list}],
            temperature=0.7,
            max_completion_tokens=4096,
            timeout=self.timeout_seconds,
        )
        self._check_cancel()

        art_style_prompts = response.choices[0].message.content.strip()
        self.log_signal.emit(f"  共性提取完成。")

        iteration_record = {
            "step": len(iterations) + 1,
            "type": "commonality_extraction",
            "round": current_round,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "art_style_prompts": art_style_prompts,
            "model_used": self.model_name,
        }
        iterations.append(iteration_record)
        return iterations, art_style_prompts

    def _step_refinement_check(self, client, iterations, current_round, check_index, image_path, current_prompts):
        self._check_cancel()
        basename = os.path.basename(image_path)
        self.log_signal.emit(f"[Round {current_round}] Phase 2 [{check_index}/{self.images_per_round}]: 检查图片 \"{basename}\"...")
        self.progress_signal.emit(f"Round {current_round}/{self.total_rounds} — 图片检查 {check_index}/{self.images_per_round}")

        history_text = self._build_iteration_history_text(iterations)

        refine_prompt = read_prompt_file(STYLE_ITER_REFINE_PROMPT_FILE).strip()
        refine_prompt += (
            "\n\n=== CURRENT ART STYLE PROMPTS (TO EVALUATE & REVISE) ===\n"
            f"{current_prompts}\n\n"
            "=== ITERATION HISTORY (ALL PREVIOUS STEPS) ===\n"
            f"{history_text}\n\n"
            "Please analyze the image above against the Current Art Style Prompts, "
            "consider the Iteration History, and produce your JSON response with differences_found and revised_prompts."
        )

        mime_type, base64_image = compress_and_encode_image(image_path)
        if not base64_image:
            self.log_signal.emit(f"  跳过：无法编码图片 {basename}")
            return iterations, current_prompts

        self._check_cancel()
        self.log_signal.emit("  正在向 LLM 发送风格差异分析请求...")

        response = client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": refine_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]}
            ],
            temperature=0.5,
            max_completion_tokens=8192,
            timeout=self.timeout_seconds,
        )
        self._check_cancel()

        try:
            result_json = json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError as e:
            self.log_signal.emit(f"  警告：JSON 解析失败 ({e})，保留当前 prompts 不变。")
            return iterations, current_prompts

        differences_found = str(result_json.get("differences_found", "")).strip()
        revised_prompts = str(result_json.get("revised_prompts", current_prompts)).strip()
        confidence = result_json.get("confidence", 0.5)

        if not revised_prompts:
            revised_prompts = current_prompts

        self.log_signal.emit(f"  检查完成 (置信度: {confidence})。差异: {differences_found[:200]}...")

        iteration_record = {
            "step": len(iterations) + 1,
            "type": "refinement_check",
            "round": current_round,
            "check_index": check_index,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image": os.path.abspath(image_path),
            "prompts_before": current_prompts,
            "differences_analysis": differences_found,
            "prompts_after": revised_prompts,
            "confidence": confidence,
            "model_used": self.model_name,
        }
        iterations.append(iteration_record)
        return iterations, revised_prompts

    def _step_local_refinement(self, client, iterations, current_prompts, total_rounds):
        """在所有轮次完成后，将所有参考图裁剪为局部区域，分批发送给 LLM 提取局部细节，再合并到主 prompts。"""
        self._check_cancel()
        self.log_signal.emit(f"[Local Refine] 所有 {total_rounds} 轮迭代完成，开始局部区域精细化分析...")
        self.progress_signal.emit("局部细化中 — 裁剪区域并分批分析…")

        # 1) Create unique temp directory for crops per run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        crop_dir = os.path.join(self.output_dir, f"local_crops_{timestamp}")
        os.makedirs(crop_dir, exist_ok=True)

        # 2) Crop all images
        all_crops = []
        self.log_signal.emit(f"  [Local Refine] 正在裁剪 {len(self.image_paths)} 张参考图...")
        for index, path in enumerate(self.image_paths, start=1):
            self._check_cancel()
            crops = _crop_image_regions(path, crop_dir, crop_count=4)
            all_crops.extend(crops)
            self.log_signal.emit(f"  [Local Refine] [{index}/{len(self.image_paths)}] {os.path.basename(path)} → {len(crops)} 个局部区域")

        if not all_crops:
            self.log_signal.emit("  [Local Refine] 警告：未能生成任何裁剪区域，跳过局部细化。")
            return iterations, current_prompts

        self.log_signal.emit(f"  [Local Refine] 共生成 {len(all_crops)} 个局部裁剪区域。")

        # 3) Batch process crops (4-6 per batch)
        BATCH_SIZE = 5
        localized_findings = []
        for batch_idx in range(0, len(all_crops), BATCH_SIZE):
            self._check_cancel()
            batch_crops = all_crops[batch_idx:batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1
            total_batches = (len(all_crops) + BATCH_SIZE - 1) // BATCH_SIZE

            self.log_signal.emit(f"  [Local Refine] 批次 {batch_num}/{total_batches}：正在分析 {len(batch_crops)} 个局部区域...")
            self.progress_signal.emit(f"局部细化 — 批次 {batch_num}/{total_batches}")

            content_list = [{
                "type": "text",
                "text": (
                    read_prompt_file(STYLE_ITER_LOCAL_EXTRACT_PROMPT_FILE).strip() +
                    "\n\nNOTE: These are CROPPED LOCAL REGIONS from reference images. "
                    "Your findings will later be merged into a main art style description. "
                    "Focus on localized rendering details that may not be visible in full-image views."
                )
            }]

            for crop_path in batch_crops:
                mime_type, base64_image = compress_and_encode_image(crop_path)
                if base64_image:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high"
                        }
                    })

            self._check_cancel()
            response = client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content_list}],
                temperature=0.5,
                max_completion_tokens=4096,
                timeout=self.timeout_seconds,
            )
            self._check_cancel()

            try:
                result_json = json.loads(response.choices[0].message.content.strip())
            except json.JSONDecodeError as e:
                self.log_signal.emit(f"  [Local Refine] 批次 {batch_num} JSON 解析失败 ({e})，跳过。")
                continue

            findings = str(result_json.get("localized_style_findings", "")).strip()
            crop_types = str(result_json.get("crop_types_analyzed", "")).strip()
            confidence = result_json.get("confidence", 0.5)

            if findings:
                localized_findings.append({
                    "batch": batch_num,
                    "crop_types": crop_types,
                    "findings": findings,
                    "confidence": confidence,
                })
                self.log_signal.emit(f"  [Local Refine] 批次 {batch_num} 完成 (置信度: {confidence})，发现: {findings[:120]}...")

                # Record intermediate extraction in iterations
                it_record = {
                    "step": len(iterations) + 1,
                    "type": "local_extract",
                    "round": f"round-{total_rounds}-local-extract-batch-{batch_num}",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "batch_index": batch_num,
                    "crop_count": len(batch_crops),
                    "crop_types_analyzed": crop_types,
                    "localized_style_findings": findings,
                    "confidence": confidence,
                    "model_used": self.model_name,
                }
                iterations.append(it_record)
            else:
                self.log_signal.emit(f"  [Local Refine] 批次 {batch_num} 完成但无有效发现。")

        if not localized_findings:
            self.log_signal.emit("  [Local Refine] 所有批次均未产生有效发现，跳过合并。")
            return iterations, current_prompts

        # 4) Merge all localized findings into main prompts
        self._check_cancel()
        self.log_signal.emit(f"  [Local Refine] 正在合并 {len(localized_findings)} 组局部发现到主 prompts...")
        self.progress_signal.emit("局部细化 — 合并局部发现到主描述中")

        all_findings_text = "\n\n---\n\n".join(
            f"### Batch {f['batch']} (Crop types: {f['crop_types']}, Confidence: {f['confidence']})\n{f['findings']}"
            for f in localized_findings
        )

        merge_prompt = (
            read_prompt_file(STYLE_ITER_LOCAL_MERGE_PROMPT_FILE).strip() +
            "\n\n=== MAIN ART STYLE PROMPTS (TO ENRICH) ===\n" +
            current_prompts +
            "\n\n=== LOCALIZED STYLE FINDINGS (FROM CROPPED REGIONS) ===\n" +
            all_findings_text +
            "\n\nPlease merge the localized findings into the main prompts and return your JSON response."
        )

        self._check_cancel()
        response = client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": merge_prompt}],
            temperature=0.5,
            max_completion_tokens=8192,
            timeout=self.timeout_seconds,
        )
        self._check_cancel()

        try:
            result_json = json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError as e:
            self.log_signal.emit(f"  [Local Refine] 合并 JSON 解析失败 ({e})，保留原 prompts。")
            return iterations, current_prompts

        merge_analysis = str(result_json.get("merge_analysis", "")).strip()
        merged_prompts = str(result_json.get("merged_prompts", current_prompts)).strip()
        confidence = result_json.get("confidence", 0.5)

        if not merged_prompts:
            merged_prompts = current_prompts

        self.log_signal.emit(f"  [Local Refine] 合并完成 (置信度: {confidence})。")
        self.log_signal.emit(f"  [Local Refine] 合并分析: {merge_analysis[:300]}...")

        merge_record = {
            "step": len(iterations) + 1,
            "type": "local_merge",
            "round": f"round-{total_rounds}-local-merge",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompts_before": current_prompts,
            "merge_analysis": merge_analysis,
            "batches_merged": len(localized_findings),
            "prompts_after": merged_prompts,
            "confidence": confidence,
            "model_used": self.model_name,
        }
        iterations.append(merge_record)
        return iterations, merged_prompts

    def _step_final_review(self, client, iterations, current_prompts, total_rounds):
        """在所有轮次完成后，将所有参考图和最终 prompts 发送给 LLM 做一次最终审查。"""
        self._check_cancel()
        self.log_signal.emit(f"[Final Review] 所有 {total_rounds} 轮迭代已完成，正在进行最终综合审查...")
        self.progress_signal.emit("最终审查中 — 综合所有参考图进行最终修正…")

        content_list = [{
            "type": "text",
            "text": read_prompt_file(STYLE_ITER_FINAL_REVIEW_PROMPT_FILE).strip()
        }]

        # Include iteration history as context
        history_text = self._build_iteration_history_text(iterations)
        context_text = (
            "\n\n=== FINAL ART STYLE PROMPTS (AFTER ALL ROUNDS) ===\n"
            f"{current_prompts}\n\n"
            "=== COMPLETE ITERATION HISTORY ===\n"
            f"{history_text}\n\n"
            "Now review ALL reference images below against the Final Prompts and produce your JSON response."
        )
        content_list[0]["text"] += context_text

        # Encode ALL reference images
        for index, path in enumerate(self.image_paths, start=1):
            self._check_cancel()
            self.log_signal.emit(f"  [Final Review] [{index}/{len(self.image_paths)}] 编码图片: {os.path.basename(path)}")
            mime_type, base64_image = compress_and_encode_image(path)
            if base64_image:
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high"
                    }
                })

        self._check_cancel()
        self.log_signal.emit("  [Final Review] 正在向 LLM 发送最终综合审查请求，请耐心等待...")

        response = client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content_list}],
            temperature=0.5,
            max_completion_tokens=8192,
            timeout=self.timeout_seconds,
        )
        self._check_cancel()

        try:
            result_json = json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError as e:
            self.log_signal.emit(f"  [Final Review] 警告：JSON 解析失败 ({e})，保留最终 prompts 不变。")
            return iterations, current_prompts

        final_analysis = str(result_json.get("final_review_analysis", "")).strip()
        final_prompts = str(result_json.get("final_prompts", current_prompts)).strip()
        confidence = result_json.get("confidence", 0.5)

        if not final_prompts:
            final_prompts = current_prompts

        self.log_signal.emit(f"  [Final Review] 审查完成 (置信度: {confidence})。")
        self.log_signal.emit(f"  [Final Review] 分析摘要: {final_analysis[:300]}...")

        iteration_record = {
            "step": len(iterations) + 1,
            "type": "final_review",
            "round": f"round-{total_rounds}-after",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompts_before": current_prompts,
            "final_review_analysis": final_analysis,
            "prompts_after": final_prompts,
            "confidence": confidence,
            "model_used": self.model_name,
        }
        iterations.append(iteration_record)
        return iterations, final_prompts

    def run(self):
        try:
            self._check_cancel()
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_seconds)
        except Exception as e:
            self.log_signal.emit(f"初始化 API 客户端失败: {e}")
            self.finish_signal.emit("error", "", "")
            return

        # Determine output directory
        if not self.output_dir:
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            self.output_dir = os.path.join("data", date_str)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize or load existing state
        if self.existing_state and isinstance(self.existing_state, dict):
            iterations = self.existing_state.get("iterations", [])
            self.log_signal.emit(f"📂 已加载已有训练状态，包含 {len(iterations)} 个迭代步骤。")
            # Determine starting round from existing iterations
            if iterations:
                # Filter only numeric rounds (exclude final_review entries with string rounds like "round-3-after")
                numeric_rounds = [it.get("round", 1) for it in iterations
                                  if isinstance(it.get("round", 1), (int, float))]
                last_round = int(max(numeric_rounds)) if numeric_rounds else 0
                start_round = last_round + 1
                self.log_signal.emit(f"  上次训练到了 Round {last_round}，将从 Round {start_round} 继续。")
            else:
                start_round = 1
        else:
            iterations = []
            start_round = 1

        self.log_signal.emit(f"🚀 开始多轮迭代画风提取：共 {self.total_rounds} 轮，每轮检查 {self.images_per_round} 张图片。")
        self.log_signal.emit(f"📁 数据集共 {len(self.image_paths)} 张图片。")
        self.log_signal.emit(f"💾 输出目录: {self.output_dir}")

        current_prompts = ""

        # Inject seed prompts as initial iteration if provided and not loading existing state
        if self.seed_prompts and not self.existing_state:
            self.log_signal.emit(f"📝 已导入初始 Prompts 作为训练起点。")
            seed_record = {
                "step": 0,
                "type": "imported_prompts",
                "round": 0,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "art_style_prompts": self.seed_prompts,
                "model_used": "",
            }
            iterations.insert(0, seed_record)
            current_prompts = self.seed_prompts

        state = self._build_state(len(iterations), iterations)
        prefix = self.file_prefix if self.file_prefix else "style"
        output_path = os.path.join(self.output_dir, f"{prefix}_style_iter_result.json")

        try:
            for round_num in range(start_round, start_round + self.total_rounds):
                self._check_cancel()

                # Adjust images_per_round if dataset is too small
                effective_images_per_round = min(self.images_per_round, len(self.image_paths))
                if effective_images_per_round < self.images_per_round:
                    self.log_signal.emit(f"⚠️ 图片数量 ({len(self.image_paths)}) 不足每轮检查数 ({self.images_per_round})，已自动调整为 {effective_images_per_round}。")

                # Phase 1: Commonality extraction
                iterations, current_prompts = self._step_commonality_extraction(
                    client, iterations, round_num
                )
                state = self._build_state(len(iterations), iterations)
                state["final_art_style_prompts"] = current_prompts
                self._save_state(state, output_path)
                self.log_signal.emit(f"  💾 已保存 (step {len(iterations)})")

                # Phase 2: Refinement checks
                available_images = list(self.image_paths)
                selected_images = random.sample(
                    available_images,
                    min(effective_images_per_round, len(available_images))
                )

                for check_idx, image_path in enumerate(selected_images, start=1):
                    self._check_cancel()
                    iterations, current_prompts = self._step_refinement_check(
                        client, iterations, round_num, check_idx, image_path, current_prompts
                    )
                    state = self._build_state(len(iterations), iterations)
                    state["final_art_style_prompts"] = current_prompts
                    self._save_state(state, output_path)
                    self.log_signal.emit(f"  💾 已保存 (step {len(iterations)})")

                # Phase 3 (optional): Test image generation
                if self.enable_test_gen and self.test_prompt:
                    self._check_cancel()
                    test_files = self._generate_test_image(current_prompts, round_num, output_path)
                    if "test_images" not in state:
                        state["test_images"] = {}
                    state["test_images"][f"round_{round_num}"] = {
                        "round": round_num,
                        "prompts_used": current_prompts,
                        "test_prompt": self.test_prompt,
                        "generated_files": test_files,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    self._save_state(state, output_path)

                self.log_signal.emit(f"✅ Round {round_num} 完成。")

            # Local refinement: crop all images into regions, batch-analyze, merge into prompts
            iterations, current_prompts = self._step_local_refinement(
                client, iterations, current_prompts, self.total_rounds
            )
            state = self._build_state(len(iterations), iterations)
            state["final_art_style_prompts"] = current_prompts
            self._save_state(state, output_path)
            self.log_signal.emit(f"  💾 局部细化结果已保存 (step {len(iterations)})")

            # Final comprehensive review: send all images + final prompts to LLM
            iterations, current_prompts = self._step_final_review(
                client, iterations, current_prompts, self.total_rounds
            )
            state = self._build_state(len(iterations), iterations)
            state["final_art_style_prompts"] = current_prompts
            self._save_state(state, output_path)
            self.log_signal.emit(f"  💾 最终审查结果已保存 (step {len(iterations)})")

            self.log_signal.emit("=" * 60)
            self.log_signal.emit(f"🎉 全部 {self.total_rounds} 轮迭代完成！")
            self.log_signal.emit(f"📄 最终结果已保存至: {output_path}")
            self.log_signal.emit(f"📊 共执行 {len(iterations)} 个步骤。")
            self.finish_signal.emit("success", current_prompts, output_path)

        except StyleIterCancelledError:
            self.log_signal.emit("🛑 已取消多轮迭代画风提取任务。")
            if iterations:
                state = self._build_state(len(iterations), iterations)
                state["final_art_style_prompts"] = self._get_current_prompts(iterations)
                try:
                    self._save_state(state, output_path)
                    self.log_signal.emit(f"💾 已保存当前进度至: {output_path}")
                except Exception:
                    pass
            self.finish_signal.emit("cancelled", self._get_current_prompts(iterations), output_path)
        except Exception as e:
            self.log_signal.emit(f"❌ 多轮迭代发生错误: {e}")
            if iterations:
                state = self._build_state(len(iterations), iterations)
                state["final_art_style_prompts"] = self._get_current_prompts(iterations)
                try:
                    self._save_state(state, output_path)
                except Exception:
                    pass
            self.finish_signal.emit("error", self._get_current_prompts(iterations), output_path)


class ImageDropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(100, 100))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                self.add_image_item(file_path)

    def truncate_text(self, text, max_len=16):
        if len(text) <= max_len:
            return text
        name, ext = os.path.splitext(text)
        keep_len = max_len - len(ext) - 3
        if keep_len > 0:
            return name[:keep_len] + "..." + ext
        return text[:max_len] + "..."

    def add_image_item(self, file_path):
        for i in range(self.count()):
            if self.item(i).data(Qt.ItemDataRole.UserRole) == file_path:
                return

        item = QListWidgetItem(self)
        item.setData(Qt.ItemDataRole.UserRole, file_path)

        filename = os.path.basename(file_path)
        item.setText(self.truncate_text(filename))
        item.setToolTip(filename)

        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            item.setIcon(
                QIcon(
                    pixmap.scaled(
                        100,
                        100,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            )
        self.addItem(item)

    def set_image_paths(self, paths):
        self.clear()
        for p in paths:
            if os.path.isfile(p):
                self.add_image_item(p)


class StyleAnalyzerWidget(QWidget):
    def __init__(self, config_getter_func, timeout_getter_func=None,
                 img_config_getter_func=None, styles_getter_func=None,
                 test_gen_default_getter_func=None, test_gen_changed_callback=None,
                 test_prompt_getter_func=None, test_prompt_changed_callback=None):
        super().__init__()
        self.get_config = config_getter_func
        self.get_timeout = timeout_getter_func
        self.get_img_config = img_config_getter_func
        self.get_styles = styles_getter_func
        self.get_test_gen_default = test_gen_default_getter_func
        self.on_test_gen_changed = test_gen_changed_callback
        self.get_test_prompt_default = test_prompt_getter_func
        self.on_test_prompt_changed = test_prompt_changed_callback
        self.thread = None
        self._loaded_json_path = ""
        self._existing_state = None
        self._loaded_image_paths = []
        self._output_dir = ""
        self._imported_prompts = ""
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Top: hint
        top_layout = QHBoxLayout()
        hint_label = QLabel("拖拽多张同画风图片至下方区域进行多轮迭代综合分析：")
        top_layout.addWidget(hint_label)

        self.add_btn = QPushButton("添加图片")
        self.add_btn.clicked.connect(self.browse_images)
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.clear_images)

        top_layout.addStretch()
        top_layout.addWidget(self.add_btn)
        top_layout.addWidget(self.clear_btn)
        layout.addLayout(top_layout)

        # Image list
        self.image_list = ImageDropListWidget()
        self.image_list.setMinimumHeight(150)
        layout.addWidget(self.image_list)

        # Parameters group
        params_group = QGroupBox("迭代参数")
        params_layout = QFormLayout()

        self.total_rounds_spin = QSpinBox()
        self.total_rounds_spin.setRange(1, 50)
        self.total_rounds_spin.setValue(3)
        self.total_rounds_spin.setToolTip("总共执行多少轮迭代训练")
        params_layout.addRow("训练轮次:", self.total_rounds_spin)

        self.images_per_round_spin = QSpinBox()
        self.images_per_round_spin.setRange(1, 100)
        self.images_per_round_spin.setValue(2)
        self.images_per_round_spin.setToolTip("每轮随机抽取多少张图片进行差异检查和修正")
        params_layout.addRow("每轮检查图片数:", self.images_per_round_spin)

        self.file_prefix_input = QLineEdit()
        self.file_prefix_input.setPlaceholderText("必填，用于区分不同画风的 JSON 文件名，如 cyberpunk、watercolor")
        self.file_prefix_input.setClearButtonEnabled(True)
        self.file_prefix_input.setToolTip("此前缀将作为 JSON 文件名的一部分，避免不同画风结果互相覆盖")
        self.file_prefix_input.setStyleSheet("QLineEdit { background-color: #fffdf0; }")
        params_layout.addRow("* 画风前缀:", self.file_prefix_input)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Test generation group
        test_gen_group = QGroupBox("测试生图")
        test_gen_layout = QFormLayout()

        self.enable_test_gen_cb = QCheckBox("每轮结束后生成测试图片")
        self.enable_test_gen_cb.setToolTip(
            "勾选后，每轮迭代完成后将基于当前艺术风格 prompts 拼接下方输入的测试提示词，"
            "调用生图 API 生成一张测试图片，用于直观评估风格 prompts 的准确性。"
        )
        self.enable_test_gen_cb.setChecked(
            bool(self.get_test_gen_default()) if self.get_test_gen_default else True
        )
        self.enable_test_gen_cb.toggled.connect(self._on_test_gen_toggled)
        test_gen_layout.addRow(self.enable_test_gen_cb)

        self.test_prompt_input = QLineEdit()
        self.test_prompt_input.setPlaceholderText(
            "输入测试提示词，例如：1girl, solo, standing, looking at viewer"
        )
        self.test_prompt_input.setClearButtonEnabled(True)
        # 自动保存输入内容
        self.test_prompt_input.editingFinished.connect(self._on_test_prompt_changed)
        # 恢复上次保存的内容
        if self.get_test_prompt_default:
            saved = self.get_test_prompt_default()
            if saved:
                self.test_prompt_input.setText(saved)
        test_gen_layout.addRow("测试提示词:", self.test_prompt_input)

        test_gen_group.setLayout(test_gen_layout)
        layout.addWidget(test_gen_group)

        # Import JSON
        import_layout = QHBoxLayout()
        self.import_json_btn = QPushButton("📂 导入已有训练 JSON 继续训练")
        self.import_json_btn.clicked.connect(self.import_existing_json)
        self.clear_import_btn = QPushButton("清除导入")
        self.clear_import_btn.setEnabled(False)
        self.clear_import_btn.clicked.connect(self.clear_imported_state)
        self.import_prompts_btn = QPushButton("📝 导入已有 Prompts")
        self.import_prompts_btn.setToolTip("输入已有的艺术风格 Prompts 文本作为训练起点，从 Round 1 开始迭代优化")
        self.import_prompts_btn.clicked.connect(self.import_prompts_text)
        import_layout.addWidget(self.import_json_btn)
        import_layout.addWidget(self.clear_import_btn)
        import_layout.addWidget(self.import_prompts_btn)
        import_layout.addStretch()
        self.import_status_label = QLabel("")
        self.import_status_label.setStyleSheet("color: #1b8f3a; font-weight: bold;")
        import_layout.addWidget(self.import_status_label)
        layout.addLayout(import_layout)

        # Action buttons
        action_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("开始多轮迭代画风提取")
        self.analyze_btn.setFixedHeight(40)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.cancel_btn = QPushButton("取消任务")
        self.cancel_btn.setFixedHeight(40)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.setFixedHeight(40)
        self.clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        action_layout.addWidget(self.analyze_btn)
        action_layout.addWidget(self.cancel_btn)
        action_layout.addWidget(self.clear_log_btn)
        layout.addLayout(action_layout)

        # Status
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        # Progress
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #0b63c7; font-weight: bold;")
        layout.addWidget(self.progress_label)

        # Log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("这里会显示任务日志...")
        self.log_text.setMaximumHeight(140)
        layout.addWidget(self.log_text)

        # Result area
        result_header = QHBoxLayout()
        result_header.addWidget(QLabel("当前艺术风格 Prompts:"))
        result_header.addStretch()
        self.open_output_dir_btn = QPushButton("打开输出目录")
        self.open_output_dir_btn.setEnabled(False)
        self.open_output_dir_btn.clicked.connect(self._open_output_dir)
        result_header.addWidget(self.open_output_dir_btn)
        layout.addLayout(result_header)

        self.result_edit = QTextEdit()
        self.result_edit.setPlaceholderText("迭代完成后，艺术风格 prompts 将显示在这里…")
        layout.addWidget(self.result_edit)

        # Output info
        self.output_path_label = QLabel("")
        self.output_path_label.setStyleSheet("color: gray; font-size: 11px;")
        self.output_path_label.setWordWrap(True)
        layout.addWidget(self.output_path_label)

        self.setLayout(layout)

    def _get_image_paths(self):
        paths = []
        for i in range(self.image_list.count()):
            path = self.image_list.item(i).data(Qt.ItemDataRole.UserRole)
            if path and os.path.isfile(path):
                paths.append(path)
        return paths

    def browse_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        for f in files:
            self.image_list.add_image_item(f)

    def clear_images(self):
        self.image_list.clear()
        self._loaded_image_paths = []

    def _on_test_gen_toggled(self, checked):
        if self.on_test_gen_changed:
            self.on_test_gen_changed(bool(checked))

    def _on_test_prompt_changed(self):
        if self.on_test_prompt_changed:
            self.on_test_prompt_changed(self.test_prompt_input.text())

    def set_test_gen_default(self, checked):
        self.enable_test_gen_cb.blockSignals(True)
        self.enable_test_gen_cb.setChecked(bool(checked))
        self.enable_test_gen_cb.blockSignals(False)

    def set_test_prompt_default(self, text):
        if text and not self.test_prompt_input.text():
            self.test_prompt_input.blockSignals(True)
            self.test_prompt_input.setText(str(text).strip())
            self.test_prompt_input.blockSignals(False)

    def import_existing_json(self):
        json_path, _ = QFileDialog.getOpenFileName(
            self, "选择已有的训练结果 JSON", "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not json_path:
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_state = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "导入失败", f"无法读取 JSON 文件:\n{e}")
            return

        if not isinstance(existing_state, dict):
            QMessageBox.warning(self, "格式错误", "JSON 文件格式不正确。")
            return

        iterations = existing_state.get("iterations", [])
        if not iterations:
            QMessageBox.warning(self, "数据为空", "JSON 文件中没有迭代记录。")
            return

        dataset_images = existing_state.get("dataset", {}).get("images", [])
        if dataset_images:
            existing_dataset = [p for p in dataset_images if os.path.isfile(p)]
            if not existing_dataset:
                QMessageBox.warning(
                    self, "图片路径失效",
                    "JSON 中记录的图片路径在当前环境下均不存在。\n"
                    "您仍需手动添加图片到下方列表，但迭代历史将被加载。"
                )
            else:
                self.image_list.set_image_paths(existing_dataset)
                self._loaded_image_paths = existing_dataset

        self._existing_state = existing_state
        self._loaded_json_path = json_path

        # 从文件名或状态中提取画风前缀，自动填入输入框
        basename = os.path.basename(json_path)
        suffix = "_style_iter_result.json"
        if basename.endswith(suffix) and len(basename) > len(suffix):
            filename_prefix = basename[:-len(suffix)]
        else:
            filename_prefix = ""

        # 若状态中尚未记录 file_prefix，自动补写
        state_prefix = existing_state.get("file_prefix", "")
        if not state_prefix and filename_prefix:
            existing_state["file_prefix"] = filename_prefix
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(existing_state, f, ensure_ascii=False, indent=2)
                self.log_msg(f"  已自动补写 file_prefix: \"{filename_prefix}\"")
            except Exception:
                pass

        # 自动填入前缀输入框
        prefix_to_fill = state_prefix or filename_prefix
        if prefix_to_fill:
            self.file_prefix_input.setText(prefix_to_fill)

        # Filter only numeric rounds for display
        numeric_rounds = [it.get("round", 1) for it in iterations
                          if isinstance(it.get("round", 1), (int, float))]
        last_round = int(max(numeric_rounds)) if numeric_rounds else 0
        total_steps = len(iterations)
        self.import_status_label.setText(
            f"已导入: {os.path.basename(json_path)} "
            f"(已完成 {last_round} 轮, {total_steps} 个步骤)"
        )
        self.clear_import_btn.setEnabled(True)
        self.log_msg(f"📂 已导入训练状态: {os.path.basename(json_path)}")
        self.log_msg(f"   已完成 {last_round} 轮, 共 {total_steps} 个步骤。")

        final_prompts = existing_state.get("final_art_style_prompts", "")
        if final_prompts:
            self.result_edit.setPlainText(final_prompts)

        if dataset_images:
            missing_images = [p for p in dataset_images if not os.path.isfile(p)]
            if missing_images:
                self.log_msg(f"   ⚠️ {len(missing_images)} 张图片路径已失效，请手动添加。")

    def clear_imported_state(self):
        self._existing_state = None
        self._loaded_json_path = ""
        self._loaded_image_paths = []
        self._imported_prompts = ""
        self.import_status_label.setText("")
        self.clear_import_btn.setEnabled(False)
        self.import_prompts_btn.setText("📝 导入已有 Prompts")
        self.import_status_label.setStyleSheet("color: #1b8f3a; font-weight: bold;")
        self.log_msg("已清除导入的训练状态。")

    def import_prompts_text(self):
        from PyQt6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getMultiLineText(
            self, "导入已有 Prompts",
            "请粘贴已有的艺术风格 Prompts 文本：\n\n这些 Prompts 将作为训练起点，从 Round 1 开始迭代优化。",
            self._imported_prompts if self._imported_prompts else ""
        )
        if not ok or not text.strip():
            return
        self._imported_prompts = text.strip()
        self.import_status_label.setText("已导入外部 Prompts (将从 Round 1 开始训练)")
        self.import_status_label.setStyleSheet("color: #c77d0f; font-weight: bold;")
        self.import_prompts_btn.setText("📝 已导入 Prompts ✏️")
        self.clear_import_btn.setEnabled(True)
        self.result_edit.setPlainText(self._imported_prompts)
        self.log_msg(f"📝 已导入外部 Prompts 作为训练起点 ({len(self._imported_prompts)} 字符)。")
        self.log_msg(f"   将从 Round 1 开始迭代优化。")

    def log_msg(self, text):
        append_log_line(self.log_text, text)

    def set_task_state(self, state, detail=""):
        set_task_status(self.status_label, state, detail)

    def set_running_state(self, running, cancelling=False):
        self.analyze_btn.setEnabled(not running)
        self.add_btn.setEnabled(not running)
        self.clear_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running and (not cancelling))
        self.import_json_btn.setEnabled(not running)
        self.clear_import_btn.setEnabled(not running and bool(self._existing_state or self._imported_prompts))
        self.import_prompts_btn.setEnabled(not running)
        self.total_rounds_spin.setEnabled(not running)
        self.images_per_round_spin.setEnabled(not running)
        self.enable_test_gen_cb.setEnabled(not running)
        self.test_prompt_input.setEnabled(not running)
        self.file_prefix_input.setEnabled(not running)

        if not running:
            self.progress_label.setText("")

    def start_analysis(self):
        image_paths = self._get_image_paths()

        missing_prompt_files = get_style_analyzer_missing_prompt_files()
        if missing_prompt_files:
            missing_text = "\n".join(missing_prompt_files)
            QMessageBox.warning(self, "缺少 Prompt 文件", f"以下 Prompt 文件不存在，请补齐后再执行：\n{missing_text}")
            self.log_msg(f"❌ 缺少 Prompt 文件，已中止：\n{missing_text}")
            return

        if not image_paths and not self._loaded_image_paths:
            QMessageBox.information(self, "提示", "请至少添加一张图片。")
            return

        # Use loaded paths if image list is empty but we have loaded state
        effective_paths = image_paths if image_paths else self._loaded_image_paths

        total_rounds = self.total_rounds_spin.value()
        images_per_round = self.images_per_round_spin.value()

        # Validate file prefix
        file_prefix = self.file_prefix_input.text().strip()
        if not file_prefix:
            QMessageBox.warning(self, "缺少画风前缀", "请填写「画风前缀」字段（必填）。\n\n该前缀用于区分不同画风的 JSON 文件名，例如 cyberpunk、watercolor。")
            return

        # Basic sanitization: reject suspicious characters
        invalid_chars = set(r'\/:*?"<>|')
        if any(c in file_prefix for c in invalid_chars):
            QMessageBox.warning(self, "无效前缀", f"画风前缀不能包含以下字符：\\ / : * ? \" < > |")
            return

        if len(effective_paths) < images_per_round:
            reply = QMessageBox.warning(
                self, "图片数量不足",
                f"数据集仅有 {len(effective_paths)} 张图片，但每轮需要检查 {images_per_round} 张。\n\n"
                f"请添加更多图片或将「每轮检查图片数」调整为 ≤ {len(effective_paths)}。",
                QMessageBox.StandardButton.Ok
            )
            return

        base_url, api_key, model_name = self.get_config()

        if not api_key or not model_name:
            QMessageBox.warning(self, "缺少配置", "请确保在【全局配置】中配置了文本分析的 API Key 和模型！")
            return

        # Test generation validation
        enable_test_gen = self.enable_test_gen_cb.isChecked()
        test_prompt = self.test_prompt_input.text().strip()
        if enable_test_gen and not test_prompt:
            QMessageBox.warning(self, "缺少测试提示词", "已勾选「每轮结束后生成测试图片」，但未输入测试提示词。\n请在下方输入框填写测试提示词，或取消勾选测试生图。")
            return

        # Image generation config for test images
        img_api_type = ""
        img_instructions = ""
        img_aspect_ratio = "1:1"
        if enable_test_gen:
            if self.get_img_config:
                img_cfg = self.get_img_config()
                if isinstance(img_cfg, (tuple, list)) and len(img_cfg) >= 4:
                    # (base_url, api_key, model_name, api_type)
                    img_api_type = str(img_cfg[3] or "").strip()
            if self.get_styles:
                styles_data = self.get_styles()
                # Use the first style's instructions or empty string
                if styles_data:
                    first_entry = next(iter(styles_data.values()), "")
                    img_instructions = str(normalize_style_entry(first_entry)["prompt"] or "").strip()

        timeout_seconds = int(self.get_timeout()) if self.get_timeout else 120

        self.set_running_state(True)
        self.set_task_state("running", f"准备迭代 {total_rounds} 轮")
        self.log_msg(f"\n{'=' * 60}")
        self.log_msg(f"开始多轮迭代画风提取：共 {total_rounds} 轮，每轮检查 {images_per_round} 张图片。")
        self.log_msg(f"数据集: {len(effective_paths)} 张图片。")
        self.result_edit.clear()
        self.progress_label.setText("准备中…")

        self.thread = StyleIterativeWorkerThread(
            image_paths=effective_paths,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            total_rounds=total_rounds,
            images_per_round=images_per_round,
            existing_state=self._existing_state,
            output_dir=self._output_dir,
            timeout_seconds=timeout_seconds,
            enable_test_gen=enable_test_gen,
            test_prompt=test_prompt,
            img_api_type=img_api_type,
            img_instructions=img_instructions,
            img_aspect_ratio=img_aspect_ratio,
            file_prefix=file_prefix,
            seed_prompts=self._imported_prompts if not self._existing_state else "",
        )
        self.thread.log_signal.connect(self.log_msg)
        self.thread.progress_signal.connect(self.progress_label.setText)
        self.thread.finish_signal.connect(self.on_analysis_finished)
        self.thread.save_signal.connect(self._on_save_signal)
        self.thread.start()

    def cancel_analysis(self):
        if self.thread is None:
            self.log_msg("当前没有正在运行的任务。")
            return
        self.thread.request_cancel()
        self.set_running_state(True, cancelling=True)
        self.set_task_state("cancelling", "等待当前步骤结束")
        self.log_msg("已请求取消任务，等待当前步骤结束...")

    def _on_save_signal(self, output_path, result_text):
        self.output_path_label.setText(f"最新保存: {output_path}")
        self.open_output_dir_btn.setEnabled(True)
        if result_text:
            self.result_edit.setPlainText(result_text)

    def on_analysis_finished(self, status, text, output_path):
        thread = self.thread
        self.thread = None
        self.set_running_state(False)

        if thread is not None:
            thread.deleteLater()

        self._output_dir = os.path.dirname(output_path) if output_path else ""
        self.output_path_label.setText(f"输出文件: {output_path}" if output_path else "")
        self.open_output_dir_btn.setEnabled(bool(output_path))

        if status == "success" and text:
            self.result_edit.setPlainText(text)
            self.set_task_state("success", "多轮迭代完成")
            self.log_msg("多轮迭代画风提取完成。")
        elif status == "cancelled":
            self.set_task_state("cancelled", "任务已取消")
            if text:
                self.result_edit.setPlainText(text)
        else:
            self.set_task_state("error", "请查看任务日志")
            if text:
                self.result_edit.setPlainText(text)
            if status != "cancelled":
                QMessageBox.warning(self, "错误", "分析过程中发生错误，请查看任务日志。")

    def _open_output_dir(self):
        import subprocess
        target_dir = self._output_dir
        if not target_dir or not os.path.isdir(target_dir):
            if self.output_path_label.text():
                candidate = self.output_path_label.text().replace("输出文件: ", "").replace("最新保存: ", "").strip()
                if candidate:
                    target_dir = os.path.dirname(candidate)
        if target_dir and os.path.isdir(target_dir):
            try:
                os.startfile(target_dir)
            except Exception:
                subprocess.run(["explorer", target_dir], shell=True)
        else:
            QMessageBox.information(self, "提示", "输出目录尚不存在。")
