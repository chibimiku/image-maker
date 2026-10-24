"""Durable, serial conversational Gemini editing, independent of Qt widgets."""
import base64
import copy
import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from openai import OpenAI

from modules.others.api_backend import generate_image_aigc2d
from utils.analysis_gpt_prompt import normalize_chat_base
from utils.analysis_fallback import is_refusal_text
from utils.output_isolation import resolve_output_target, is_test_context
from utils.prompt_loader import read_prompt_file

MAX_IMAGES = 24
IN_FLIGHT = {"routing", "planning", "generating"}


def new_session():
    return {"schema": 1, "id": uuid.uuid4().hex, "images": [], "turns": []}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_session(session, path):
    """Write before dispatch; never include credentials or base64 attachments."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def load_session(path):
    if not Path(path).exists():
        return new_session()
    session = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(session, dict) or session.get("schema") != 1 or not isinstance(session.get("images"), list) or not isinstance(session.get("turns"), list):
        raise ValueError("不支持的对话记录格式")
    if not isinstance(session.get("id"), str) or not re.fullmatch(r"[a-f0-9]{32}", session["id"]):
        raise ValueError("对话 ID 无效")
    for index, row in enumerate(session["images"], 1):
        if (not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] != index
                or not isinstance(row.get("original"), str) or not isinstance(row.get("versions"), list)
                or not row["versions"]):
            raise ValueError("图片记录无效")
        for version in row["versions"]:
            if (not isinstance(version, dict) or not isinstance(version.get("path"), str)
                    or not re.fullmatch(r"[a-f0-9]{64}", str(version.get("sha256", "")))):
                raise ValueError("图片版本记录无效")
    for turn in session["turns"]:
        if (not isinstance(turn, dict) or not isinstance(turn.get("id"), str)
                or not isinstance(turn.get("user"), str) or not isinstance(turn.get("tasks"), list)):
            raise ValueError("对话轮次无效")
        for task in turn["tasks"]:
            if (not isinstance(task, dict) or type(task.get("image_id")) is not int
                    or task["image_id"] not in range(1, len(session["images"]) + 1)):
                raise ValueError("任务图片 ID 无效")
            row = image_by_id(session, task["image_id"])
            resolve_version(row, task.get("version"))
            if (not isinstance(task.get("instruction"), str) or not isinstance(task.get("input"), dict)
                    or not isinstance(task["input"].get("path"), str)
                    or not re.fullmatch(r"[a-f0-9]{64}", str(task["input"].get("sha256", "")))
                    or not isinstance(task.get("status"), str) or not isinstance(task.get("attempts"), list)):
                raise ValueError("任务记录无效")
        if turn.get("status") in IN_FLIGHT | {"queued", "running"}:
            turn["status"] = "interrupted"
            turn["message"] = "上次运行已中断；不会自动重发，请查看记录后手动重试。"
        for task in turn.get("tasks", []):
            if task.get("status") in IN_FLIGHT | {"queued"}:
                task["status"] = "interrupted"
            if task.get("review_status") in {"pending", "reviewing"}:
                task["review_status"] = "interrupted"
    return session


def register_images(session, paths):
    by_path = {os.path.normcase(row["original"]): row for row in session["images"]}
    ids = []
    for path in paths:
        absolute = os.path.abspath(path)
        key = os.path.normcase(absolute)
        if key not in by_path or sha256(absolute) != by_path[key]["versions"][0]["sha256"]:
            with Image.open(absolute) as image:
                image.verify()
            row = {"id": len(session["images"]) + 1, "original": absolute,
                   "versions": [{"path": absolute, "sha256": sha256(absolute), "turn_id": None}]}
            session["images"].append(row)
            by_path[key] = row
        ids.append(by_path[key]["id"])
    return ids


def image_by_id(session, image_id):
    return next(row for row in session["images"] if row["id"] == image_id)


def resolve_version(row, version):
    if version == "latest":
        version = len(row["versions"]) - 1
    if type(version) is not int or not 0 <= version < len(row["versions"]):
        raise ValueError(f"图片 #{row['id']} 的版本不存在")
    return version, row["versions"][version]


def checked_source(version):
    path = version["path"]
    if not os.path.isfile(path) or sha256(path) != version["sha256"]:
        raise ValueError("输入图片已缺失或被修改，已停止该项；请重新添加图片")
    return path


def preview_url(path, edge=1024):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((edge, edge))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def request_json(stage, context, attachments, config):
    if not all(config.get(key) for key in ("base_url", "api_key", "model")):
        raise ValueError("请配置支持看图的文字分析 API")
    content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}]
    for label, path, edge in attachments:
        content.extend([{"type": "text", "text": label},
                        {"type": "image_url", "image_url": {"url": preview_url(path, edge)}}])
    # No SDK retry or refusal fallback: one intentional operation per dispatch.
    with OpenAI(base_url=normalize_chat_base(config["base_url"]), api_key=config["api_key"],
                timeout=120, max_retries=0) as client:
        response = client.chat.completions.create(
            model=config["model"], max_completion_tokens=3500,
            messages=[{"role": "system", "content": read_prompt_file(f"image-edit-dialogue/{stage}.md")},
                      {"role": "user", "content": content}])
    choices = list(response.choices or [])
    if not choices or choices[0].finish_reason in {"length", "content_filter"} or getattr(choices[0].message, "refusal", None):
        raise ValueError("分析响应被截断或拒绝，未发送图片编辑请求")
    text = str(choices[0].message.content or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise ValueError("分析响应不是有效 JSON，未发送图片编辑请求") from exc
    if not isinstance(value, dict):
        raise ValueError("分析响应格式错误")
    value["analysis_usage"] = response.usage.model_dump() if response.usage else None
    return value


def validate_route(value, session, allowed_ids):
    if value.get("action") not in {"edit", "reply", "clarify"} or not isinstance(value.get("message"), str):
        raise ValueError("意图分析缺少 action/message")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or len(tasks) > MAX_IMAGES:
        raise ValueError("意图分析任务列表无效")
    if value["action"] != "edit":
        if tasks:
            raise ValueError("讨论或澄清不能附带编辑任务")
        return []
    if not tasks:
        raise ValueError("意图分析未选定图片")
    seen, validated = set(), []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("编辑任务格式无效")
        image_id = task.get("image_id")
        if type(image_id) is not int or image_id not in allowed_ids or image_id in seen:
            raise ValueError("意图分析选择了范围外或重复的图片")
        seen.add(image_id)
        instruction = task.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 6000:
            raise ValueError("编辑要求为空或过长")
        index, version = resolve_version(image_by_id(session, image_id), task.get("version"))
        validated.append({"image_id": image_id, "version": index, "instruction": instruction,
                          "input": copy.deepcopy(version), "status": "queued", "attempts": []})
    return validated


def validate_plan(value):
    if value.get("decision") not in {"edit", "skip", "clarify"} or not isinstance(value.get("summary"), str):
        raise ValueError("逐图分析缺少 decision/summary")
    if type(value.get("sexualized")) is not bool or value.get("age_assessment") not in {"adult", "minor", "uncertain", "not_applicable"}:
        raise ValueError("逐图分析缺少有效的适用性字段")
    if value["sexualized"] and value["age_assessment"] != "adult":
        value = {**value, "decision": "skip", "prompt": "", "summary": "角色成年状态不明确，跳过性感化编辑。"}
    if value["decision"] == "edit" and (not isinstance(value.get("prompt"), str) or not value["prompt"].strip() or len(value["prompt"]) > 4000):
        raise ValueError("逐图编辑指令为空或超过 4000 字符")
    if value["decision"] == "edit" and is_refusal_text(value["prompt"]):
        raise ValueError("逐图分析返回拒绝话术，未发送图片编辑请求")
    return value


def history_context(session):
    return [{"user": t["user"][:6000], "message": t.get("message", "")[:2000],
             "tasks": [{k: task.get(k) for k in ("image_id", "version", "instruction", "status", "summary")}
                       for task in t.get("tasks", [])]}
            for t in session["turns"][-12:]]


def review_result(task, output_path, text_config):
    value = request_json("review", {"instruction": task["instruction"], "edit_prompt": task["plan"]["prompt"]},
                         [("INPUT before editing", checked_source(task["input"]), 1536),
                          ("EDITED RESULT to inspect", output_path, 1536)], text_config)
    if (value.get("verdict") not in {"pass", "uncertain", "fail"}
            or not isinstance(value.get("summary"), str) or not value["summary"].strip()
            or not isinstance(value.get("issues"), list)
            or any(not isinstance(issue, str) for issue in value["issues"])):
        raise ValueError("出图复核响应格式无效")
    return value


class DialogueRunner:
    def __init__(self, session, path, text_config, image_config, emit=lambda event: None, cancelled=lambda: False):
        self.session, self.path = copy.deepcopy(session), path
        self.text_config, self.image_config = dict(text_config), dict(image_config)
        self.emit, self.cancelled = emit, cancelled

    def checkpoint(self):
        save_session(self.session, self.path)
        self.emit({"type": "session", "session": copy.deepcopy(self.session)})

    def run(self, user, allowed_ids, retry_turn_id=None, preferred_versions=None):
        if not user.strip() or len(user) > 12000:
            raise ValueError("请输入 1–12000 字符的编辑要求")
        if not allowed_ids or len(allowed_ids) > MAX_IMAGES:
            raise ValueError(f"请将本轮图片范围限定在 1–{MAX_IMAGES} 张")
        # Validate the node before planning, including when the global channel is GPT.
        if not self.image_config.get("api_key") or not str(self.image_config.get("model", "")).lower().startswith("gemini"):
            raise ValueError("请在对话编辑中选择已配置的 Gemini 模型和 aigc2d 节点")
        history = history_context(self.session)
        turn = {"id": uuid.uuid4().hex, "user": user, "time": datetime.now().isoformat(),
                "status": "routing", "message": "正在理解要求和选择图片…", "tasks": [],
                "models": {"analysis": self.text_config.get("model"), "edit": self.image_config.get("model"),
                           "resolution": self.image_config.get("resolution"), "retry_limit": 0,
                           "timeout": self.image_config.get("timeout", 360)},
                "allowed_ids": list(allowed_ids)}
        self.session["turns"].append(turn)
        self.checkpoint()
        try:
            if retry_turn_id:
                previous = next(t for t in self.session["turns"] if t["id"] == retry_turn_id)
                turn["retry_of"] = retry_turn_id
                turn["tasks"] = copy.deepcopy([task for task in previous["tasks"]
                                               if task["status"] in {"error", "interrupted", "cancelled"}])
                for task in turn["tasks"]:
                    if task["image_id"] not in allowed_ids:
                        raise ValueError("重试范围与原任务不一致")
                    task.update(status="queued", attempts=[])
                    task.pop("error", None)
                if not turn["tasks"]:
                    raise ValueError("该轮没有可重试的失败项")
                turn["message"] = "按原要求、原输入重试失败项；成功项不重复生成。"
            else:
                inventory, attachments = [], []
                for image_id in allowed_ids:
                    row = image_by_id(self.session, image_id)
                    index, source = resolve_version(row, "latest")
                    inventory.append({"image_id": image_id, "filename": os.path.basename(row["original"]),
                                      "latest_version": index, "versions": list(range(len(row["versions"]))),
                                      "display_index": row.get("display_index", image_id)})
                    attachments.append((f"Image #{image_id}, version {index}", checked_source(source), 320))
                if self.cancelled():
                    turn.update(status="cancelled", message="已停止，未发送编辑请求。")
                    return
                route = request_json("route", {"user": user, "allowed_ids": allowed_ids,
                                              "images": inventory, "history": history,
                                              "ui_input_versions": preferred_versions or {}}, attachments, self.text_config)
                turn["tasks"] = validate_route(route, self.session, allowed_ids)
                turn["route"] = route
                for task in turn["tasks"]:
                    if task["image_id"] in (preferred_versions or {}):
                        index, source = resolve_version(image_by_id(self.session, task["image_id"]), preferred_versions[task["image_id"]])
                        task.update(version=index, input=copy.deepcopy(source))
                turn["message"] = route["message"]
                turn["action"] = route["action"]
                if route["action"] != "edit":
                    turn["status"] = route["action"]
                    return
            turn["status"] = "running"
            self.checkpoint()
            for task in turn["tasks"]:
                if self.cancelled():
                    task["status"] = "cancelled"
                    self.checkpoint()
                    continue
                try:
                    source = checked_source(task["input"])
                    task["status"] = "planning"
                    self.checkpoint()
                    if not task.get("plan"):
                        task["plan"] = validate_plan(request_json("plan", {
                            "user": user, "instruction": task["instruction"], "image_id": task["image_id"],
                            "input_version": task["version"], "history": history},
                            [(f"Image #{task['image_id']} to edit", source, 1536)], self.text_config))
                    plan = validate_plan(task["plan"])
                    task["summary"] = plan["summary"]
                    if plan["decision"] != "edit":
                        task["status"] = plan["decision"]
                        self.checkpoint()
                        continue
                    if self.cancelled():
                        task["status"] = "cancelled"
                        self.checkpoint()
                        continue
                    checked_source(task["input"])
                    task["status"] = "generating"
                    task["attempts"].append({"dispatched_at": datetime.now().isoformat()})
                    self.checkpoint()
                    metadata = generate_image_aigc2d(
                        prompt=plan["prompt"], image_paths=[source], model=self.image_config["model"],
                        api_type="aigc2d", aspect_ratio="auto", resolution=self.image_config.get("resolution"),
                        face_quality_boost=False, retry_limit=0, request_timeout=self.image_config.get("timeout", 360),
                        save_sub_dir=self.image_config.get("output_dir") or "image-edit-dialogue",
                        file_prefix=f"chat-{turn['id'][:8]}-i{task['image_id']}",
                        return_metadata=True, config_snapshot=self.image_config)
                    # A sent request is collected even if Stop is clicked, so a returned paid result is retained.
                    files = metadata.get("saved_files", []) if isinstance(metadata, dict) else []
                    raw = metadata.get("server_response_raw", {}) if isinstance(metadata, dict) else {}
                    task["attempts"][-1].update(returned_at=datetime.now().isoformat(),
                                                usage=raw.get("usageMetadata"), model_version=raw.get("modelVersion"))
                    if not files:
                        raise ValueError("Gemini 未返回图片；可能被拒绝、超时或请求失败。未自动重试，费用以服务商账单为准。")
                    versions = image_by_id(self.session, task["image_id"])["versions"]
                    output_versions, additions = [], []
                    for file in files:
                        absolute = os.path.abspath(file)
                        if os.path.normcase(absolute) == os.path.normcase(source):
                            raise ValueError("接口回退到输入原图，不能作为新产物")
                        with Image.open(absolute) as image:
                            image.verify()
                        additions.append({"path": absolute, "sha256": sha256(absolute), "turn_id": turn["id"],
                                          "parent_version": task["version"], "prompt": plan["prompt"]})
                        output_versions.append(len(versions) + len(additions) - 1)
                    versions.extend(additions)
                    task.update(status="success", output_versions=output_versions)
                    # Persist returned paid outputs BEFORE review. An interrupted review must
                    # never turn a received image into a retryable generation failure.
                    task["review_status"] = "pending"
                    self.checkpoint()
                    task["reviews"] = []
                    for output_index in output_versions:
                        if self.cancelled():
                            break
                        task["review_status"] = "reviewing"
                        self.checkpoint()
                        try:
                            review = review_result(task, versions[output_index]["path"], self.text_config)
                        except Exception as exc:
                            review = {"verdict": "uncertain", "summary": f"图片已保存，出图复核未完成：{str(exc)[:500]}", "issues": []}
                        task["reviews"].append(review)
                        versions[output_index]["review"] = review
                        self.checkpoint()
                    task["review_status"] = "completed" if len(task["reviews"]) == len(output_versions) else "cancelled"
                    if task["reviews"]:
                        task["summary"] += "\n复核：" + "；".join(review["summary"] for review in task["reviews"])
                except Exception as exc:
                    task.update(status="success" if task.get("output_versions") else "error", error=str(exc)[:2000])
                self.checkpoint()
            counts = {status: sum(t["status"] == status for t in turn["tasks"])
                      for status in ("success", "error", "skip", "clarify", "cancelled")}
            flagged = sum(task["status"] == "success" and (
                not task.get("reviews") or task.get("review_status") != "completed"
                or any(review["verdict"] != "pass" for review in task["reviews"])) for task in turn["tasks"])
            review_message = (f"其中 {flagged} 项复核异常或未完成，需查看。" if flagged else
                              "结果复核已完成。" if counts["success"] else "")
            turn.update(status="cancelled" if self.cancelled() else "completed",
                        message=(f"本轮返回图片 {counts['success']} 项，失败 {counts['error']} 项，"
                                 f"跳过 {counts['skip']} 项，需澄清 {counts['clarify']} 项，停止 {counts['cancelled']} 项。"
                                 f"{review_message}结果和复核意见已保留，可选择版本继续修改；不会自动重画。"))
        except Exception as exc:
            turn.update(status="error", message=str(exc)[:2000])
        finally:
            self.checkpoint()
        return self.session


def default_session_path():
    if not is_test_context():
        return os.path.abspath("data/image-edit-dialogue/current.json")
    directory, filename = resolve_output_target("data/" + datetime.now().strftime("%Y%m%d") + "/image-edit-dialogue", "current.json")
    return os.path.abspath(os.path.join(directory, filename))
