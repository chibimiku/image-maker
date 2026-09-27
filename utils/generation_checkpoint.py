"""Persistent, artifact-validated checkpoints for analysis image generation."""
import json
import os
import time

ANATOMY_GATE_VERSION = 3


STAGE_LABELS = {
    "first": "GPT 首图", "pipeline": "Gemini 重绘与本地工序",
    "face": "五官画风修订", "quality": "质量审计与修订",
    "identity": "身份审计与修订", "hands": "逐角色人体结构审计与修订",
    "adjust": "画风收尾", "final_review": "最终结构与画风复核", "publish": "最终发布",
}


class GenerationCheckpoint:
    def __init__(self, path, snapshot=None):
        self.path = os.path.abspath(path)
        self.data = {"version": 1, "snapshot": snapshot or {}, "stages": {}}
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as stream:
                self.data = json.load(stream)
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        temp = self.path + ".tmp"
        with open(temp, "w", encoding="utf-8") as stream:
            json.dump(self.data, stream, ensure_ascii=False, indent=2)
        os.replace(temp, self.path)

    def begin(self, stage, inputs):
        entry = self.data["stages"].get(stage, {})
        outputs = entry.get("outputs") or []
        if entry.get("status") == "success" and outputs and all(os.path.isfile(p) for p in outputs):
            return False, list(outputs)
        if entry.get("status") == "success":
            self.reset_from(stage)
        self.data["current_stage"] = stage
        self.data["status"] = "running"
        self.data["stages"][stage] = {"status": "running", "inputs": list(inputs)}
        self.save()
        return True, list(inputs)

    def complete(self, stage, outputs):
        self.data["stages"][stage].update(status="success", outputs=list(outputs))
        self.data["last_outputs"] = list(outputs)
        self.save()

    def fail(self, error, cancelled=False):
        stage = self.data.get("current_stage", "first")
        self.data["status"] = "cancelled" if cancelled else "error"
        self.data["error"] = str(error)
        entry = self.data["stages"].setdefault(stage, {})
        if entry.get("status") != "success":
            entry.update(status=self.data["status"], error=str(error))
        self.save()

    def reset_from(self, stage):
        if stage == "first":
            self.data.pop("first_safe_review", None)
        keys = list(STAGE_LABELS)
        for key in keys[keys.index(stage):]:
            self.data["stages"].pop(key, None)
            for operation in list(self.data.get("operations", {})):
                if operation.startswith(key + "-"):
                    self.data["operations"].pop(operation)
        self.save()

    def operation(self, key, call, image=False):
        """Keep successful network calls even if the subsequent audit fails."""
        operations = self.data.setdefault("operations", {})
        if key in operations:
            result = operations[key]
            if not image or (result and all(os.path.isfile(p) for p in result)):
                return result
        result = call()
        if image and (not result or not all(os.path.isfile(p) for p in result)):
            raise RuntimeError(key + " 未返回有效图片")
        operations[key] = result
        if image:
            self.data["last_outputs"] = list(result)
        self.save()
        return result

def import_generation_checkpoint(path, styles=None, analysis_result=None):
    """Read new checkpoints or migrate an older first-image request manifest."""
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("snapshot") and isinstance(data.get("stages"), dict):
        return os.path.abspath(path), data
    first = str(data.get("first_image") or "")
    if not first or not os.path.isfile(first):
        raise ValueError("不是有效工序断点，或旧请求的 GPT 首图已不存在")
    from utils.analysis_gen import build_first_pass_request
    result = dict(analysis_result or {})
    request = build_first_pass_request(styles or {}, data.get("style_name", ""), result,
                                       content_text=data.get("prompt", ""), tier="short")
    request["prompt"] = data.get("prompt", "")
    request["image_paths"] = [r["path"] for r in data.get("references", []) if r.get("path")]
    ref = request.get("style_ref_path") or next(iter(request["image_paths"]), "")
    snapshot = dict(request_payload=request, steps=data.get("steps") or {},
        firmware=data.get("firmware_text", ""), size=data.get("size", "1024x1536"),
        quality=data.get("quality", "high"), output_format=os.path.splitext(first)[1].lstrip("."),
        file_prefix=os.path.basename(first).split("_")[0], model_name=data.get("model", "gpt-image-2"),
        api_type="aigc-2d-gpt", mode=data.get("requested_mode", "generate"),
        style_ref_path=ref, style_clauses=list(request.get("clauses") or []) + list(request.get("proportion_clauses") or []),
        analysis_result=result)
    target = os.path.join(os.path.dirname(os.path.abspath(first)), "generation-checkpoint-imported.json")
    checkpoint = GenerationCheckpoint(target, snapshot)
    if not checkpoint.data["stages"]:
        checkpoint.begin("first", [])
        checkpoint.complete("first", [first])
        checkpoint.data.update(status="error", current_stage="pipeline", imported_legacy=True)
        existing = [p for p in data.get("outputs", []) if os.path.isfile(p)]
        if existing:
            checkpoint.data["last_outputs"] = existing
        checkpoint.save()
    return target, checkpoint.data
