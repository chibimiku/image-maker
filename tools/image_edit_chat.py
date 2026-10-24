"""Run the image-edit tab's conversational pipeline without Qt."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.image_generation import image_edit_dialogue as rt
from modules.others.api_backend import get_api_config
from utils.analysis_gpt_prompt import load_text_api_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Persistent conversation JSON")
    parser.add_argument("--images", nargs="+", help="Local images in this turn's allowed scope")
    parser.add_argument("--instruction", help="Natural language instruction")
    parser.add_argument("--model", help="Override the configured Gemini image model")
    parser.add_argument("--resolution", choices=["1K", "2K", "4K"])
    parser.add_argument("--timeout", type=int, help="Gemini request timeout in seconds")
    parser.add_argument("--output-dir", help="Explicit output folder, e.g. an isolated test-result directory")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly retry only the last turn's failed items")
    parser.add_argument("--dry-run", action="store_true", help="Show scope/configuration with no requests or writes")
    args = parser.parse_args()
    session = rt.load_session(args.session)
    ids = rt.register_images(session, args.images or [])
    for index, image_id in enumerate(ids, 1):
        rt.image_by_id(session, image_id)["display_index"] = index
    text, image = load_text_api_config(), get_api_config(api_type="aigc2d")
    if args.model:
        image["model"] = args.model
    if args.resolution:
        image["resolution"] = args.resolution
    if args.timeout:
        if not 30 <= args.timeout <= 900:
            parser.error("--timeout must be between 30 and 900")
        image["timeout"] = args.timeout
    if args.output_dir:
        image["output_dir"] = str(Path(args.output_dir).resolve())
    retry_id, user = None, args.instruction or ""
    if args.retry_failed:
        if not session["turns"]:
            parser.error("No previous turn")
        previous = session["turns"][-1]
        retry_id, user = previous["id"], previous["user"]
        ids = [t["image_id"] for t in previous["tasks"] if t["status"] in {"error", "interrupted", "cancelled"}]
    if not user or not ids:
        parser.error("Provide --instruction and --images, or --retry-failed")
    if args.dry_run:
        print(json.dumps({"image_ids": ids, "image_model": image.get("model"), "text_model": text.get("model"),
                          "resolution": image.get("resolution"), "instruction": user, "requests": 0}, ensure_ascii=False))
        return 0
    def event(value):
        turn = value["session"]["turns"][-1]
        print(json.dumps({"status": turn["status"], "message": turn["message"],
                          "tasks": [{"id": t["image_id"], "status": t["status"]} for t in turn["tasks"]]}, ensure_ascii=False), flush=True)
    runner = rt.DialogueRunner(session, args.session, text, image, event)
    runner.run(user, ids, retry_turn_id=retry_id)
    turn = runner.session["turns"][-1]
    return 1 if turn["status"] == "error" or any(t["status"] == "error" for t in turn["tasks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
