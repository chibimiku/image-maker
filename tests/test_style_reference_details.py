import copy
from pathlib import Path

import pytest
from PIL import Image

from utils.style_reference import (append_gpt_style_details, detail_reference_paths,
                                   file_hash, style_face_quality_boost)
from utils.styles import build_ref_gen_params, build_content_style_image_roles


def entry(tmp_path):
    path = tmp_path / "reference.png"
    Image.new("RGB", (200, 300), "purple").save(path)
    return {"prompt": "paint", "prompt_gpt": "short paint", "ref_image": str(path),
            "reference_details_enabled": True, "reference_detail_crops": [
                {"role": "face_eyes", "box": [0.1, 0.1, 0.8, 0.5],
                 "source_sha256": file_hash(path)}]}


def test_crop_is_exact_and_repairs_corrupt_cache(tmp_path):
    data = entry(tmp_path)
    refs = detail_reference_paths(data, data["ref_image"], tmp_path / "cache")
    path = Path(refs[0]["path"])
    with Image.open(path) as crop:
        assert crop.size == (140, 120)
        assert crop.getpixel((0, 0)) == (128, 0, 128)
    expected = path.read_bytes()
    path.write_bytes(b"corrupt")
    detail_reference_paths(data, data["ref_image"], tmp_path / "cache")
    assert path.read_bytes() == expected


@pytest.mark.parametrize("box", [[-0.1, 0, 1, 1], [0, 0, 0, 1], [0, 0, float("nan"), 1],
                                  [0, 0, True, 1], [0, 0, 0.01, 0.01], [0, 0, 1]])
def test_invalid_region_rejected_before_sending(tmp_path, box):
    data = entry(tmp_path)
    data["reference_detail_crops"][0]["box"] = box
    with pytest.raises(ValueError):
        detail_reference_paths(data, data["ref_image"], tmp_path / "cache")


def test_stale_hash_and_too_many_crops_rejected(tmp_path):
    data = entry(tmp_path)
    data["reference_detail_crops"][0]["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="哈希"):
        detail_reference_paths(data, data["ref_image"])
    data["reference_detail_crops"] *= 3
    with pytest.raises(ValueError):
        detail_reference_paths(data, data["ref_image"])


def test_disabled_and_off_modes_do_not_load_crops(tmp_path):
    data = entry(tmp_path)
    data["reference_detail_crops"][0]["source_sha256"] = "stale"
    assert build_ref_gen_params({"x": data}, "x", "off")[2] == []
    data["reference_details_enabled"] = False
    assert detail_reference_paths(data, "missing") == []


def test_gemini_full_reference_stays_primary(tmp_path):
    data = entry(tmp_path)
    before = copy.deepcopy(data)
    head, post, refs = build_ref_gen_params({"x": data}, "x", "priority", "aigc2d")
    assert len(refs) == 2 and refs[0] == data["ref_image"]
    assert "Image 1: full style reference" in post
    assert "Image 2: face_eyes" in post
    assert "immediately above" not in post
    assert data == before


def test_gpt_content_numbers_and_last_full_reference(tmp_path):
    data = entry(tmp_path)
    prompt, paths = append_gpt_style_details(
        build_content_style_image_roles(1), ["content.png", data["ref_image"]],
        data, data["ref_image"], 1)
    assert paths[0] == "content.png" and paths[-1] == data["ref_image"]
    assert "Image 2 (the LAST image)" not in prompt
    assert "Image 1: content reference" in prompt
    assert "Image 2: face_eyes" in prompt and "Image 3: full style" in prompt
    with pytest.raises(ValueError, match="16"):
        append_gpt_style_details("", ["content"] * 15 + [data["ref_image"]], data, data["ref_image"], 15)


def test_shared_gpt_first_pass_includes_detail(tmp_path):
    from utils.analysis_gen import build_first_pass_request
    data = entry(tmp_path)
    request = build_first_pass_request({"x": data}, "x", {}, content_text="One adult woman.")
    assert len(request["image_paths"]) == 2
    assert request["image_paths"][-1] == data["ref_image"]
    assert "Image 1: face_eyes" in request["prompt"]


def test_face_boost_reference_default_and_explicit_override(tmp_path):
    data = entry(tmp_path)
    assert not style_face_quality_boost({"x": data}, "x", "priority", "aigc2d")
    assert style_face_quality_boost({"x": data}, "x", "off", "aigc2d")
    data["face_quality_boost"] = True
    assert style_face_quality_boost({"x": data}, "x", "priority", "aigc2d")
    data["gemini_priority_profile"] = {"head_file": "h", "post_file": "p", "face_quality_boost": False}
    assert not style_face_quality_boost({"x": data}, "x", "priority", "aigc2d")


def test_manual_crop_dialog_does_not_infer_regions(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from utils.style_reference_widget import ReferenceCropDialog
    app = QApplication.instance() or QApplication([])
    data = entry(tmp_path)
    dialog = ReferenceCropDialog(data["ref_image"])
    with pytest.raises(ValueError):
        dialog.specification()
    dialog.box = [0.1, 0.1, 0.8, 0.5]
    assert dialog.specification() == data["reference_detail_crops"][0]
    dialog.close()


def test_exif_orientation_matches_transmitted_full_and_crop(tmp_path):
    from modules.others.api_backend import _encode_image_for_transmission
    import io
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (100, 200), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    data = {"reference_details_enabled": True, "reference_detail_crops": [
        {"role": "face_eyes", "box": [0, 0, .5, 1], "source_sha256": file_hash(path)}]}
    refs = detail_reference_paths(data, path, tmp_path / "cache")
    with Image.open(refs[0]["path"]) as crop:
        assert crop.size == (100, 100)
    mime, encoded = _encode_image_for_transmission(path)
    with Image.open(io.BytesIO(encoded)) as transmitted:
        assert transmitted.size == (200, 100)


def test_manifest_distinguishes_original_and_transmitted_bytes(tmp_path):
    from utils.style_reference import reference_input_manifest
    data = entry(tmp_path)
    refs = detail_reference_paths(data, data["ref_image"], tmp_path / "cache")
    result = reference_input_manifest([refs[0]["path"], data["ref_image"]], encoding="gemini")
    assert [r["role"] for r in result] == ["style_detail", "style_full"]
    assert result[0]["file_sha256"] != result[0]["transmitted_sha256"]
    gpt = reference_input_manifest([data["ref_image"]])
    assert gpt[0]["file_sha256"] == gpt[0]["transmitted_sha256"]
    assert gpt[0]["encoding"] == "gpt" and not gpt[0]["reencoded"]


def test_paid_plan_reuses_success_and_blocks_changed_inputs(tmp_path, monkeypatch):
    import json
    import runpy
    from modules.others import api_backend as backend
    from utils import send_budget
    data = entry(tmp_path)
    data["reference_details_enabled"] = False
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["save_sub_dir"]) / "result.png"
        Image.new("RGB", (64, 64), "blue").save(target)
        return {"saved_files": [str(target)]}
    monkeypatch.setattr(backend, "generate_image_aigc2d", generate)
    monkeypatch.setattr(backend, "get_api_config", lambda **kw: {"model": "test-model", "resolution": "2K"})
    for key in (send_budget.ENV_DIR, send_budget.ENV_LIMIT, "IMAGE_MAKER_IMAGE_MAX_RETRIES"):
        monkeypatch.setenv(key, "")
    namespace = runpy.run_path(str(Path(__file__).with_name("test_style_ref.py")))
    run = namespace["run_plan"]
    run.__globals__["load_styles"] = lambda: {"x": data}
    (tmp_path / "subject.txt").write_text("One adult woman.", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"style": "x", "prompt_file": "subject.txt", "output_dir": str(tmp_path / "out"),
                               "aspect_ratio": "2:3", "max_calls": 1,
                               "variants": [{"name": "A", "face_quality_boost": False}]}), encoding="utf-8")
    assert run(str(plan)) == 0
    assert run(str(plan)) == 0
    assert len(calls) == 1
    Image.new("RGB", (200, 300), "green").save(data["ref_image"])
    with pytest.raises(ValueError, match="Frozen"):
        run(str(plan))
    assert len(calls) == 1
