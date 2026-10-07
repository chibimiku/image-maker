from utils.image_references import reference_changes, REMOVED_REFERENCE


def remap(text, old, new):
    for start, end, replacement in reversed(reference_changes(text, old, new)):
        text = text[:start] + replacement + text[end:]
    return text


def test_swapped_images_keep_all_references_bound_to_their_original_images():
    text = "让Image 1跑步，用图2姿势。IMAGE 1 的服装；image2 的动作。Image 3只参考画风。"
    assert remap(text, ["person", "pose", "style"], ["pose", "person", "style"]) == (
        "让Image 2跑步，用图1姿势。IMAGE 2 的服装；image1 的动作。Image 3只参考画风。")


def test_appending_content_moves_only_the_automatic_style_reference():
    assert remap("Image 1角色，Image 2画风", ["person", "style"], ["person", "pose", "style"]) == "Image 1角色，Image 3画风"


def test_removing_referenced_image_never_silently_retargets_it():
    assert remap("图1人物，图2动作，Image 3画风", ["person", "pose", "style"], ["person", "style"]) == f"图1人物，{REMOVED_REFERENCE}动作，Image 2画风"


def test_unrelated_numbers_and_words_are_unchanged():
    text = "1536x1024, 第1章，filename_image1.png, Image 10, Image 1"
    assert remap(text, ["person"], ["pose", "person"]) == "1536x1024, 第1章，filename_image1.png, Image 10, Image 2"
