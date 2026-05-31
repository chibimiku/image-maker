import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")


def get_prompt_path(relative_path: str) -> str:
    path_text = str(relative_path or "").strip().replace("/", os.sep).replace("\\", os.sep)
    if not path_text:
        raise ValueError("Prompt 相对路径不能为空")
    return os.path.join(PROMPTS_DIR, path_text)


def read_prompt_file(relative_path: str) -> str:
    prompt_path = get_prompt_path(relative_path)
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt_file(relative_path: str, replacements=None) -> str:
    text = read_prompt_file(relative_path)
    for key, value in (replacements or {}).items():
        text = text.replace("{" + str(key) + "}", str(value))
    return text


def find_missing_prompt_files(relative_paths):
    missing = []
    for relative_path in (relative_paths or []):
        prompt_path = get_prompt_path(relative_path)
        if not os.path.isfile(prompt_path):
            missing.append(prompt_path)
    return missing
