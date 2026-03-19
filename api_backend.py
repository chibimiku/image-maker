# api_backend.py
import os
import json
import base64
import requests
import re
import logging
import copy
import uuid
from datetime import datetime

# ================= 1. 日志系统配置 =================
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("whatai_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(fmt='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(LOG_DIR, f"{today_date}.log")
    file_handler = logging.FileHandler(filename=log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ================= 2. 核心功能函数 =================
def load_config(config_path="config-image.json"):
    if not os.path.exists(config_path):
        logger.error(f"未找到配置文件: {config_path}")
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def generate_image_whatai(prompt: str, image_paths: list = None, model: str = "nano-banana-2", aspect_ratio: str = "1:1", instructions: str = "") -> list:
    """
    独立出来的图片生成核心逻辑
    """
    config = load_config()
    # 兼容原有的 base_url 命名
    api_base = config.get("base_url", "https://api.whatai.cc/v1").rstrip('/')
    api_key = config.get("api_key")
    
    if not api_key:
        logger.error("配置文件 config-image.json 中缺少 'api_key' 参数。")
        raise ValueError("配置文件 config-image.json 中缺少 'api_key' 参数。")

    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 将用户保存的 Instructions 和具体的 prompt 拼接传入
    combined_prompt = f"--ar {aspect_ratio} ,  {instructions}  {prompt}"
    content_list = [{"type": "text", "text": combined_prompt}]

    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                ext = os.path.splitext(img_path)[-1].lower()
                mime_type = "image/png" if ext == ".png" else "image/jpeg" # <- 替换为这行
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{to_base64(img_path)}"}
                })
            else:
                logger.warning(f"找不到本地图片文件 {img_path}，已跳过。")

    data = {
        "aspect_ratio": aspect_ratio,
        "model": model,
        "messages": [{"role": "user", "content": content_list}]
    }

    # 隐藏 Base64 打印日志
    safe_data = copy.deepcopy(data)
    for msg in safe_data.get("messages", []):
        for content in msg.get("content", []):
            if type(content) == dict and content.get("type") == "image_url":
                content["image_url"]["url"] = "<BASE64_IMAGE_DATA_OMITTED>"
    
    logger.info("=== 发起 API 请求 ===")
    logger.info(f"请求数据:\n{json.dumps(safe_data, ensure_ascii=False, indent=2)}")

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求发生异常: {e}")
        return []

    try:
        resp_json = resp.json()
        
        # 【新增】打印服务器返回的原始完整 JSON 信息
        logger.info(f"=== 服务器原始返回信息 ===\n{json.dumps(resp_json, ensure_ascii=False, indent=2)}")
        
        content_str = resp_json["choices"][0]["message"]["content"]
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"解析返回 JSON 失败: {e}")
        return []

    # 解析 Markdown 提取图片
    img_urls = re.findall(r'!\[.*?\]\((https?://[^\)]+)\)', content_str)
    if not img_urls:
        logger.warning("未在返回的文本中找到图片链接。")
        return []

    today_str = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join("data", today_str)
    os.makedirs(save_dir, exist_ok=True)
    saved_files = []
    
    for idx, img_url in enumerate(img_urls):
        try:
            # 【优化】获取完整的 response 以读取内容
            img_resp = requests.get(img_url, timeout=30)
            img_resp.raise_for_status()
            img_data = img_resp.content
            
            # 【新增】通过二进制文件头（Magic Bytes）准确识别图片格式
            ext = ".png"  # 默认兜底后缀
            if img_data.startswith(b'\xff\xd8'):
                ext = ".jpg"
            elif img_data.startswith(b'\x89PNG\r\n\x1a\n'):
                ext = ".png"
            elif img_data.startswith(b'RIFF') and img_data[8:12] == b'WEBP':
                ext = ".webp"
            elif img_data.startswith(b'GIF87a') or img_data.startswith(b'GIF89a'):
                ext = ".gif"
            
            # 使用识别出的正确后缀保存文件
            file_name = f"output_{datetime.now().strftime('%H%M%S')}_{idx}_{uuid.uuid4().hex[:6]}{ext}"
            file_path = os.path.join(save_dir, file_name)
            
            with open(file_path, "wb") as f:
                f.write(img_data)
                
            logger.info(f"✅ 成功保存图片 ({ext} 格式): {file_path}")
            saved_files.append(file_path)
            
        except Exception as e:
            logger.error(f"下载图片失败 {img_url}: {e}")

    return saved_files