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
import cohere

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

def fetch_llm_json(base_url: str, api_key: str, model: str, system_prompt: str, user_content: str, temperature: float = 0.5, merge_system_prompt: bool = True) -> str:
    """
    通用 LLM 对话请求函数，专门用于获取 JSON 格式文本，并记录完整请求和响应日志
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    if merge_system_prompt:
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_content}"}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        # 如果你使用的大模型 API 兼容 OpenAI，开启此选项能极大提高返回纯 JSON 的概率
        "response_format": { "type": "json_object" } 
    }

    logger.info("=== 发起 LLM 提示词请求 ===")
    logger.info(f"请求 URL: {url}")
    # 打印完整的 payload 到日志中
    logger.info(f"请求 Payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        resp_json = resp.json()
        
        # 将服务器返回的原始完整 JSON 记录到日志
        logger.info(f"=== 服务器原始返回完整信息 ===\n{json.dumps(resp_json, ensure_ascii=False, indent=2)}")
        
        # 提取模型回复的文本
        return resp_json['choices'][0]['message']['content'].strip()
        
    except requests.exceptions.Timeout:
        logger.error("LLM 请求超时。")
        return ""
    except Exception as e:
        logger.error(f"LLM 请求发生异常: {str(e)}")
        if 'resp' in locals():
            logger.error(f"服务器返回信息: {resp.text}")
        return ""

def fetch_cohere_json(system_prompt: str, user_content: str, temperature: float = 0.5) -> str:
    """
    专门用于读取 config-cohere.json 并请求 Cohere API 的函数
    """
    config_path = "config-cohere.json"
    if not os.path.exists(config_path):
        logger.error(f"未找到 Cohere 配置文件: {config_path}")
        return ""
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"读取 {config_path} 失败: {e}")
        return ""
        
    base_url = config.get("base_url", "https://api.cohere.com/v1").rstrip('/')
    # 智能补全 URL 路径
    if not base_url.endswith("/chat") and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    url = f"{base_url}/chat" if not base_url.endswith("/chat") else base_url
    
    api_key = config.get("api_key", "")
    model = config.get("model", "command-r-plus")
    merge_system_prompt = config.get("merge_system_prompt", False)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # 按照 Cohere API 的要求构造 message 和 preamble (系统提示词)
    if merge_system_prompt:
        message = f"{system_prompt}\n\n{user_content}"
        preamble = ""
    else:
        message = user_content
        preamble = system_prompt
    
    payload = {
        "model": model,
        "message": message,
        "temperature": temperature,
        "response_format": { "type": "json_object" } # 强制要求 Cohere 输出 JSON
    }
    
    if preamble:
        payload["preamble"] = preamble

    logger.info("=== 发起 Cohere LLM 提示词请求 ===")
    logger.info(f"请求 headers: {headers}")
    logger.info(f"请求 URL: {url}")
    logger.info(f"请求 Payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        resp_json = resp.json()
        
        logger.info(f"=== Cohere 服务器原始返回完整信息 ===\n{json.dumps(resp_json, ensure_ascii=False, indent=2)}")
        
        # Cohere V1 Endpoint 返回的文本内容在 'text' 字段中
        return resp_json.get('text', '').strip()
        
    except requests.exceptions.Timeout:
        logger.error("Cohere 请求超时。")
        return ""
    except Exception as e:
        logger.error(f"Cohere 请求发生异常: {str(e)}")
        if 'resp' in locals():
            logger.error(f"服务器返回信息: {resp.text}")
        return ""