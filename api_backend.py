# api_backend.py
from logging import config
import os
import json
import base64
import requests
import re
import logging
import copy
import uuid
from datetime import datetime
import time
import mimetypes

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

def get_api_config(config_path="config-image.json", api_type=None):
    """获取指定API类型的配置"""
    config = load_config(config_path)
    # 如果指定了api_type，直接返回对应配置
    if api_type:
        return config.get("apis", {}).get(api_type, {})
    # 否则返回当前API类型的配置
    current_api = config.get("current_api", "whatup")
    return config.get("apis", {}).get(current_api, {})

def to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def generate_image_whatai(prompt: str, image_paths: list = None, model: str = "nano-banana-2", aspect_ratio: str = "1:1", instructions: str = "", resolution = "1K", api_type: str = None, save_sub_dir: str = None, file_prefix: str = None) -> list:
    """
    独立出来的图片生成核心逻辑
    """
    config = get_api_config(api_type=api_type)
    # 兼容原有的 base_url 命名
    api_base = config.get("base_url", "https://api.whatai.cc/v1").rstrip('/')
    api_key = config.get("api_key")
    timeout_val = config.get("timeout", 120)      # <--- 读取超时配置，默认120
    max_retries = config.get("max_retries", 1)    # <--- 读取重试配置，默认1

    # TODO: 处理resolution，但是whatai其实根本不接受这个参数，目前只能放在prompt里让模型自己理解了
    
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

    # ================= 替换掉原来的 try 块，改为重试循环 =================
    resp = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"正在进行第 {attempt} 次重试 (最大重试次数: {max_retries})...")
            
            resp = requests.post(url, headers=headers, json=data, timeout=timeout_val)
            resp.raise_for_status()
            break  # 如果没有抛出异常，说明请求成功，跳出循环
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"网络请求发生异常 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2)  # 重试前稍微休息2秒，避免频繁打满后端
            else:
                logger.error("达到最大重试次数，图片生成请求最终失败。")
                return []
    # ====================================================================

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
    if save_sub_dir:
        save_dir = os.path.join("data", today_str, save_sub_dir)
    else:
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
            prefix = f"{file_prefix}_" if file_prefix else ""
            file_name = f"{prefix}output_{datetime.now().strftime('%H%M%S')}_{idx}_{uuid.uuid4().hex[:6]}{ext}"
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

def generate_image_aigc2d(prompt: str, image_paths: list = None, model: str = "gemini-3.1-flash-image-preview", aspect_ratio: str = "1:1", instructions: str = "", resolution: str = None, api_type: str = None, save_sub_dir: str = None, file_prefix: str = None) -> list:
    """
    AIGC2D 专用的图片生成核心逻辑
    入参跟 generate_image_whatai 保持完全一致
    """
    # 从统一配置文件中加载 aigc2d 配置
    config = get_api_config(api_type="aigc2d" if not api_type else api_type)

    api_base = config.get("base_url", "https://next.aigc2d.com/v1beta").rstrip('/')
    api_key = config.get("api_key")
    timeout_val = config.get("timeout", 180)
    max_retries = config.get("max_retries", 1)
    # 从配置中读取resolution，如果传入了参数则使用传入的参数
    if resolution is None:
        resolution = config.get("resolution", "1K")

    if not api_key:
        logger.error("配置文件 config-image.json 中缺少 'api_key' 参数。")
        return []

    # AIGC2D 接口 URL 拼接规则
    url = f"{api_base}/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    # 组合提示词
    combined_prompt = f"{instructions} \n {prompt}".strip() if instructions else prompt
    parts = [{"text": combined_prompt}]

    # 处理传入的参考图片（支持多图，按照入参列表追加）
    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                mime_type, _ = mimetypes.guess_type(img_path)
                if not mime_type:
                    mime_type = "image/jpeg"
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": to_base64(img_path)
                    }
                })
            else:
                logger.warning(f"找不到本地图片文件 {img_path}，已跳过。")

    # 构造请求 Payload
    payload = {
        "contents": [
            {
                "parts": parts
            }
        ],
        "generationConfig": {
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": resolution
            }
        }
    }

    # 日志脱敏：隐藏 Base64 字符串
    safe_payload = copy.deepcopy(payload)
    for part in safe_payload.get("contents", [])[0].get("parts", []):
        if "inline_data" in part and "data" in part["inline_data"]:
            part["inline_data"]["data"] = "<BASE64_IMAGE_DATA_OMITTED>"
            
    logger.info("=== 发起 AIGC2D API 请求 ===")
    logger.info(f"请求数据:\n{json.dumps(safe_payload, ensure_ascii=False, indent=2)}")

    # 带重试机制的请求
    resp = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"正在进行第 {attempt} 次重试 (最大重试次数: {max_retries})...")
            
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_val)
            resp.raise_for_status()
            break  
        except requests.exceptions.RequestException as e:
            logger.warning(f"网络请求发生异常 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                logger.error("达到最大重试次数，AIGC2D 图片生成请求最终失败。")
                if resp is not None:
                    logger.error(f"最后一次响应内容: {resp.text}")
                return []

    # 解析返回 JSON
    try:
        resp_json = resp.json()
        # 记录脱敏后的原始返回（避免返回巨量 base64 撑爆日志）
        safe_resp_json = copy.deepcopy(resp_json)
        for cand in safe_resp_json.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "inlineData" in part:
                    part["inlineData"]["data"] = "<BASE64_IMAGE_DATA_OMITTED>"
                elif "inline_data" in part:
                    part["inline_data"]["data"] = "<BASE64_IMAGE_DATA_OMITTED>"
                    
        logger.info(f"=== AIGC2D 服务器返回信息 ===\n{json.dumps(safe_resp_json, ensure_ascii=False, indent=2)}")
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"解析 AIGC2D 返回 JSON 失败: {e}")
        return []

    # 提取图片并保存
    today_str = datetime.now().strftime("%Y%m%d")
    if save_sub_dir:
        save_dir = os.path.join("data", today_str, save_sub_dir)
    else:
        save_dir = os.path.join("data", today_str)
    os.makedirs(save_dir, exist_ok=True)
    saved_files = []

    candidates = resp_json.get("candidates", [])
    if not candidates:
        logger.warning("AIGC2D 返回的 JSON 中没有找到 candidates 节点。")
        return []

    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            # AIGC2D 的 Gemini API 结构可能是 inlineData 或 inline_data
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data and "data" in inline_data:
                mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
                ext = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(mime_type, ".png")

                # 生成形如：P01_142305-a1b2c3.png 的文件名
                time_str = datetime.now().strftime('%H%M%S')
                random_str = uuid.uuid4().hex[:6]
                prefix = f"{file_prefix}_" if file_prefix else ""
                file_name = f"{prefix}{time_str}-{random_str}{ext}"
                file_path = os.path.join(save_dir, file_name)

                try:
                    image_bytes = base64.b64decode(inline_data["data"])
                    with open(file_path, "wb") as f:
                        f.write(image_bytes)
                    logger.info(f"✅ 成功保存图片 ({ext} 格式): {file_path}")
                    saved_files.append(file_path)
                except Exception as e:
                    logger.error(f"写入图片文件失败: {e}")
                    
            elif "text" in part:
                # 顺手记录一下模型可能返回的额外文本提示
                logger.info(f"模型文本反馈: {part['text']}")

    return saved_files