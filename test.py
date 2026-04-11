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
# 自动创建 log 目录，保障容错性
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)

# 初始化 Logger
logger = logging.getLogger("whatai_logger")
logger.setLevel(logging.INFO)

# 避免重复添加 Handler（适用于在 Jupyter 等交互式环境中多次运行）
if not logger.handlers:
    # 定义日志格式
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 文件 Handler (动态按天命名，例如 log/2026-03-18.log)
    today_date = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(LOG_DIR, f"{today_date}.log")
    file_handler = logging.FileHandler(filename=log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ================= 2. 核心功能函数 =================
def load_config(config_path="config-image.json"):
    """读取本地 API 配置文件"""
    if not os.path.exists(config_path):
        logger.error(f"未找到配置文件: {config_path}")
        raise FileNotFoundError(f"未找到配置文件: {config_path}，请确保其与脚本在同一目录下。")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_base64(path):
    """将本地图片转换为 Base64 编码"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _looks_like_base64_text(value: str) -> bool:
    if not isinstance(value, str):
        return False
    compact = value.strip().replace("\n", "").replace("\r", "")
    if len(compact) < 256:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        return False
    return True

def _sanitize_log_text(value: str) -> str:
    if not isinstance(value, str):
        return str(value)
    sanitized_text = re.sub(
        r"(data:image\/[a-zA-Z0-9.+-]+;base64,)[A-Za-z0-9+/=\r\n]+",
        r"\1<BASE64_IMAGE_DATA_OMITTED>",
        value
    )
    if _looks_like_base64_text(sanitized_text):
        return "<BASE64_IMAGE_DATA_OMITTED>"
    if len(sanitized_text) > 4000:
        return f"{sanitized_text[:4000]}...(TRUNCATED, total={len(sanitized_text)})"
    return sanitized_text

def generate_image_whatai(
    prompt: str, 
    image_paths: list = None, 
    model: str = "nano-banana-2",
    aspect_ratio: str = "1:1",
    instructions: str = ""
) -> list:
    """
    调用 Whatai 服务商的图像生成 API
    :param prompt: 提示词
    :param image_paths: 参考图片路径列表
    :param model: 使用的模型
    :param aspect_ratio: 图片长宽比，例如 "1:1", "16:9", "3:4"
    :param instructions: 生成指令，例如 "请生成一个与参考图片风格相似的图片"
    :return: 成功保存到本地的图片路径列表
    """
    # 加载配置
    config = load_config()
    api_base = config.get("api_base", "https://api.whatai.cc/v1")
    api_key = config.get("api_key")
    
    if not api_key:
        logger.error("配置文件 config-image.json 中缺少 'api_key' 参数。")
        raise ValueError("配置文件 config-image.json 中缺少 'api_key' 参数。")

    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构建 messages
    content_list = [{"type": "text", "text": f"--ar {aspect_ratio} ,  {instructions}  {prompt}"}]

    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                ext = os.path.splitext(img_path)[-1].lower()
                mime_type = "image/png" if ext == ".png" else "image/jpeg"
                
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{to_base64(img_path)}"
                    }
                })
            else:
                logger.warning(f"找不到本地图片文件 {img_path}，已跳过。")

    data = {
        "model": model,
        "aspect_ratio": aspect_ratio,
        #"instructions": instructions, #这个比不支持
        "messages": [
            {
                "role": "user",
                "content": content_list
            }
        ]
    }

    # 打印和保存省略了 Base64 图片数据的原始请求
    safe_data = copy.deepcopy(data)
    for msg in safe_data.get("messages", []):
        for content in msg.get("content", []):
            if type(content) == dict and content.get("type") == "image_url":
                content["image_url"]["url"] = "<BASE64_IMAGE_DATA_OMITTED_FOR_LOG>"
    
    logger.info("=== 发起新的 API 请求 ===")
    logger.info(url)
    logger.info(headers)
    logger.info(f"请求数据:\n{json.dumps(safe_data, ensure_ascii=False, indent=2)}")

    # 发送请求
    logger.info(f"正在向 Whatai 接口发送生成请求 (模型: {model}, 长宽比: {aspect_ratio})...")
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=60)
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求发生异常: {e}")
        return []

    # 打印和保存原始响应
    logger.info(f"接收到接口响应状态码: {resp.status_code}")
    logger.info(f"接收到接口响应内容:\n{_sanitize_log_text(resp.text)}")

    if resp.status_code != 200:
        logger.error("请求失败，已终止处理。")
        return []

    # 解析返回的 Markdown
    try:
        resp_json = resp.json()
        content_str = resp_json["choices"][0]["message"]["content"]
    except (KeyError, json.JSONDecodeError):
        logger.error("解析返回 JSON 失败，格式不符合预期。")
        return []

    # 提取图片 URL
    img_urls = re.findall(r'!\[.*?\]\((https?://[^\)]+)\)', content_str)

    if not img_urls:
        logger.warning("接口调用成功，但未在返回的文本中找到图片链接。")
        return []

    logger.info(f"成功获取 {len(img_urls)} 张图片链接，准备下载...")

    # 创建图片保存目录 (data/YYYYMMDD/)
    today_str = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join("data", today_str)
    os.makedirs(save_dir, exist_ok=True)

    saved_files = []
    
    # 下载并保存图片
    for idx, img_url in enumerate(img_urls):
        try:
            response = requests.get(img_url, timeout=15)
            response.raise_for_status() # 确保请求成功
            img_data = response.content
            
            # 根据二进制数据的文件头 (Magic Number) 判断真实的图片类型
            ext = ".png"  # 默认兜底后缀
            if img_data.startswith(b'\xff\xd8\xff'):
                ext = ".jpg"
            elif img_data.startswith(b'\x89PNG\r\n\x1a\n'):
                ext = ".png"
            elif img_data.startswith(b'GIF8'):
                ext = ".gif"
            elif img_data.startswith(b'RIFF') and img_data[8:12] == b'WEBP':
                ext = ".webp"
            
            # 使用时间戳 + 序号 + 短UUID 防止同名冲突
            timestamp = datetime.now().strftime("%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            file_name = f"output_{timestamp}_{idx}_{short_uuid}{ext}"
            file_path = os.path.join(save_dir, file_name)

            with open(file_path, "wb") as f:
                f.write(img_data)
                
            logger.info(f"成功保存图片至: {file_path}")
            saved_files.append(file_path)
            
        except Exception as e:
            logger.error(f"下载图片失败 {img_url}: {e}")

    return saved_files


# ================= 3. 测试调用 =================
if __name__ == "__main__":
    test_prompt = '''
The image is a highly detailed fantasy-style illustration of a girl surrounded by a celestial atmosphere. The artwork employs a soft yet luminous digital painting style with meticulous line work and sparkling light effects reminiscent of starlight. The central figure stands gracefully against a cosmic sky filled with glowing stars and descending light trails, evoking an ethereal dreamscape. The composition centers on the girl’s upper body and flowing gown, emphasizing her elegance and the visual harmony between her attire and the environment.\n\nThe girl’s long, wavy hair flows outward in emerald-green and turquoise shades, cascading down her shoulders and back. Each strand reflects subtle gradients of light, suggesting a gentle, otherworldly breeze. Her eyes, though partially obscured, appear to shimmer in blue tones that mirror the stars. The lighting is primarily cool and radiant, emanating from the celestial background and subtly highlighting the crystalline details of her costume. The color palette transitions smoothly from deep midnight blue to soft aquamarine, reinforcing the serene, cosmic elegance of the setting.\n\nHer dress is the focal point of the illustration, rendered with exquisite detailing. The bodice is a deep blue, densely speckled with white star patterns, creating the illusion of a night sky wrapped around her form. The skirt flows outward gracefully, marked by constellations and elegant shooting star motifs that stretch diagonally across the fabric, glowing softly with golden-white streaks. A pale blue ribbon belt ties neatly around her waist, adding a delicate, structured contrast to the free-flowing nature of the gown.\n\nOver her shoulders drapes a translucent cape tinted with icy blue. The sheer fabric features embroidered floral patterns on the sleeves, with tiny blossoms in shades of sapphire and powder blue. The ends of the sleeves are cinched with cuffs decorated with faint lacework and dotted with small gemstones, catching light like tiny comets. The girl wears a large sapphire pendant on a ribbon choker around her neck, and another teardrop-shaped gemstone gleams at her chest, matching the jewel set within an elaborate ornament near her hair. This ornament branches outward like crystalline frost or stardust, contributing to the ethereal quality of her presence.\n\nThe background complements her appearance perfectly: thousands of softly glowing stars hang from delicate light strands, descending vertically across the top portion of the image. Wisps of clouds below suggest a skybound altitude, reinforcing the celestial aesthetic. The composition adopts a medium-frontal angle, focusing on the girl’s figure with symmetrical balance, suggesting both divinity and serenity. No other characters are present, emphasizing solitude and majesty within this starry realm. The artwork evokes themes of purity, cosmic wonder, and dreamy transcendence, successfully blending the elegance of rococo-inspired fashion with the mystique of the universe.
'''
    instructions='''You are a generative model specialized in creating images in a delicate, high-detail, modern anime style. Your creations should evoke a bright, dreamy, and elegant atmosphere.When generating characters, pay close attention to the following features to emulate the target style:Face: Render soft, youthful facial features. The face should be animated with a gentle expression and a subtle blush on the cheeks, conveying a sense of sweet innocence. If the character is female, then two strands of hair will cover each side of her face.Eyes and Eyelashes: The eyes are the central focus and must be large, expressive, and imbued with a gem-like quality. Construct the irises with intricate, multi-layered details, using a primary warm golden hue blended with lighter and darker tones to create depth and a crystalline, refractive appearance. Incorporate multiple, sharp white highlights of varying sizes to simulate a moist, lively sparkle.Include at least one highlight that is nearly pure white at bottom of pupil to simulate the brightest sparkle. Frame the eyes with long, distinct eyelashes. The upper lashes should be full and dark, with individual strands clearly defined. The lower lashes are to be rendered with finer, more delicate strokes. Place tiny, reflective highlights along the lower lash line to suggest a teary or dewy quality.Hair: Hair should be long, flowing, and voluminous. Render it with a multi-tonal, almost iridescent quality, blending a primary soft color with subtle, colorful undertones (pinks, purples, blues). Emphasize its texture by drawing fine, individual strands and applying a mix of soft glows and sharp, specular highlights. This will create a silky, shimmering appearance as if light is passing through it.Shoes: have a blend of glass and leather textures, with a mirror-like reflective surface.'''
    test_images = ['data/test/test-dress.png', 'data/test/test-shoes.png']

    #gemini-3.1-flash-image-preview-2k
    
    # 执行生成
    results = generate_image_whatai(
        prompt=test_prompt,
        image_paths=test_images,
        model="gemini-3.1-flash-image-preview-4k",
        #model="gemini-3-pro-image-preview",
        aspect_ratio="2:3",
        instructions=instructions
    )
    
    if results:
        logger.info("🎉 任务完成！所有图片已处理完毕。")
