import os
import json
import base64
import requests
import re
from datetime import datetime

def load_config(config_path="config-image.json"):
    """读取本地 API 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}，请确保其与脚本在同一目录下。")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_base64(path):
    """将本地图片转换为 Base64 编码"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def generate_image_whatai(prompt: str, image_paths: list = None, model: str = "nano-banana-2") -> list:
    """
    调用 Whatai 服务商的图像生成 API
    :param prompt: 提示词
    :param image_paths: 参考图片路径列表
    :param model: 使用的模型，默认 nano-banana-2
    :return: 成功保存到本地的图片路径列表
    """
    # 1. 加载配置
    config = load_config()
    api_base = config.get("api_base", "https://api.whatai.cc/v1")
    api_key = config.get("api_key")
    
    if not api_key:
        raise ValueError("配置文件 config-image.json 中缺少 'api_key' 参数。")

    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 2. 构建符合 Nano Banana / Gemini 规范的 messages
    # 强制在 prompt 中要求输出 PNG 格式，引导模型行为
    content_list = [{"type": "text", "text": f"{prompt} (请确保生成结果为PNG格式)"}]

    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                # 简单推断 MIME 类型
                ext = os.path.splitext(img_path)[-1].lower()
                mime_type = "image/png" if ext == ".png" else "image/jpeg"
                
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{to_base64(img_path)}"
                    }
                })
            else:
                print(f"⚠️ 警告: 找不到本地图片文件 {img_path}，已跳过。")

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content_list
            }
        ]
    }

    # 3. 发送请求
    print(f"正在向 Whatai 接口发送生成请求 (模型: {model})...")
    resp = requests.post(url, headers=headers, json=data)

    if resp.status_code != 200:
        print(f"❌ 请求失败，状态码: {resp.status_code}")
        print("返回信息:\n", resp.text)
        return []

    # 4. 解析返回的 Markdown，提取图片 URL
    resp_json = resp.json()
    try:
        content_str = resp_json["choices"][0]["message"]["content"]
    except KeyError:
        print("❌ 解析返回 JSON 失败，格式不符合预期:", resp_json)
        return []

    # 使用正则匹配 Markdown 格式中的图片：![描述](URL)
    img_urls = re.findall(r'!\[.*?\]\((https?://[^\)]+)\)', content_str)

    if not img_urls:
        print("⚠️ 接口调用成功，但未在返回的文本中找到图片链接。")
        print("接口原始返回文本:\n", content_str)
        return []

    print(f"✅ 成功获取 {len(img_urls)} 张图片链接，准备下载...")

    # 5. 创建本地保存目录 data/YYYYMMDD/
    today_str = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join("data", today_str)
    os.makedirs(save_dir, exist_ok=True)

    saved_files = []
    
    # 6. 下载并强制保存为 .png
    for idx, img_url in enumerate(img_urls):
        try:
            img_data = requests.get(img_url, timeout=15).content
            
            # 以当前时间戳命名避免覆盖
            timestamp = datetime.now().strftime("%H%M%S")
            file_name = f"output_{timestamp}_{idx}.png"
            file_path = os.path.join(save_dir, file_name)

            with open(file_path, "wb") as f:
                f.write(img_data)
                
            print(f"➡️ 成功保存图片至: {file_path}")
            saved_files.append(file_path)
            
        except Exception as e:
            print(f"❌ 下载图片失败 {img_url}: {e}")

    return saved_files

# ================= 测试调用 =================
if __name__ == "__main__":
    # 模拟外部调用
    #test_prompt = "精美的洛可可风格少女插画，Galgame CG质感，高光细节丰富"
    test_prompt = '''
A joyful teenage girl standing in bright spring sunlight, open grassy meadow filled with countless blooming cherry blossom trees, soft pink petals gently floating in the warm breeze, golden hour lighting, dreamy and romantic atmosphere, she is looking directly at the viewer with a sweet, heartwarming smile, sparkling excitement in her large expressive brown eyes, cute slightly flushed cheeks, innocent and playful expression, very lovely and approachable vibe.

She has long, silky bubblegum-pink hair flowing naturally down to her lower back, soft loose waves, several strands gently blowing across her face in the wind, hair shines with subtle iridescent highlights under sunlight, adorned with a few tiny white cherry blossom flowers naturally caught in her hair.

She is wearing the exact dress from reference test-dress.png: a delicate, elegant springtime mini dress with soft layered ruffles, light and airy fabric, beautiful gradient color transition from pale sakura pink at the bodice to deeper cherry-blossom pink at the hem, intricate lace overlays on the sweetheart neckline and short puffed sleeves, cinched waist with a subtle satin ribbon tie creating a perfect A-line silhouette, skirt flares out playfully with multiple soft tulle and chiffon layers, feminine and romantic yet youthful and fresh.

Paired perfectly with the shoes from reference test-shoes.png: dainty strappy heeled sandals in matching pale pink satin, thin delicate ankle straps with small crystal-like buckles, open toe design showing cute pedicured toes, low kitten heel giving her graceful posture without losing the innocent girlish charm.

She stands in a gentle contrapposto pose, one hand lightly holding a few fallen cherry blossom petals, the other hand naturally resting by her side, slight tilt of the head, beaming with pure happiness and springtime energy. Cinematic composition, rule of thirds, viewer feels personally greeted by her warm gaze.

Ultra-detailed textures, photorealistic skin with subtle peach fuzz and natural blush, soft bokeh background of pink-white blossoms and fresh green grass, rim lighting and glowing backlight creating dreamy halo effect around her hair, pastel color grading, vibrant yet gentle spring mood, masterpiece, award-winning anime-art fusion style with realistic rendering, 8k resolution, highly detailed, emotional lighting --ar 3:4 --stylize 250 --v 6 --q 2
'''
    test_images = ['data/test/test-dress.png', 'data/test/test-shoes.png']
    
    # 执行生成
    results = generate_image_whatai(
        prompt=test_prompt,
        image_paths=test_images,
        model="nano-banana-pro" 
    )
    
    if results:
        print("\n🎉 任务完成！所有图片已处理完毕。")