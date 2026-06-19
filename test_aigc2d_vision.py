"""
测试 aigc2d API 是否支持 OpenAI 兼容的 vision/chat completions 接口
用于验证图片分析模块能否直接使用 aigc2d 的 key 和 base_url
"""
import os
import io
import json
import base64
from openai import OpenAI
from PIL import Image

# ===== 配置：从 config-image.json 读取 aigc2d 的 key =====
def _load_aigc2d_key():
    conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf", "config-image.json")
    try:
        with open(conf_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        api = config.get("apis", {}).get("aigc2d", {})
        return api.get("api_key", ""), api.get("base_url", "")
    except Exception:
        return "", ""

API_KEY, _img_base = _load_aigc2d_key()
BASE_URL = "https://next.aigc2d.com/v1"  # OpenAI 兼容端点 (用于文本/vision分析)
MODEL = "gemini-2.5-flash"               # 尝试 Gemini Flash (支持 vision)

if not API_KEY:
    print("请先在 config-image.json 中配置 aigc2d 的 api_key")
    exit(1)

print(f"测试目标: {BASE_URL}")
print(f"模型: {MODEL}")
print(f"API Key 前缀: {API_KEY[:20]}...")
print()

# 1. 创建一张纯色测试图片 (避免依赖外部文件)
print("[1] 创建测试图片...")
img = Image.new("RGB", (256, 256), color=(73, 109, 137))
buf = io.BytesIO()
img.save(buf, format="PNG")
img_base64 = base64.b64encode(buf.getvalue()).decode()
print(f"    图片大小: 256x256, base64长度: {len(img_base64)}")
print()

# 2. 创建 OpenAI 客户端
print("[2] 创建 OpenAI 客户端...")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60)
print(f"    base_url={BASE_URL}")
print(f"    实际请求路径: {BASE_URL}/chat/completions")
print()

# 3. 发送 vision 请求 (模拟图片分析模块的请求格式)
print("[3] 发送 vision 请求...")
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an image analysis assistant. Respond in JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly in one sentence. Output ONLY a JSON object like {\"description\": \"...\"}."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}",
                        }
                    }
                ]
            }
        ],
        temperature=0.3,
        max_completion_tokens=512,
    )
    print("    请求成功!")
    print(f"    finish_reason: {response.choices[0].finish_reason}")
    print(f"    模型返回: {response.model}")
    content = response.choices[0].message.content
    print(f"    响应内容: {content[:500]}")
    print()
    print("=" * 60)
    print("结论: aigc2d 的 /v1 端点支持 OpenAI 兼容的 vision API!")
    print("图片分析模块可以直接使用以下配置:")
    print(f"  base_url: {BASE_URL}")
    print(f"  api_key:  {API_KEY[:12]}...（请从config-image.json获取完整key）")
    print(f"  model:    {MODEL}")
    print("=" * 60)

except Exception as e:
    error_msg = str(e)
    print(f"    请求失败: {error_msg[:800]}")
    print()

    # 尝试其他模型名
    print("[4] 尝试其他模型名...")
    for alt_model in ["gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-pro-vision"]:
        try:
            print(f"    尝试模型: {alt_model}...")
            response = client.chat.completions.create(
                model=alt_model,
                messages=[
                    {"role": "user", "content": "Say 'hello' in one word."}
                ],
                max_completion_tokens=32,
            )
            print(f"    成功! 模型 {alt_model} 可用.")
            print(f"    响应: {response.choices[0].message.content}")
            # Try vision with this model
            print(f"    使用 {alt_model} 测试 vision...")
            response = client.chat.completions.create(
                model=alt_model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this image in one sentence."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]}
                ],
                max_completion_tokens=256,
            )
            print(f"    Vision 响应: {response.choices[0].message.content[:300]}")
            print()
            print("=" * 60)
            print(f"结论: aigc2d 支持 vision API! 使用模型: {alt_model}")
            print(f"  base_url: {BASE_URL}")
            print(f"  api_key:  {API_KEY[:12]}...（请从config-image.json获取完整key）")
            print(f"  model:    {alt_model}")
            print("=" * 60)
            break
        except Exception as e2:
            print(f"    模型 {alt_model} 不可用: {str(e2)[:200]}")

    # 尝试不带 /v1 的 base_url
    print()
    print("[5] 尝试 base_url = https://next.aigc2d.com (不带 /v1)...")
    try:
        client2 = OpenAI(api_key=API_KEY, base_url="https://next.aigc2d.com", timeout=60)
        response = client2.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello."}],
            max_completion_tokens=32,
        )
        print(f"    成功! 响应: {response.choices[0].message.content}")
    except Exception as e3:
        print(f"    失败: {str(e3)[:200]}")

    # 获取可用模型列表
    print()
    print("[6] 尝试获取可用模型列表...")
    try:
        models = client.models.list()
        model_ids = [m.id for m in models]
        print(f"    可用模型 ({len(model_ids)}):")
        for mid in model_ids[:30]:
            print(f"      - {mid}")
        if len(model_ids) > 30:
            print(f"      ... 还有 {len(model_ids) - 30} 个")
    except Exception as e4:
        print(f"    获取模型列表失败: {str(e4)[:300]}")
