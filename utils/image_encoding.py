"""公共图片编码工具：把图片压缩到指定尺寸内并编码为 JPEG base64。

统一承担此前在 style_analyzer.py / single_analyzer.py 中各自重复实现的
compress_and_encode_image 逻辑，便于维护并统一质量参数。
"""
import io
import base64
from PIL import Image


DEFAULT_MAX_DIM = 2048
DEFAULT_QUALITY = 95


def compress_and_encode_image(image_source, max_dim=DEFAULT_MAX_DIM,
                              quality=DEFAULT_QUALITY, log_callback=None):
    """将图片压缩到 max_dim 内并编码为 JPEG base64。

    参数:
        image_source: 图片路径(str) 或 PIL Image 对象。
        max_dim: 最长边阈值，超过则等比缩放。
        quality: JPEG 质量，越小体积越小。
        log_callback: 可选回调，用于输出尺寸/压缩信息。

    返回:
        (mime_type, base64_string)；失败时返回 (None, None)。
    """
    try:
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = image_source

        if img.mode != 'RGB':
            img = img.convert('RGB')

        original_width, original_height = img.size
        size_msg = f"原始图片尺寸: {original_width}x{original_height}"
        if log_callback:
            log_callback(size_msg)
        print(size_msg)

        if max(original_width, original_height) > max_dim:
            scaling_factor = max_dim / max(original_width, original_height)
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resize_msg = f"图片已成功压缩为: {new_width}x{new_height}"
            if log_callback:
                log_callback(resize_msg)
            print(resize_msg)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return "image/jpeg", base64_string

    except Exception as e:
        error_msg = f"处理图片时发生错误: {e}"
        if log_callback:
            log_callback(error_msg)
        print(error_msg)
        return None, None
