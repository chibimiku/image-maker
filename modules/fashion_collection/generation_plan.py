from __future__ import annotations

import json
import logging
import os
import random

from .theme_profiles import ThemeProfile
from utils.styles import style_prompt

logger = logging.getLogger(__name__)

GENERIC_PROMPT = "请生成一位可爱梦幻的少女，全身像，站姿自然，画面干净，突出服装整体搭配感。"
GENERIC_INSTRUCTIONS = "请严格参考输入的服饰图片完成一位少女角色的穿搭组合，保持主服装、鞋子、袜子、发饰和包袋的款式与颜色协调一致，输出日系少女插画风格。"


PART_LABELS = {
    "dress": "连衣裙",
    "shoes": "鞋子",
    "socks": "袜子",
    "hair_accessory": "发饰",
    "bag": "包袋",
}


def load_styles_config(styles_path: str) -> dict[str, str]:
    if not styles_path or not os.path.isfile(styles_path):
        return {}
    with open(styles_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return {str(k): style_prompt(payload, k) for k in payload}


def split_style_tokens(style_value: str) -> list[str]:
    text = str(style_value or "").replace("，", ",").replace("\n", ",")
    return [token.strip() for token in text.split(",") if token.strip()]


def resolve_style_bundle(style_value: str, styles_data: dict[str, str], theme_profile: ThemeProfile | None = None) -> tuple[list[str], str]:
    tokens = split_style_tokens(style_value)
    resolved_names: list[str] = []
    resolved_texts: list[str] = []

    if not tokens and theme_profile:
        tokens = list(theme_profile.default_style_names)

    for token in tokens:
        if token in styles_data:
            resolved_names.append(token)
            style_text = str(styles_data.get(token) or "").strip()
            if style_text:
                resolved_texts.append(style_text)
        else:
            resolved_texts.append(token)

    seen_texts: list[str] = []
    for text in resolved_texts:
        if text and text not in seen_texts:
            seen_texts.append(text)
    return resolved_names, "\n\n".join(seen_texts).strip()


def resolve_prompt_and_instructions(
    prompt: str,
    instructions: str,
    theme_profile: ThemeProfile | None,
    style_text: str,
) -> tuple[str, str]:
    final_prompt = str(prompt or "").strip()
    final_instructions = str(instructions or "").strip()

    if not final_prompt:
        final_prompt = (theme_profile.default_prompt if theme_profile else "") or GENERIC_PROMPT
    if not final_instructions:
        final_instructions = (theme_profile.default_instructions if theme_profile else "") or GENERIC_INSTRUCTIONS

    if theme_profile:
        final_prompt = f"{theme_profile.title}\n{final_prompt}".strip()
    if style_text:
        final_instructions = f"{final_instructions}\n\n画风参考：\n{style_text}".strip()
    return final_prompt, final_instructions


# ---- 主题相关的场景/角色描述随机池 ----

_THEMED_SCENE_CHARACTER_POOLS: dict[str, dict] = {
    "sweet-lolita": {
        "scenes": [
            "场景设定：欧式花园下午茶氛围，暖阳从花架和玻璃温室间洒下，背景有玫瑰、蕾丝桌布与甜点陈列，整体与{part_summary}的甜美华丽感呼应。",
            "场景设定：法式甜品店内，粉白条纹遮阳棚下，大理石台面上陈列着马卡龙与草莓蛋糕，透过玻璃橱窗可见鹅卵石街道，温馨甜美与{part_summary}相得益彰。",
            "场景设定：维多利亚风格古董洋娃娃店铺内景，木质玻璃柜中陈列精美瓷偶，暖黄灯光洒在丝绒地毯上，复古精致氛围烘托{part_summary}的细节。",
            "场景设定：春日樱花庭院，粉白色花瓣随风飘落，石灯笼与朱红小桥点缀其间，柔和春光洒在{part_summary}上。",
            "场景设定：宫廷风洛可可沙龙，水晶吊灯光辉流转，金色雕花镜框与天鹅绒帷幔环绕，华丽场景衬托{part_summary}的高贵气质。",
            "场景设定：薰衣草花田边缘的白色铁艺凉亭，远处紫色花海与蓝天相接，微风拂过轻纱帷幔，{part_summary}在自然光下熠熠生辉。",
            "场景设定：静谧的欧式图书馆一角，高耸书架与皮质扶手椅营造知性氛围，壁炉火光映照{part_summary}的柔和色泽。",
            "场景设定：童话风糖果屋前，彩色糖果与糖霜装饰的门廊如梦幻世界，{part_summary}与之共同构建甜美童话感。",
        ],
        "characters_1": [
            "主角描述：一位面容精致、气质甜美的少女作为主角，体态轻盈，姿态优雅，发型与发饰呼应裙装细节，整体表现出梦幻、可爱、精心打扮后的大小姐气质。",
            "主角描述：一位温柔甜美的少女，嘴角带着浅浅微笑，举手投足间流露出少女的纯真与优雅，整体造型与{part_summary}完美融合。",
            "主角描述：一位宛如洋娃娃般精致的少女，肌肤白皙，眼神清澈灵动，发型精心打理，整体气质与{part_summary}的细腻华丽相契合。",
            "主角描述：一位自信而甜美的少女，姿态落落大方，目光柔和而笃定，全身穿搭展现出对{part_summary}的独到品味。",
        ],
        "characters_2": [
            "主角描述：两位气质协调的少女同框，一位为主视觉中心，另一位作为陪伴角色，身高和体态略有区分，互动自然，强调甜美洛丽塔姐妹感与精致配饰细节。",
            "主角描述：两位甜美元气的少女并肩而立，一人提裙轻转，一人手持花束微笑注视，互动温馨自然，服装主题统一但细节各有巧思。",
            "主角描述：双人构图，主角在前姿态优雅，配角略微侧身在后，二人穿搭色调呼应，形成和谐的甜美画卷。",
        ],
    },
}

_GENERIC_SCENES = [
    "场景设定：围绕{theme_title}主题，营造统一的时尚插画场景，背景简洁但具有空间层次，让服装和配饰成为视觉重点。",
    "场景设定：精心布置的{theme_title}风格室内场景，自然光从侧窗洒入，道具与陈设衬托{part_summary}的搭配感。",
    "场景设定：午后阳光下的{theme_title}风格户外场景，斑驳树影与柔和光晕笼罩人物，突出{part_summary}的质感与氛围。",
    "场景设定：简约而富有设计感的{theme_title}风格空间，几何线条背景与柔和色彩搭配，{part_summary}成为画面主角。",
    "场景设定：{theme_title}主题的杂志棚拍风格，干净背景配以风格化灯光，重点展示{part_summary}的穿搭效果。",
]

_GENERIC_CHARACTERS_1 = [
    "主角描述：一位主角居中出镜，人物设定与服装风格保持一致，面部、发型、姿态和配饰都围绕{theme_title}主题展开。",
    "主角描述：一位气质出众的少女作为绝对主角，自信且优雅，整体造型与{part_summary}的{theme_title}风格高度统一。",
    "主角描述：一位富有表现力的少女主角，姿态自然松弛，眼神有故事感，全身{part_summary}在{theme_title}主题下呈现最佳效果。",
]

_GENERIC_CHARACTERS_2 = [
    "主角描述：两位主角同框，服装主题一致但姿态与表情略有区分，形成主次层次，表现协调互动与成套穿搭的呼应关系。",
    "主角描述：双人组合，一人正面一人侧立，两人穿搭在{theme_title}主题下各有侧重，相互映衬形成丰富视觉层次。",
]


def _llm_generate_scene_character(
    theme_title: str,
    part_summary: str,
    theme_key: str = "",
    character_count: int = 1,
) -> tuple[str, str] | None:
    """通过 LLM 随机生成与主题切合的场景+角色描述，返回 (scene, character) 或 None。"""
    try:
        from modules.others.api_backend import fetch_llm_json, get_api_config
        cfg = get_api_config(api_type="aigc2d")
        api_base = str(cfg.get("base_url", "") or "").strip()
        api_key = cfg.get("api_key", "")
        if not api_key or not api_base:
            logger.info("场景/LLM: api_key 或 base_url 缺失，回退静态随机池")
            return None
        if "/v1beta/models/" in api_base:
            api_base = api_base.split("/v1beta/models/")[0] + "/v1"

        char_note = "双人" if character_count >= 2 else "单人"
        theme_note = f"主题「{theme_title}」" if theme_key else f"自定义主题「{theme_title}」"
        system = (
            "你是一位专业的时尚插画导演，为少女服装展示设计图面文案。\n"
            "请根据给定的主题、穿搭部位、人物数量，生成一段贴切的场景描述和角色描述。\n"
            f"生成要求：\n"
            f"- 必须严格贴合给定的主题风格，围绕{theme_note}展开\n"
            f"- 本次为{char_note}角色构图\n"
            "- 场景要新颖有创意，不要总写花园茶会，尽量每次不同\n"
            "- 角色描述要生动，展现不同的人物个性与姿态\n"
            "- 输出严格 JSON 格式，不要额外解释"
        )
        user = (
            f"主题: {theme_title}\n"
            f"穿搭部位: {part_summary}\n"
            f"角色人数: {character_count}人\n\n"
            f'请生成 JSON: {{"scene": "场景设定：...", "character": "主角描述：..."}}'
        )
        raw = fetch_llm_json(
            base_url=api_base, api_key=api_key,
            model="gemini-2.5-flash",
            system_prompt=system,
            user_content=user,
            temperature=0.9,
        )
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if not isinstance(data, dict):
            return None
        scene = str(data.get("scene", "") or "").strip()
        character = str(data.get("character", "") or "").strip()
        if not scene or not character:
            return None
        logger.info("场景/LLM 生成成功: scene=%s..., character=%s...", scene[:50], character[:50])
        return scene, character
    except Exception:
        logger.warning("场景/LLM 调用失败，回退静态随机池", exc_info=True)
        return None


def _fallback_random_scene_character(
    theme_title: str,
    part_summary: str,
    theme_key: str,
    character_count: int,
) -> tuple[str, str]:
    """静态随机池兜底。"""
    normalized_count = 2 if character_count >= 2 else 1
    pool = _THEMED_SCENE_CHARACTER_POOLS.get(theme_key)
    if pool:
        scene_templates = pool["scenes"]
        char_templates = pool.get("characters_2" if normalized_count == 2 else "characters_1", _GENERIC_CHARACTERS_1)
    else:
        scene_templates = _GENERIC_SCENES
        char_templates = _GENERIC_CHARACTERS_2 if normalized_count == 2 else _GENERIC_CHARACTERS_1

    scene_text = random.choice(scene_templates).format(
        theme_title=theme_title, part_summary=part_summary,
    )
    character_text = random.choice(char_templates).format(
        theme_title=theme_title, part_summary=part_summary,
    )
    return scene_text, character_text


def generate_random_scene_and_character(
    bundle, theme_profile: ThemeProfile | None, character_count: int = 1
) -> tuple[str, str]:
    """通过 LLM 随机生成与主题切合的场景和角色描述，失败时回退静态随机池。"""
    theme_title = theme_profile.title if theme_profile else "少女时尚"
    theme_key = theme_profile.key if theme_profile else ""
    part_names = [PART_LABELS.get(asset.part, asset.part) for asset in (bundle.assets if bundle else [])]
    part_summary = "、".join(part_names) if part_names else "服饰"

    # 优先 LLM
    result = _llm_generate_scene_character(
        theme_title=theme_title,
        part_summary=part_summary,
        theme_key=theme_key,
        character_count=character_count,
    )
    if result:
        return result

    # 回退
    return _fallback_random_scene_character(
        theme_title=theme_title,
        part_summary=part_summary,
        theme_key=theme_key,
        character_count=character_count,
    )


def build_scene_and_character_description(bundle, theme_profile: ThemeProfile | None, character_count: int = 1) -> tuple[str, str]:
    """生成主题切合的场景描述和角色描述（优先LLM动态生成，失败时回退静态随机池）。"""
    return generate_random_scene_and_character(bundle, theme_profile, character_count)


def build_reference_prompt(base_prompt: str, bundle, scene_text: str = "", character_text: str = "") -> str:
    prompt_lines = [str(base_prompt or "").strip()]
    if scene_text:
        prompt_lines.extend(["", scene_text.strip()])
    if character_text:
        prompt_lines.extend(["", character_text.strip()])
    prompt_lines.extend(["", "服饰参考清单："])
    for asset in bundle.assets:
        prompt_lines.append(f"- {PART_LABELS.get(asset.part, asset.part)}: {asset.item.title}")
        if asset.item.brand:
            prompt_lines.append(f"  品牌: {asset.item.brand}")
        if asset.prompt_hint:
            prompt_lines.append(f"  要点: {asset.prompt_hint}")
    return "\n".join(line for line in prompt_lines if line is not None).strip()


# ======================== 随机构图指令生成 ========================

_COMPOSITION_TYPES = [
    "对角线构图与引导线构图结合",
    "三分法构图，主体偏右黄金分割点",
    "中心对称构图，突出人物主体",
    "S形曲线构图，引导视线沿身体线条流动",
    "三角形构图，姿态形成稳定三角结构",
    "框架式构图，利用环境元素围合主体",
    "对角线构图，身体倾斜形成动态张力",
    "中心构图与散点式背景结合",
    "L形构图，坐姿与道具形成直角布局",
    "螺旋线构图，视线沿肢体与服装纹理盘旋聚焦面部",
]

_CAMERA_ANGLES = [
    "微仰视（Low Angle），突出服装下摆与腿部线条",
    "平视（Eye Level），自然亲切的人物视角",
    "略俯视（High Angle），展现发饰与上半身搭配",
    "微仰视（Low Angle），强调人物气场与裙摆廓形",
    "低角度仰视（Low Angle Shot），从裙摆下方向上延伸，强化连衣裙的蓬松廓形",
    "平视偏侧45度（Three-Quarter View），展现服装正侧两面细节",
    "略俯视（High Angle），营造柔美梦幻的注视感",
    "Dutch Angle 微倾斜，增加画面动感与时尚氛围",
    "过肩视角（Over-the-Shoulder），从配角视角望向主角",
    "正面低角度仰视，腿部向前景延伸，突出鞋袜细节",
]

_DEPTH_OF_FIELD = [
    "浅景深，背景柔化虚化，突出人物主体",
    "大光圈浅景深，前景有少量虚化花草作为画框",
    "中景深，人物清晰，背景轻度虚化保留环境信息",
    "浅景深，只聚焦面部与上半身服装细节",
    "浅景深，焦点落在服装纹理与配饰上",
    "适中景深，人物与近景道具清晰，远景柔和",
    "浅景深配合光斑散景（Bokeh），背景化为梦幻光斑",
]

_LIGHTING = [
    "柔和自然侧光，温暖梦幻",
    "Rembrandt 光，面部三角光区，戏剧而优雅",
    "逆光（Rim Light），发丝边缘光勾勒轮廓，正面补柔光",
    "柔和顶光配合反光板补光，均匀照亮服装细节",
    "午后暖阳斜照，窗光/花架投影洒落人物身上",
    "柔和的蝴蝶光（Butterfly Lighting），鼻下蝶形阴影",
    "双侧夹光（Clamshell Lighting），面部柔和无阴影",
    "黄昏金色时刻（Golden Hour），暖金色调笼罩全身",
    "柔和散射光（Overcast），阴天均匀漫射光",
    "暖色烛光或灯光配合冷色环境光，冷暖对比",
]

_AVOID_ITEMS = [
    "正面平视站立、僵硬姿态、道具遮挡面部",
    "正面直立、动作呆板、头发遮挡眼睛",
    "过于夸张的透视变形、手部遮挡服装细节",
    "面无表情、O字形腿、不自然的手部姿势",
    "服装纹理模糊、配饰丢失、背景喧宾夺主",
    "背面视角、远景全身看不清服装饰品细节",
    "正面呆立、手指蜷缩、发型遮挡衣领",
    "画面过暗看不清服装颜色、过度曝光丢失细节",
    "面部被阴影完全遮盖、手部透视错误",
]

# ---- 角色姿态模板 ----

_POSE_TEMPLATES = [
    # 坐姿类
    (
        "优雅地{seat_pos}于{seat_furniture}，身体微微后仰倚向椅背，"
        "躯干与椅扶手形成斜线；{leg_pose_desc}，充分展示鞋袜细节；"
        "{hand_action_desc}；{hair_desc}，面部完全露出，目光{eye_gaze}，仿佛正专注于手中的细节。"
    ),
    (
        "轻盈地侧坐于{seat_furniture}边缘，上身微微前倾，一手{hand_on}，"
        "另一手{other_hand}；双腿{leg_pose_desc}，脚尖轻点地面，"
        "展现{shoe_highlight}；{hair_desc}，目光{eye_gaze}。"
    ),
    (
        "随意地盘腿坐于{seat_ground}上，裙摆自然铺开呈扇形；"
        "身体微微后仰，双手{hand_action_desc}；"
        "头微侧，{hair_desc}，眼神{eye_gaze}，氛围轻松惬意。"
    ),
    # 站姿类
    (
        "自然站立，身体重心落于{standing_leg}，另一条腿微微弯曲前伸；"
        "上身略微侧转，形成优雅S曲线；{hand_action_desc}；"
        "{hair_desc}，目光{eye_gaze}，全身姿态松弛自然。"
    ),
    (
        "亭亭玉立，双脚{standing_feet}，身体微微后仰有如迎风；"
        "一手{hand_on}，另一手{other_hand}；"
        "{hair_desc}轻轻飘动，目光{eye_gaze}，整体充满动态美感。"
    ),
    (
        "靠立于{lean_target}旁，单手轻扶{lean_target}，"
        "身体自然倾斜形成对角线构图；另一手{other_hand}；"
        "腿部交叉站立，{leg_pose_desc}；{hair_desc}，目光{eye_gaze}。"
    ),
    # 行走/动态类
    (
        "轻盈行走中回眸，身体自然扭转，裙摆{skirt_motion}；"
        "一手{hand_on}，另一手{other_hand}；"
        "{hair_desc}随动作飘逸，目光{eye_gaze}，捕捉瞬间的动态美感。"
    ),
    (
        "微微提裙{walking_desc}，步伐轻盈；身体略微前倾，"
        "仿佛正穿过花园小径；{hand_action_desc}；"
        "{hair_desc}，目光{eye_gaze}，画面充满叙事感。"
    ),
    # 互动/道具类
    (
        "双手{hand_action_desc}；身体微微弯腰前倾，"
        "姿态专注而温柔；{hair_desc}垂落于肩侧，"
        "目光{eye_gaze}，面部表情柔和。"
    ),
    (
        "一手轻{hand_on}，另一手{other_hand}；"
        "身体微转形成优雅的三分面角度，{leg_pose_desc}；"
        "{hair_desc}，目光{eye_gaze}，姿态如同时尚杂志封面。"
    ),
]

# ---- 可替换元素池 ----

_SEAT_POS = ["侧坐", "端坐", "斜倚", "慵懒倚坐", "半坐半靠"]
_SEAT_FURNITURE = [
    "复古古董扶手椅", "洛可可雕花椅", "铁艺花园椅", "天鹅绒沙发",
    "藤编摇椅", "欧式贵妃榻", "木质长椅", "蕾丝布艺椅",
]
_SEAT_GROUND = ["毛绒地毯", "草地上", "花瓣散落的台阶", "柔软的坐垫", "野餐垫"]
_LEAN_TARGET = ["雕花立柱", "大理石栏杆", "花园拱门", "落地窗框", "书架"]

_LEG_POSE = [
    "左腿轻搭在右腿上，小腿沿着椅面方向向前延伸，脚尖微绷",
    "双腿并拢斜放一侧，膝盖轻靠，小腿自然下垂",
    "一腿自然弯曲踩地，另一腿向前舒展，脚尖微微点地",
    "双腿交叉，脚踝处轻叠，小腿向侧面优雅延伸",
    "双腿自然分开微曲，脚尖轻触地面，膝盖内收呈优雅角度",
]
_STANDING_LEG = ["左脚", "右脚"]
_STANDING_FEET = [
    "呈丁字步站立，脚尖微向外展",
    "与肩同宽，自然分开",
    "一前一后错开，重心落于后脚",
    "并拢站立，膝盖微靠",
]

_HAND_ACTION = [
    "右手轻抚膝上一本翻开的复古精装书，指尖轻触书页，手肘自然下垂",
    "左手提着珠珠包随意垂落在身侧裙摆旁",
    "双手交叠轻放于膝上，手指自然舒展",
    "一手轻托下颌，指尖微触脸颊，另一手自然垂放",
    "双手捧着一小束鲜花置于胸前，指尖轻柔环绕花茎",
    "一手拿着复古茶匙轻轻搅动杯中，另一手扶着杯碟",
    "一手握着遮阳伞柄斜靠在肩上，另一手自然垂落",
    "一手调整耳畔碎发，手肘抬起形成优美弧线",
]
_HAND_ON = ["轻扶椅扶手", "搭在膝上", "轻触桌面茶杯", "扶着帽檐", "拎着小手提包"]
_OTHER_HAND = [
    "自然垂落于身侧", "轻放在裙摆褶皱处", "手指微蜷搭在膝上",
    "轻轻捻起裙角", "手背轻贴腰侧", "搭在椅背上",
]

_HAIR = [
    "粉色长发柔软散落于肩头与背后",
    "金色波浪卷发自然披散在背后",
    "黑色长直发柔顺垂落腰际",
    "棕色微卷发蓬松垂于肩侧",
    "银色长发光泽流转铺于背后",
    "栗色双马尾自然垂落于胸前两侧",
    "浅紫色长发蓬松散开",
    "蓝色渐变长发飘逸",
]
_EYE_GAZE = [
    "温柔低垂", "柔和望向远方", "轻轻瞥向镜头", "宁静注视前方",
    "含情脉脉", "略带羞涩地偏开", "明亮有神地注视", "慵懒半阖",
]

_SHOE_HIGHLIGHT = ["玛丽珍鞋的搭扣细节", "高跟鞋的优雅弧度", "芭蕾平底鞋的蝴蝶结", "靴子的蕾丝系带", "凉鞋的精致绑带"]
_SKIRT_MOTION = ["如花朵般展开", "轻盈飘扬", "柔美摆动", "自然旋开", "如波浪般飘舞"]
_WALKING_DESC = ["缓步前行", "小碎步快走", "轻跃而下台阶", "踮脚旋转", "信步游走"]

# ---- 构图 → 推荐画幅比例映射 ----
# 每种构图类型对应 2-3 个合适的比例（含横纵版），随机选取
_COMPOSITION_RATIO_MAP: dict[str, list[str]] = {
    "对角线构图与引导线构图结合": ["2:3", "3:4", "9:16"],
    "三分法构图，主体偏右黄金分割点": ["3:4", "2:3", "16:9"],
    "中心对称构图，突出人物主体": ["1:1", "3:4", "4:3"],
    "S形曲线构图，引导视线沿身体线条流动": ["2:3", "3:4"],
    "三角形构图，姿态形成稳定三角结构": ["3:4", "2:3", "1:1"],
    "框架式构图，利用环境元素围合主体": ["3:4", "1:1", "2:3"],
    "对角线构图，身体倾斜形成动态张力": ["2:3", "16:9", "3:4"],
    "中心构图与散点式背景结合": ["1:1", "3:4"],
    "L形构图，坐姿与道具形成直角布局": ["16:9", "3:2", "3:4"],
    "螺旋线构图，视线沿肢体与服装纹理盘旋聚焦面部": ["2:3", "3:4", "9:16"],
}


def generate_random_composition() -> tuple[str, str]:
    """随机生成一套人物动作构图 instructions，返回 (文本, 推荐画幅比例)。"""
    composition = random.choice(_COMPOSITION_TYPES)
    camera = random.choice(_CAMERA_ANGLES)
    depth = random.choice(_DEPTH_OF_FIELD)
    lighting = random.choice(_LIGHTING)
    avoid = random.choice(_AVOID_ITEMS)

    # 根据构图类型选择合适的画幅比例
    ratio_pool = _COMPOSITION_RATIO_MAP.get(composition, ["3:4", "2:3", "1:1"])
    aspect_ratio = random.choice(ratio_pool)

    # 随机选择一个姿态模板并填充随机元素
    pose_template = random.choice(_POSE_TEMPLATES)
    pose = pose_template.format(
        seat_pos=random.choice(_SEAT_POS),
        seat_furniture=random.choice(_SEAT_FURNITURE),
        seat_ground=random.choice(_SEAT_GROUND),
        lean_target=random.choice(_LEAN_TARGET),
        leg_pose_desc=random.choice(_LEG_POSE),
        standing_leg=random.choice(_STANDING_LEG),
        standing_feet=random.choice(_STANDING_FEET),
        hand_action_desc=random.choice(_HAND_ACTION),
        hand_on=random.choice(_HAND_ON),
        other_hand=random.choice(_OTHER_HAND),
        hair_desc=random.choice(_HAIR),
        eye_gaze=random.choice(_EYE_GAZE),
        shoe_highlight=random.choice(_SHOE_HIGHLIGHT),
        skirt_motion=random.choice(_SKIRT_MOTION),
        walking_desc=random.choice(_WALKING_DESC),
    )

    lines = [
        f" - 构图类型: {composition}",
        f" - 镜头角度: {camera}",
        f" - 角色姿态: {pose}",
        f" - 景深: {depth}",
        f" - 光线: {lighting}",
        f" - 需避免: {avoid}",
    ]
    return "\n".join(lines), aspect_ratio
