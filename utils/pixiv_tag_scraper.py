"""
Pixiv 热门标签抓取器

抓取策略（按优先级）：
1. 尝试从 Pixiv 公开 API 抓取热门标签及其作品数
2. 使用内置的精选热门标签列表（~1000个）作为兜底，含估算作品数

输出格式（JSON）：
[
    {
        "tag": "女の子",              # 日文 Pixiv 标签
        "en_keywords": ["girl", "female"],  # 英文关键词，用于与 WD14 预测结果匹配
        "category": "character",       # 分类
        "count": 8500000               # 该标签在 Pixiv 上的作品数（精确或估算）
    },
    ...
]
"""

import os
import json
import re
import urllib.request
import urllib.error
import ssl
import time
import math
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data")
CACHE_FILE = os.path.join(CACHE_DIR, "pixiv_tags_cache.json")

# ============================================================
# 内置精选热门 Pixiv 标签列表（兜底数据）
# 格式: (日文标签, [英文关键词], 分类, 估算作品数)
# count 基于 2025-2026 年 Pixiv 上各标签大致作品数的估算
# ============================================================
_FALLBACK_TAGS: list[tuple[str, list[str], str, int]] = [
    # ============ 超热门标签（千万~百万级） ============
    ("女の子", ["girl", "female", "1girl"], "character", 8500000),
    ("魔法少女", ["magical_girl"], "character", 70000),
    ("百合", ["yuri", "girls_love"], "character", 65000),

    # ============ 人物核心描述（百万~十万级） ============
    ("男の子", ["boy", "male", "1boy"], "character", 1800000),
    ("美少女", ["beautiful_girl", "bishoujo"], "character", 1200000),
    ("少女", ["girl", "shoujo", "young_girl"], "character", 900000),
    ("女の子二人", ["2girls", "two_girls"], "character", 700000),
    ("二人", ["2girls", "2boys", "couple", "pair", "duo", "two_people"], "character", 500000),
    ("三人", ["3girls", "3boys", "trio", "three_people"], "character", 350000),
    ("複数", ["multiple_girls", "multiple", "group"], "character", 250000),
    ("お姉さん", ["oneesan", "older_sister", "mature_woman"], "character", 200000),
    ("幼女", ["young_girl", "little_girl", "loli"], "character", 180000),
    ("ロリ", ["loli"], "character", 160000),
    ("ショタ", ["shota", "young_boy"], "character", 140000),
    ("ちびキャラ", ["chibi", "super_deformed"], "character", 120000),
    ("ケモノ", ["kemono", "furry", "animal_ears"], "character", 100000),
    ("獣人", ["furry", "kemono", "beastman"], "character", 80000),
    ("人外", ["non-human", "monster_girl", "inhuman"], "character", 70000),
    ("天使", ["angel", "wings"], "character", 65000),
    ("悪魔", ["demon", "devil", "succubus"], "character", 60000),
    ("エルフ", ["elf", "pointy_ears"], "character", 55000),
    ("吸血鬼", ["vampire"], "character", 50000),
    ("人魚", ["mermaid"], "character", 45000),
    ("ロボット", ["robot", "android", "mecha_musume"], "character", 40000),
    ("メカ", ["mecha", "robot"], "character", 35000),
    ("ドラゴン", ["dragon"], "character", 30000),
    ("妖怪", ["youkai", "yokai"], "character", 28000),
    ("猫", ["cat", "neko"], "character", 25000),
    ("狐", ["fox", "kitsune"], "character", 22000),
    ("狼", ["wolf", "ookami"], "character", 20000),
    ("犬", ["dog", "inu"], "character", 18000),

    # ============ 发色（十万~万级） ============
    ("黒髪", ["black_hair", "dark_hair"], "hairstyle", 1800000),
    ("白髪", ["white_hair", "silver_hair"], "hairstyle", 1200000),
    ("金髪", ["blonde_hair", "blond_hair", "yellow_hair"], "hairstyle", 1000000),
    ("銀髪", ["silver_hair", "grey_hair", "gray_hair"], "hairstyle", 800000),
    ("赤髪", ["red_hair"], "hairstyle", 500000),
    ("青髪", ["blue_hair"], "hairstyle", 450000),
    ("ピンク髪", ["pink_hair"], "hairstyle", 400000),
    ("茶髪", ["brown_hair"], "hairstyle", 350000),
    ("紫髪", ["purple_hair"], "hairstyle", 300000),
    ("緑髪", ["green_hair"], "hairstyle", 250000),
    ("オレンジ髪", ["orange_hair"], "hairstyle", 200000),
    ("水色髪", ["aqua_hair", "light_blue_hair"], "hairstyle", 180000),
    ("桃髪", ["peach_hair"], "hairstyle", 150000),
    ("グラデーション髪", ["gradient_hair", "multicolored_hair"], "hairstyle", 120000),
    ("メッシュ", ["hair_streaks", "colored_streaks", "two-tone_hair"], "hairstyle", 100000),
    ("インナーカラー", ["inner_colored_hair"], "hairstyle", 80000),

    # ============ 发型（十万~万级） ============
    ("ロングヘアー", ["long_hair", "very_long_hair"], "hairstyle", 2000000),
    ("ショートヘア", ["short_hair"], "hairstyle", 1500000),
    ("ポニーテール", ["ponytail"], "hairstyle", 1200000),
    ("ツインテール", ["twintails"], "hairstyle", 800000),
    ("セミロング", ["medium_hair"], "hairstyle", 600000),
    ("おさげ", ["braid", "braided_hair"], "hairstyle", 400000),
    ("三つ編み", ["braid", "braided_hair", "twin_braids"], "hairstyle", 350000),
    ("ボブ", ["bob_cut"], "hairstyle", 300000),
    ("姫カット", ["hime_cut"], "hairstyle", 250000),
    ("サイドテール", ["side_ponytail"], "hairstyle", 200000),
    ("アホ毛", ["ahoge", "antenna_hair"], "hairstyle", 180000),
    ("縦ロール", ["drill_hair", "vertical_roll"], "hairstyle", 150000),
    ("ストレート", ["straight_hair"], "hairstyle", 120000),
    ("ウェーブ", ["wavy_hair"], "hairstyle", 100000),
    ("ボリューム", ["voluminous_hair"], "hairstyle", 80000),
    ("お団子", ["bun", "hair_bun"], "hairstyle", 70000),
    ("前髪", ["bangs", "fringe"], "hairstyle", 60000),
    ("ぱっつん前髪", ["blunt_bangs", "straight_bangs"], "hairstyle", 50000),
    ("ハーフアップ", ["half_updo", "half_up"], "hairstyle", 45000),
    ("シニヨン", ["chignon", "low_bun"], "hairstyle", 40000),
    ("ツーサイドアップ", ["two_side_up"], "hairstyle", 35000),
    ("くせ毛", ["curly_hair", "frizzy_hair"], "hairstyle", 30000),
    ("編み込み", ["braided_hair", "french_braid"], "hairstyle", 28000),
    ("ショートボブ", ["short_bob"], "hairstyle", 25000),
    ("マッシュ", ["mushroom_cut", "bowl_cut"], "hairstyle", 20000),
    ("おかっぱ", ["bob", "pageboy"], "hairstyle", 18000),
    ("ベリーショート", ["very_short_hair", "pixie_cut"], "hairstyle", 15000),

    # ============ 瞳色（十万~万级） ============
    ("青い目", ["blue_eyes"], "character", 800000),
    ("赤い目", ["red_eyes"], "character", 700000),
    ("緑の目", ["green_eyes"], "character", 400000),
    ("金色の目", ["golden_eyes", "yellow_eyes"], "character", 300000),
    ("紫色の目", ["purple_eyes"], "character", 250000),
    ("オッドアイ", ["heterochromia", "different_eyes"], "character", 200000),
    ("茶色の目", ["brown_eyes"], "character", 150000),
    ("銀の目", ["silver_eyes", "grey_eyes"], "character", 100000),
    ("ピンクの目", ["pink_eyes"], "character", 80000),
    ("黒い目", ["black_eyes", "dark_eyes"], "character", 70000),

    # ============ 服装 - 制服 / 学校系（百万~万级） ============
    ("制服", ["school_uniform", "seifuku", "uniform"], "outfit", 1500000),
    ("セーラー服", ["serafuku", "sailor_uniform", "sailor_collar"], "outfit", 800000),
    ("ブレザー", ["blazer", "school_blazer"], "outfit", 600000),
    ("体操服", ["gym_uniform", "bloomers"], "outfit", 200000),
    ("ブルマ", ["bloomers"], "outfit", 180000),
    ("スク水", ["school_swimsuit", "sukumizu"], "outfit", 150000),
    ("学生服", ["school_uniform", "gakuran"], "outfit", 120000),
    ("学ラン", ["gakuran", "male_school_uniform"], "outfit", 80000),
    ("スモック", ["smock"], "outfit", 40000),
    ("エプロン", ["apron"], "outfit", 35000),
    ("白衣", ["lab_coat", "white_coat"], "outfit", 30000),

    # ============ 服装 - 和风系（百万~万级） ============
    ("和服", ["kimono", "japanese_clothes", "wafuku"], "outfit", 800000),
    ("着物", ["kimono"], "outfit", 600000),
    ("浴衣", ["yukata"], "outfit", 400000),
    ("巫女服", ["miko", "shrine_maiden"], "outfit", 250000),
    ("袴", ["hakama"], "outfit", 150000),
    ("振袖", ["furisode", "long_sleeved_kimono"], "outfit", 100000),
    ("羽織", ["haori"], "outfit", 80000),
    ("十二単", ["juunihitoe", "twelve-layered_kimono"], "outfit", 30000),
    ("忍者装束", ["ninja_outfit"], "outfit", 25000),
    ("甲冑", ["samurai_armor", "yoroi"], "outfit", 20000),

    # ============ 服装 - 日常 / 时尚系（百万~万级） ============
    ("ドレス", ["dress"], "outfit", 700000),
    ("ワンピース", ["one_piece", "onepiece"], "outfit", 500000),
    ("スカート", ["skirt"], "outfit", 400000),
    ("ミニスカート", ["miniskirt", "short_skirt"], "outfit", 300000),
    ("Tシャツ", ["t-shirt", "tshirt"], "outfit", 250000),
    ("パーカー", ["hoodie", "hood"], "outfit", 200000),
    ("ジャケット", ["jacket"], "outfit", 180000),
    ("コート", ["coat"], "outfit", 150000),
    ("セーター", ["sweater"], "outfit", 120000),
    ("カーディガン", ["cardigan"], "outfit", 100000),
    ("ジーンズ", ["jeans"], "outfit", 90000),
    ("ショートパンツ", ["shorts", "short_shorts"], "outfit", 80000),
    ("キャミソール", ["camisole"], "outfit", 70000),
    ("タンクトップ", ["tank_top"], "outfit", 60000),
    ("ベスト", ["vest"], "outfit", 50000),
    ("Yシャツ", ["dress_shirt", "button-up_shirt"], "outfit", 45000),
    ("ポロシャツ", ["polo_shirt"], "outfit", 40000),
    ("オーバーオール", ["overalls"], "outfit", 35000),
    ("サロペット", ["salopette", "jumper_skirt"], "outfit", 30000),
    ("スーツ", ["suit"], "outfit", 28000),
    ("タキシード", ["tuxedo"], "outfit", 25000),
    ("スウェット", ["sweatshirt", "sweatpants"], "outfit", 20000),
    ("ジャージ", ["jersey", "tracksuit"], "outfit", 18000),
    ("パーカードレス", ["hoodie_dress"], "outfit", 15000),
    ("デニム", ["denim"], "outfit", 12000),

    # ============ 服装 - 洛丽塔/甜系（十万~万级） ============
    ("ゴシック", ["gothic", "gothic_lolita"], "outfit", 200000),
    ("ロリータ", ["lolita_fashion", "lolita"], "outfit", 180000),
    ("ゴスロリ", ["gothic_lolita"], "outfit", 120000),
    ("甘ロリ", ["sweet_lolita"], "outfit", 80000),
    ("クラシカルロリータ", ["classical_lolita"], "outfit", 40000),
    ("パンクロリータ", ["punk_lolita"], "outfit", 25000),

    # ============ 服装 - 职业/特殊（十万~万级） ============
    ("メイド服", ["maid", "maid_uniform", "maid_headdress"], "outfit", 300000),
    ("ナース服", ["nurse", "nurse_uniform"], "outfit", 180000),
    ("軍服", ["military_uniform"], "outfit", 150000),
    ("警察", ["police_uniform"], "outfit", 80000),
    ("消防士", ["firefighter"], "outfit", 30000),
    ("パイロット", ["pilot_uniform"], "outfit", 25000),
    ("レースクイーン", ["race_queen"], "outfit", 20000),
    ("CA", ["flight_attendant", "cabin_attendant"], "outfit", 18000),
    ("ウェイトレス", ["waitress"], "outfit", 15000),

    # ============ 服装 - 泳装（百万~万级） ============
    ("水着", ["swimsuit", "bikini"], "outfit", 600000),
    ("ビキニ", ["bikini", "two-piece_swimsuit"], "outfit", 400000),
    ("競泳水着", ["competition_swimsuit", "racing_swimsuit", "one-piece_swimsuit"], "outfit", 150000),
    ("ワンピース水着", ["one-piece_swimsuit"], "outfit", 100000),
    ("フリル水着", ["frilled_swimsuit"], "outfit", 50000),
    ("タンキニ", ["tankini"], "outfit", 25000),

    # ============ 服装 - 民族/文化系（十万~万级） ============
    ("チャイナドレス", ["china_dress", "cheongsam", "qipao"], "outfit", 250000),
    ("漢服", ["hanfu", "chinese_traditional"], "outfit", 80000),
    ("韓服", ["hanbok", "korean_traditional"], "outfit", 50000),
    ("アラビアン", ["arabian", "belly_dancer"], "outfit", 40000),
    ("ウェスタン", ["western", "cowboy", "cowgirl"], "outfit", 30000),
    ("バイキング", ["viking"], "outfit", 20000),

    # ============ 服装 - 特殊衣装（十万~万级） ============
    ("ウェディングドレス", ["wedding_dress", "bride"], "outfit", 180000),
    ("パジャマ", ["pajamas", "sleepwear"], "outfit", 120000),
    ("鎧", ["armor"], "outfit", 100000),
    ("バニーガール", ["bunny_girl", "bunny_suit"], "outfit", 80000),
    ("チアガール", ["cheerleader"], "outfit", 60000),
    ("アイドル衣装", ["idol_costume"], "outfit", 50000),
    ("魔法少女衣装", ["magical_girl_costume"], "outfit", 45000),
    ("巫女", ["miko", "shrine_maiden"], "outfit", 40000),
    ("シスター", ["nun", "sister"], "outfit", 35000),
    ("ダンサー", ["dancer"], "outfit", 30000),
    ("バレリーナ", ["ballerina", "ballet"], "outfit", 28000),
    ("サンタコス", ["santa_costume"], "outfit", 25000),
    ("ハロウィン衣装", ["halloween_costume"], "outfit", 20000),
    ("着ぐるみ", ["kigurumi", "costume"], "outfit", 18000),
    ("制服風", ["uniform_style"], "outfit", 15000),
    ("私服", ["casual_clothes", "civilian_clothes"], "outfit", 12000),

    # ============ 服装细节 / 领口 / 袖型（十万~万级） ============
    ("フリル", ["frills", "frilled"], "outfit", 400000),
    ("レース", ["lace"], "outfit", 350000),
    ("刺繍", ["embroidery", "embroidered"], "outfit", 150000),
    ("リボン", ["ribbon"], "outfit", 120000),
    ("オフショルダー", ["off_shoulder", "bare_shoulders"], "outfit", 100000),
    ("ノースリーブ", ["sleeveless"], "outfit", 80000),
    ("半袖", ["short_sleeves"], "outfit", 70000),
    ("長袖", ["long_sleeves"], "outfit", 60000),
    ("七分袖", ["three-quarter_sleeves"], "outfit", 30000),
    ("透け", ["see-through", "transparent", "sheer"], "outfit", 50000),
    ("背中開き", ["backless", "open_back"], "outfit", 45000),
    ("肩出し", ["bare_shoulders", "shoulder_reveal"], "outfit", 40000),
    ("へそ出し", ["midriff", "navel_reveal", "crop_top"], "outfit", 35000),
    ("サイドスリット", ["side_slit"], "outfit", 30000),
    ("Vネック", ["v_neck"], "outfit", 25000),
    ("タートルネック", ["turtleneck"], "outfit", 20000),
    ("フード", ["hood"], "outfit", 18000),
    ("ボタン", ["buttons"], "outfit", 15000),
    ("ファスナー", ["zipper"], "outfit", 12000),
    ("ベルト", ["belt"], "outfit", 10000),
    ("サスペンダー", ["suspenders"], "outfit", 8000),
    ("ポケット", ["pockets"], "outfit", 6000),

    # ============ 袜子（十万~万级） ============
    ("ニーソックス", ["kneehighs", "knee_socks"], "accessory", 500000),
    ("オーバーニーソックス", ["over-knee_socks", "thighhighs"], "accessory", 400000),
    ("サイハイソックス", ["thighhighs"], "accessory", 300000),
    ("ストッキング", ["stockings", "pantyhose"], "accessory", 200000),
    ("ガーターベルト", ["garter_belt"], "accessory", 150000),
    ("絶対領域", ["zettai_ryouiki", "thigh_gap", "absolute_territory"], "accessory", 120000),
    ("くるぶしソックス", ["ankle_socks"], "accessory", 60000),
    ("フリルソックス", ["frilled_socks"], "accessory", 50000),
    ("レースソックス", ["lace_socks"], "accessory", 40000),
    ("ルーズソックス", ["loose_socks", "slouch_socks"], "accessory", 35000),
    ("タイツ", ["tights"], "accessory", 30000),
    ("網タイツ", ["fishnet_tights", "net_tights"], "accessory", 25000),
    ("縞ニーソ", ["striped_kneehighs", "striped_socks"], "accessory", 20000),

    # ============ 鞋子（十万~万级） ============
    ("ハイヒール", ["high_heels"], "accessory", 150000),
    ("パンプス", ["pumps", "high_heels"], "accessory", 100000),
    ("ローファー", ["loafers"], "accessory", 80000),
    ("スニーカー", ["sneakers"], "accessory", 70000),
    ("ブーツ", ["boots"], "accessory", 60000),
    ("編み上げブーツ", ["lace-up_boots"], "accessory", 50000),
    ("ロングブーツ", ["long_boots", "knee_boots"], "accessory", 45000),
    ("サンダル", ["sandals"], "accessory", 40000),
    ("ミュール", ["mules"], "accessory", 25000),
    ("バレエシューズ", ["ballet_shoes", "flats"], "accessory", 20000),
    ("下駄", ["geta", "wooden_sandals"], "accessory", 18000),
    ("草履", ["zori", "japanese_sandals"], "accessory", 15000),
    ("裸足", ["barefoot"], "accessory", 12000),
    ("足袋", ["tabi", "split-toe_socks"], "accessory", 8000),

    # ============ 头部配饰（十万~万级） ============
    ("猫耳", ["cat_ears", "nekomimi"], "accessory", 350000),
    ("獣耳", ["animal_ears", "kemonomimi"], "accessory", 250000),
    ("眼鏡", ["glasses", "eyeglasses"], "accessory", 200000),
    ("メガネ", ["glasses"], "accessory", 180000),
    ("サングラス", ["sunglasses"], "accessory", 100000),
    ("ヘッドフォン", ["headphones"], "accessory", 80000),
    ("帽子", ["hat"], "accessory", 70000),
    ("ベレー帽", ["beret"], "accessory", 50000),
    ("麦わら帽子", ["straw_hat"], "accessory", 40000),
    ("カチューシャ", ["hairband", "headband"], "accessory", 35000),
    ("ヘッドドレス", ["headdress"], "accessory", 30000),
    ("花飾り", ["flower_ornament", "hair_flower"], "accessory", 25000),
    ("リボンの髪飾り", ["hair_ribbon"], "accessory", 20000),
    ("ティアラ", ["tiara"], "accessory", 15000),
    ("王冠", ["crown"], "accessory", 12000),
    ("ベール", ["veil"], "accessory", 10000),
    ("カチューシャリボン", ["ribbon_headband"], "accessory", 8000),
    ("ヘアピン", ["hairpin"], "accessory", 7000),
    ("バレッタ", ["barrette"], "accessory", 6000),
    ("シュシュ", ["scrunchy", "hair_scrunchie"], "accessory", 5000),
    ("イヤーマフ", ["earmuffs"], "accessory", 5000),
    ("ゴーグル", ["goggles"], "accessory", 4000),

    # ============ 饰品 / 珠宝（十万~万级） ============
    ("ネックレス", ["necklace"], "accessory", 80000),
    ("チョーカー", ["choker"], "accessory", 60000),
    ("イヤリング", ["earrings"], "accessory", 50000),
    ("ピアス", ["piercing", "ear_piercing"], "accessory", 40000),
    ("指輪", ["ring"], "accessory", 35000),
    ("ブレスレット", ["bracelet"], "accessory", 30000),
    ("腕時計", ["wristwatch"], "accessory", 25000),
    ("アンクレット", ["anklet"], "accessory", 20000),
    ("ブローチ", ["brooch"], "accessory", 15000),
    ("ロザリオ", ["rosary"], "accessory", 10000),
    ("十字架", ["cross", "crucifix"], "accessory", 8000),
    ("鈴", ["bell"], "accessory", 6000),

    # ============ 手套 / 手部（十万~万级） ============
    ("手袋", ["gloves"], "accessory", 150000),
    ("レースグローブ", ["lace_gloves"], "accessory", 80000),
    ("フィンガーレスグローブ", ["fingerless_gloves"], "accessory", 50000),
    ("アームカバー", ["arm_covers", "arm_warmers"], "accessory", 40000),
    ("リストバンド", ["wristband"], "accessory", 20000),
    ("包帯", ["bandages", "bandaged_arms"], "accessory", 15000),
    ("ネイル", ["nail_polish", "nails"], "accessory", 12000),
    ("指ぬきグローブ", ["fingerless_gloves"], "accessory", 10000),

    # ============ 围巾 / 披肩（十万~万级） ============
    ("マフラー", ["scarf", "muffler"], "accessory", 120000),
    ("マント", ["cape", "cloak"], "accessory", 80000),
    ("ショール", ["shawl"], "accessory", 40000),
    ("ストール", ["stole", "wrap"], "accessory", 25000),
    ("ポンチョ", ["poncho"], "accessory", 15000),
    ("ケープ", ["cape"], "accessory", 12000),

    # ============ 道具 / 武器（十万~万级） ============
    ("刀", ["katana", "sword", "japanese_sword"], "accessory", 200000),
    ("武器", ["weapon"], "accessory", 150000),
    ("剣", ["sword"], "accessory", 100000),
    ("銃", ["gun", "firearm"], "accessory", 80000),
    ("弓", ["bow"], "accessory", 50000),
    ("盾", ["shield"], "accessory", 40000),
    ("ぬいぐるみ", ["stuffed_toy", "plushie"], "accessory", 35000),
    ("傘", ["umbrella"], "accessory", 30000),
    ("杖", ["staff", "wand"], "accessory", 25000),
    ("本", ["book"], "accessory", 20000),
    ("花束", ["bouquet", "flower_bouquet"], "accessory", 18000),
    ("楽器", ["musical_instrument"], "accessory", 15000),
    ("ギター", ["guitar"], "accessory", 12000),
    ("ピアノ", ["piano"], "accessory", 10000),
    ("バイオリン", ["violin"], "accessory", 8000),
    ("トランプ", ["playing_cards"], "accessory", 6000),
    ("スマホ", ["smartphone", "phone"], "accessory", 5000),
    ("タブレット", ["tablet"], "accessory", 4000),
    ("カメラ", ["camera"], "accessory", 3500),
    ("双眼鏡", ["binoculars"], "accessory", 3000),
    ("ランタン", ["lantern"], "accessory", 2500),
    ("鏡", ["mirror"], "accessory", 2000),
    ("人形", ["doll", "puppet"], "accessory", 2000),
    ("風船", ["balloon"], "accessory", 1500),
    ("鎖", ["chains"], "accessory", 1200),

    # ============ 食物 / 饮品（十万~万级） ============
    ("食べ物", ["food"], "accessory", 80000),
    ("ケーキ", ["cake"], "accessory", 50000),
    ("アイス", ["ice_cream"], "accessory", 40000),
    ("お菓子", ["candy", "sweets", "snack"], "accessory", 35000),
    ("マカロン", ["macaron"], "accessory", 25000),
    ("チョコレート", ["chocolate"], "accessory", 20000),
    ("飴", ["candy", "lollipop"], "accessory", 18000),
    ("りんご飴", ["candy_apple"], "accessory", 15000),
    ("ドーナツ", ["donut", "doughnut"], "accessory", 12000),
    ("クッキー", ["cookie"], "accessory", 10000),
    ("たい焼き", ["taiyaki"], "accessory", 8000),
    ("おにぎり", ["rice_ball", "onigiri"], "accessory", 6000),
    ("ラーメン", ["ramen"], "accessory", 5000),
    ("コーヒー", ["coffee"], "accessory", 4000),
    ("紅茶", ["tea", "black_tea"], "accessory", 3500),
    ("ジュース", ["juice"], "accessory", 3000),
    ("ワイン", ["wine"], "accessory", 2500),

    # ============ 动作 / 姿势（十万~万级） ============
    ("座る", ["sitting"], "action", 400000),
    ("立つ", ["standing"], "action", 300000),
    ("寝そべる", ["lying", "lying_down"], "action", 200000),
    ("走る", ["running"], "action", 150000),
    ("ジャンプ", ["jumping"], "action", 120000),
    ("振り返る", ["looking_back", "turning_around"], "action", 100000),
    ("手を伸ばす", ["reaching_out"], "action", 80000),
    ("片足立ち", ["one_foot_raised", "standing_on_one_leg"], "action", 60000),
    ("腕組み", ["crossed_arms", "arms_crossed"], "action", 50000),
    ("指をさす", ["pointing"], "action", 45000),
    ("Vサイン", ["v_sign", "peace_sign"], "action", 40000),
    ("ピース", ["peace_sign", "v_sign"], "action", 35000),
    ("敬礼", ["salute"], "action", 30000),
    ("お辞儀", ["bowing"], "action", 25000),
    ("抱きしめる", ["hugging", "embracing"], "action", 20000),
    ("手をつなぐ", ["holding_hands"], "action", 18000),
    ("踊る", ["dancing"], "action", 15000),
    ("泳ぐ", ["swimming"], "action", 12000),
    ("飛ぶ", ["flying", "floating"], "action", 10000),
    ("倒れる", ["falling"], "action", 8000),
    ("もたれる", ["leaning"], "action", 7000),
    ("ぶら下がる", ["hanging"], "action", 6000),
    ("逆立ち", ["handstand"], "action", 5000),
    ("プリーツダンス", ["dance", "dancing"], "action", 4000),
    ("膝立ち", ["kneeling"], "action", 3500),
    ("ひざまくら", ["lap_pillow"], "action", 3000),
    ("あごのせ", ["chin_rest", "hand_on_chin"], "action", 2500),
    ("肩車", ["shoulder_ride", "piggyback"], "action", 2000),
    ("お姫様抱っこ", ["princess_carry", "bridal_carry"], "action", 1500),
    ("逆さま", ["upside-down"], "action", 1000),

    # ============ 表情（十万~万级） ============
    ("笑顔", ["smile", "smiling"], "expression", 800000),
    ("微笑み", ["smile", "gentle_smile"], "expression", 500000),
    ("照れ", ["blush", "embarrassed"], "expression", 400000),
    ("泣く", ["crying", "tears"], "expression", 300000),
    ("怒り", ["angry", "anger"], "expression", 200000),
    ("驚き", ["surprised", "shock"], "expression", 180000),
    ("無表情", ["expressionless", "blank_expression"], "expression", 150000),
    ("ウインク", ["wink", "winking"], "expression", 120000),
    ("ジト目", ["jito", "half-closed_eyes", "annoyed"], "expression", 100000),
    ("あくび", ["yawning"], "expression", 80000),
    ("舌出し", ["tongue_out", "blep"], "expression", 60000),
    ("赤面", ["blush", "blushing", "red_face"], "expression", 50000),
    ("にっこり", ["grin", "beaming"], "expression", 45000),
    ("にやにや", ["smirk", "grin"], "expression", 40000),
    ("困り顔", ["troubled_expression", "worried"], "expression", 35000),
    ("恐怖", ["scared", "frightened", "terrified"], "expression", 30000),
    ("嫌悪", ["disgust"], "expression", 25000),
    ("退屈", ["bored", "boredom"], "expression", 20000),
    ("眠そう", ["sleepy", "tired"], "expression", 18000),
    ("慌てる", ["panicked", "flustered"], "expression", 15000),
    ("ドヤ顔", ["smug_face", "smug"], "expression", 12000),
    ("ぷんすか", ["pouting"], "expression", 10000),
    ("キメ顔", ["determined_expression"], "expression", 8000),
    ("恍惚", ["ecstasy", "enraptured"], "expression", 6000),
    ("真顔", ["serious_face", "straight_face"], "expression", 5000),
    ("無垢", ["innocent_expression"], "expression", 4000),

    # ============ 构图/视角（十万~万级） ============
    ("見つめ", ["looking_at_viewer"], "other", 500000),
    ("アップ", ["close-up", "portrait"], "other", 300000),
    ("全身", ["full_body"], "other", 250000),
    ("俯瞰", ["bird's_eye_view", "from_above"], "other", 150000),
    ("あおり", ["from_below", "low_angle"], "other", 120000),
    ("バストアップ", ["bust_shot", "chest_up"], "other", 100000),
    ("横顔", ["profile", "side_view"], "other", 80000),
    ("後ろ姿", ["from_behind"], "other", 60000),
    ("斜め", ["diagonal_composition"], "other", 40000),
    ("接写", ["extreme_close-up", "macro"], "other", 30000),
    ("寄り", ["close_shot"], "other", 25000),
    ("引き", ["wide_shot", "long_shot"], "other", 20000),

    # ============ 背景 / 环境（十万~万级） ============
    ("背景あり", ["background", "scenery"], "other", 300000),
    ("シンプル背景", ["simple_background"], "other", 200000),
    ("白背景", ["white_background"], "other", 150000),
    ("空", ["sky"], "other", 120000),
    ("星空", ["starry_sky", "night_sky"], "other", 100000),
    ("夜景", ["night_scape", "night_view"], "other", 80000),
    ("海", ["ocean", "sea", "beach"], "other", 70000),
    ("花", ["flower", "flowers"], "other", 60000),
    ("桜", ["sakura", "cherry_blossoms"], "other", 55000),
    ("雨", ["rain"], "other", 50000),
    ("雪", ["snow"], "other", 45000),
    ("夕焼け", ["sunset", "dusk"], "other", 40000),
    ("水中", ["underwater"], "other", 35000),
    ("教室", ["classroom"], "other", 30000),
    ("森", ["forest", "woods"], "other", 28000),
    ("公園", ["park"], "other", 25000),
    ("街", ["city", "street", "town"], "other", 22000),
    ("廃墟", ["ruins", "abandoned"], "other", 20000),
    ("部屋", ["room", "indoor"], "other", 18000),
    ("カフェ", ["cafe", "coffee_shop"], "other", 15000),
    ("神社", ["shrine", "shinto_shrine"], "other", 12000),
    ("お城", ["castle"], "other", 10000),
    ("遊園地", ["amusement_park", "theme_park"], "other", 8000),
    ("屋上", ["rooftop"], "other", 7000),
    ("図書館", ["library"], "other", 6000),
    ("温泉", ["hot_spring", "onsen"], "other", 5000),
    ("病院", ["hospital"], "other", 4500),
    ("教会", ["church", "chapel"], "other", 4000),
    ("ステージ", ["stage"], "other", 3500),
    ("宇宙", ["space"], "other", 3000),
    ("月", ["moon", "lunar"], "other", 2800),
    ("太陽", ["sun", "sunlight"], "other", 2500),
    ("雲", ["clouds"], "other", 2000),
    ("虹", ["rainbow"], "other", 1800),
    ("雷", ["lightning", "thunder"], "other", 1500),
    ("霧", ["fog", "mist"], "other", 1200),
    ("風", ["wind"], "other", 1000),

    # ============ 季节 / 时间（十万~万级） ============
    ("春", ["spring", "haru"], "other", 80000),
    ("夏", ["summer", "natsu"], "other", 70000),
    ("秋", ["autumn", "fall", "aki"], "other", 60000),
    ("冬", ["winter", "fuyu"], "other", 50000),
    ("朝", ["morning"], "other", 40000),
    ("昼", ["daytime", "noon"], "other", 35000),
    ("夜", ["night"], "other", 30000),
    ("夕方", ["evening", "twilight"], "other", 25000),
    ("クリスマス", ["christmas"], "other", 20000),
    ("ハロウィン", ["halloween"], "other", 18000),
    ("七夕", ["tanabata"], "other", 6000),
    ("花火", ["fireworks"], "other", 5000),
    ("紅葉", ["autumn_leaves", "momiji"], "other", 4000),
    ("雪景色", ["snow_scene", "snowscape"], "other", 3500),
    ("新緑", ["fresh_greenery"], "other", 3000),
    ("梅雨", ["rainy_season", "tsuyu"], "other", 2500),
    ("卒業", ["graduation"], "other", 2000),
    ("夏祭り", ["summer_festival"], "other", 1500),

    # ============ 光影 / 效果（十万~万级） ============
    ("逆光", ["backlighting"], "other", 60000),
    ("朝日", ["morning_light"], "other", 40000),
    ("夕日", ["sunset_light"], "other", 35000),
    ("月光", ["moonlight"], "other", 30000),
    ("光のエフェクト", ["light_effects", "light_particles"], "other", 25000),
    ("影", ["shadow"], "other", 20000),
    ("木漏れ日", ["sunbeam", "sunbeams", "komorebi"], "other", 15000),
    ("ライティング", ["lighting", "dramatic_lighting"], "other", 12000),
    ("ネオン", ["neon", "neon_lights"], "other", 10000),
    ("発光", ["glowing", "luminescent"], "other", 8000),
    ("炎", ["flame", "fire"], "other", 7000),
    ("煙", ["smoke"], "other", 6000),
    ("魔法陣", ["magic_circle"], "other", 5000),
    ("エフェクト", ["effects", "particle_effects"], "other", 4000),
    ("ブラー", ["blur", "motion_blur"], "other", 3000),
    ("レンズフレア", ["lens_flare"], "other", 2500),

    # ============ 画风 / 美术风格（十万~万级） ============
    ("厚塗り", ["thick_paint", "impasto", "digital_painting"], "other", 100000),
    ("水彩", ["watercolor"], "other", 80000),
    ("アニメ塗り", ["anime_style"], "other", 60000),
    ("鉛筆", ["pencil", "sketch"], "other", 50000),
    ("モノクロ", ["monochrome", "grayscale", "black_and_white"], "other", 40000),
    ("線画", ["line_art", "lineart"], "other", 35000),
    ("ドット絵", ["pixel_art"], "other", 30000),
    ("油絵", ["oil_painting"], "other", 25000),
    ("ラフ", ["rough", "rough_sketch"], "other", 15000),
    ("デフォルメ", ["deformed", "chibi", "super_deformed"], "other", 12000),
    ("パステル", ["pastel", "pastel_colors"], "other", 10000),
    ("幻想的", ["fantasy", "fantastical", "ethereal"], "other", 8000),
    ("和風", ["japanese_style", "wa-style"], "other", 6000),
    ("洋風", ["western_style"], "other", 5000),
    ("中華風", ["chinese_style"], "other", 4000),
    ("SF", ["sci-fi", "science_fiction"], "other", 3500),
    ("スチームパンク", ["steampunk"], "other", 3000),
    ("サイバーパンク", ["cyberpunk"], "other", 2500),

    # ============ 身体特征 / 体型（十万~万级） ============
    ("巨乳", ["large_breasts"], "character", 250000),
    ("貧乳", ["flat_chest", "small_breasts"], "character", 200000),
    ("グラマー", ["glamorous", "curvy"], "character", 150000),
    ("スレンダー", ["slender", "slim"], "character", 120000),
    ("褐色肌", ["dark_skin", "tan", "brown_skin"], "character", 80000),
    ("色白", ["pale_skin", "fair_skin"], "character", 60000),
    ("日焼け", ["suntan", "tanned"], "character", 50000),
    ("そばかす", ["freckles"], "character", 40000),
    ("泣きぼくろ", ["mole_under_eye", "beauty_mark"], "character", 35000),
    ("ほくろ", ["mole", "beauty_mark"], "character", 30000),
    ("傷跡", ["scar"], "character", 25000),
    ("八重歯", ["fangs", "canine_teeth"], "character", 20000),
    ("つり目", ["tsurime", "tareme", "upturned_eyes"], "character", 18000),
    ("たれ目", ["droopy_eyes", "tareme"], "character", 15000),
    ("糸目", ["closed_eyes", "slit_eyes"], "character", 12000),
    ("包帯", ["bandages", "bandaged"], "character", 10000),
    ("眼帯", ["eyepatch"], "character", 8000),
    ("包帯少女", ["bandaged_girl"], "character", 5000),
    ("角", ["horns"], "character", 4000),
    ("しっぽ", ["tail"], "character", 3500),
    ("羽", ["wings"], "character", 3000),

    # ============ 状态 / 场景描述（万级） ============
    ("戦闘", ["battle", "fighting"], "action", 50000),
    ("休息", ["resting", "break"], "action", 40000),
    ("食事", ["eating"], "action", 35000),
    ("睡眠", ["sleeping"], "action", 30000),
    ("読書", ["reading"], "action", 25000),
    ("勉強", ["studying"], "action", 20000),
    ("お風呂", ["bath", "bathing"], "action", 18000),
    ("シャワー", ["shower"], "action", 15000),
    ("着替え", ["changing_clothes", "dressing"], "action", 12000),
    ("掃除", ["cleaning"], "action", 10000),
    ("料理", ["cooking"], "action", 8000),
    ("運転", ["driving"], "action", 6000),
    ("買い物", ["shopping"], "action", 5000),
    ("お茶", ["tea_time", "tea_party"], "action", 4000),
    ("遊ぶ", ["playing"], "action", 3500),
    ("待つ", ["waiting"], "action", 3000),
    ("考える", ["thinking"], "action", 2500),
    ("見る", ["looking", "watching"], "action", 2000),
    ("聞く", ["listening"], "action", 1500),
    ("話す", ["talking", "speaking"], "action", 1200),
    ("笑う", ["laughing"], "expression", 1000),

    # ============ 更多配饰 / 细节（万~千级） ============
    ("マフラー巻き", ["scarf_wrapped"], "accessory", 8000),
    ("ロケット", ["locket", "pendant"], "accessory", 6000),
    ("ペンダント", ["pendant"], "accessory", 5000),
    ("数珠", ["prayer_beads", "juzu"], "accessory", 4000),
    ("お守り", ["charm", "omamori"], "accessory", 3500),
    ("紋章", ["crest", "emblem"], "accessory", 3000),
    ("腕章", ["armband"], "accessory", 2500),
    ("バッジ", ["badge", "pin"], "accessory", 2000),
    ("コサージュ", ["corsage"], "accessory", 1500),
    ("リュック", ["backpack"], "accessory", 1200),
    ("バッグ", ["bag", "handbag", "purse"], "accessory", 1000),
    ("トートバッグ", ["tote_bag"], "accessory", 800),
    ("ショルダーバッグ", ["shoulder_bag"], "accessory", 700),
    ("スーツケース", ["suitcase"], "accessory", 600),
    ("扇子", ["folding_fan", "sensu"], "accessory", 500),
    ("うちわ", ["uchiwa", "round_fan"], "accessory", 400),
    ("団扇", ["uchiwa", "round_fan"], "accessory", 350),
    ("ハンカチ", ["handkerchief"], "accessory", 300),
    ("ティッシュ", ["tissue"], "accessory", 250),
    ("羽ペン", ["quill_pen"], "accessory", 200),

    # ============ 更多服装/面料质感（万~千级） ============
    ("パフスリーブ", ["puff_sleeves"], "outfit", 20000),
    ("フレアスカート", ["flare_skirt"], "outfit", 15000),
    ("タイトスカート", ["tight_skirt", "pencil_skirt"], "outfit", 12000),
    ("ロングスカート", ["long_skirt"], "outfit", 10000),
    ("キュロット", ["culottes"], "outfit", 8000),
    ("サスペンダースカート", ["suspender_skirt"], "outfit", 6000),
    ("エプロンドレス", ["apron_dress"], "outfit", 5000),
    ("ジャンパースカート", ["jumper_skirt"], "outfit", 4000),
    ("シャツワンピ", ["shirt_dress"], "outfit", 3500),
    ("ベビードール", ["babydoll"], "outfit", 3000),
    ("コルセット", ["corset"], "outfit", 2800),
    ("レギンス", ["leggings"], "outfit", 2500),
    ("スパッツ", ["spats", "tight_shorts"], "outfit", 2000),
    ("ボディスーツ", ["bodysuit", "leotard"], "outfit", 1800),
    ("レオタード", ["leotard"], "outfit", 1500),
    ("ビスチェ", ["bustier"], "outfit", 1200),
    ("ドレープ", ["draped", "draped_clothing"], "outfit", 1000),
    ("プリーツ", ["pleats", "pleated"], "outfit", 900),
    ("ベルベット", ["velvet"], "outfit", 800),
    ("シフォン", ["chiffon"], "outfit", 700),
    ("サテン", ["satin"], "outfit", 600),
    ("ニット", ["knit", "knitwear"], "outfit", 500),
    ("チェック柄", ["checkered", "plaid", "tartan"], "outfit", 400),
    ("ストライプ", ["striped", "stripes"], "outfit", 350),
    ("ドット柄", ["polka_dot", "dotted"], "outfit", 300),
    ("花柄", ["floral_print", "flower_pattern"], "outfit", 250),
    ("水玉", ["polka_dot"], "outfit", 200),
    ("迷彩", ["camouflage"], "outfit", 150),
    ("豹柄", ["leopard_print"], "outfit", 100),
    ("ボーダー", ["striped", "border_pattern"], "outfit", 80),

    # ============ 更多发型/发饰（万~千级） ============
    ("くせっ毛", ["curly_hair"], "hairstyle", 25000),
    ("天然パーマ", ["natural_perm", "wavy_hair"], "hairstyle", 15000),
    ("パーマ", ["perm", "permed_hair"], "hairstyle", 12000),
    ("ソバージュ", ["wavy_hair", "beach_wave"], "hairstyle", 8000),
    ("スパイラル", ["spiral_curls", "ringlet"], "hairstyle", 5000),
    ("カーリーヘア", ["curly_hair", "curl"], "hairstyle", 4000),
    ("アップヘア", ["updo", "up_hair"], "hairstyle", 3500),
    ("ポンパドール", ["pompadour"], "hairstyle", 2500),
    ("モヒカン", ["mohawk"], "hairstyle", 2000),
    ("ツンツン髪", ["spiky_hair"], "hairstyle", 1500),
    ("寝癖", ["bed_head", "bed_hair"], "hairstyle", 1200),
    ("髪結い", ["hair_tied", "tied_hair"], "hairstyle", 1000),
    ("髪留め", ["hair_clip", "hair_tie"], "hairstyle", 800),
    ("かんざし", ["kanzashi", "hairpin"], "hairstyle", 600),
    ("くし", ["comb"], "hairstyle", 500),
    ("ヘアブラシ", ["hairbrush"], "hairstyle", 400),
    ("ウィッグ", ["wig"], "hairstyle", 300),

    # ============ 更多环境/建筑（万~千级） ============
    ("海岸", ["shore", "seaside", "coast"], "other", 20000),
    ("砂浜", ["sandy_beach"], "other", 15000),
    ("川", ["river", "stream"], "other", 12000),
    ("湖", ["lake"], "other", 10000),
    ("滝", ["waterfall"], "other", 8000),
    ("山", ["mountain"], "other", 7000),
    ("草原", ["grassland", "meadow"], "other", 6000),
    ("花畑", ["flower_field", "flower_garden"], "other", 5000),
    ("砂漠", ["desert"], "other", 4500),
    ("洞窟", ["cave"], "other", 4000),
    ("火山", ["volcano"], "other", 3000),
    ("氷", ["ice", "frozen"], "other", 2500),
    ("氷河", ["glacier"], "other", 2000),
    ("夕暮れ", ["twilight", "dusk"], "other", 1800),
    ("明け方", ["dawn", "daybreak"], "other", 1500),
    ("真昼", ["noon", "midday"], "other", 1200),
    ("真夜中", ["midnight"], "other", 1000),
    ("校舎", ["school_building"], "other", 900),
    ("体育館", ["gymnasium"], "other", 800),
    ("プール", ["pool", "swimming_pool"], "other", 700),
    ("駅", ["train_station", "station"], "other", 600),
    ("電車", ["train"], "other", 550),
    ("バス停", ["bus_stop"], "other", 500),
    ("橋", ["bridge"], "other", 450),
    ("塔", ["tower"], "other", 400),
    ("灯台", ["lighthouse"], "other", 350),
    ("工場", ["factory"], "other", 300),
    ("倉庫", ["warehouse"], "other", 250),
    ("研究室", ["laboratory", "lab"], "other", 200),
    ("和室", ["japanese_room", "tatami"], "other", 150),
    ("洋室", ["western_room"], "other", 130),
    ("キッチン", ["kitchen"], "other", 120),
    ("浴室", ["bathroom"], "other", 110),
    ("ベランダ", ["balcony", "veranda"], "other", 100),
    ("廊下", ["hallway", "corridor"], "other", 90),
    ("階段", ["stairs", "staircase"], "other", 80),
    ("エレベーター", ["elevator"], "other", 70),
    ("窓", ["window"], "other", 60),
    ("ドア", ["door"], "other", 50),

    # ============ 更多植物/自然（万~千级） ============
    ("薔薇", ["rose"], "other", 15000),
    ("ひまわり", ["sunflower"], "other", 12000),
    ("百合の花", ["lily"], "other", 10000),
    ("チューリップ", ["tulip"], "other", 8000),
    ("あじさい", ["hydrangea"], "other", 7000),
    ("コスモス", ["cosmos"], "other", 6000),
    ("たんぽぽ", ["dandelion"], "other", 5000),
    ("すみれ", ["violet"], "other", 4000),
    ("椿", ["camellia", "tsubaki"], "other", 3500),
    ("梅", ["plum_blossom", "ume"], "other", 3000),
    ("藤", ["wisteria", "fuji"], "other", 2500),
    ("竹", ["bamboo"], "other", 2000),
    ("紅葉", ["autumn_leaves", "momiji", "maple"], "other", 1800),
    ("いちょう", ["ginkgo"], "other", 1500),
    ("松", ["pine"], "other", 1200),
    ("苔", ["moss"], "other", 1000),
    ("きのこ", ["mushroom"], "other", 800),
    ("四つ葉のクローバー", ["four-leaf_clover"], "other", 600),
    ("サボテン", ["cactus"], "other", 500),
    ("観葉植物", ["houseplant", "foliage_plant"], "other", 400),

    # ============ 更多动物/宠物（万~千级） ============
    ("猫耳娘", ["cat_girl", "nekomimi"], "character", 50000),
    ("犬耳", ["dog_ears", "inumimi"], "character", 35000),
    ("うさぎ耳", ["rabbit_ears", "bunny_ears"], "character", 30000),
    ("うさぎ", ["rabbit", "bunny"], "character", 25000),
    ("鳥", ["bird"], "character", 20000),
    ("フクロウ", ["owl"], "character", 15000),
    ("ハムスター", ["hamster"], "character", 12000),
    ("クマ", ["bear"], "character", 10000),
    ("パンダ", ["panda"], "character", 8000),
    ("イルカ", ["dolphin"], "character", 6000),
    ("くじら", ["whale"], "character", 5000),
    ("ペンギン", ["penguin"], "character", 4000),
    ("カエル", ["frog"], "character", 3000),
    ("ヘビ", ["snake"], "character", 2500),
    ("トカゲ", ["lizard"], "character", 2000),
    ("カメ", ["turtle", "tortoise"], "character", 1500),
    ("蝶", ["butterfly"], "character", 1200),
    ("小鳥", ["small_bird", "songbird"], "character", 1000),
    ("カラス", ["crow", "raven"], "character", 800),
    ("鳩", ["pigeon", "dove"], "character", 600),
    ("金魚", ["goldfish"], "character", 500),
    ("熱帯魚", ["tropical_fish"], "character", 400),
    ("タコ", ["octopus", "tako"], "character", 300),
    ("イカ", ["squid", "ika"], "character", 250),

    # ============ 更多画材/技法（万~千级） ============
    ("アクリル画", ["acrylic_painting"], "other", 15000),
    ("透明水彩", ["transparent_watercolor"], "other", 10000),
    ("コピック", ["copic", "marker"], "other", 8000),
    ("ボールペン画", ["ballpoint_pen", "pen_drawing"], "other", 6000),
    ("色鉛筆", ["colored_pencil"], "other", 5000),
    ("パステル画", ["pastel_drawing"], "other", 4000),
    ("木炭画", ["charcoal_drawing"], "other", 3000),
    ("墨絵", ["ink_painting", "sumi-e"], "other", 2500),
    ("版画", ["print", "woodblock_print"], "other", 2000),
    ("切り絵", ["paper_cut", "kirie"], "other", 1500),
    ("刺繍アート", ["embroidery_art"], "other", 1200),
    ("フォトバッシュ", ["photobash", "photo_bashing"], "other", 1000),
    ("3DCG", ["3dcg", "3d_render"], "other", 900),
    ("ドローイング", ["drawing"], "other", 800),
    ("コラージュ", ["collage"], "other", 700),
    ("グラフィティ", ["graffiti"], "other", 600),
    ("タイポグラフィ", ["typography"], "other", 500),
    ("アイコンデザイン", ["icon_design"], "other", 400),
    ("スケッチ", ["sketch", "rough_sketch"], "other", 150),

    # ============ 更多氛围/情感标签（万~千级） ============
    ("穏やか", ["calm", "peaceful", "serene"], "other", 25000),
    ("明るい", ["bright", "cheerful"], "other", 8000),
    ("暖かい", ["warm", "cozy", "warm_atmosphere"], "other", 7000),
    ("耽美", ["aesthetic", "artistic", "tanbi"], "other", 3500),
    ("退廃的", ["decadent"], "other", 2500),
    ("ダーク", ["dark", "gothic"], "other", 2000),
    ("ポップ", ["pop", "pop_style"], "other", 1500),
    ("カジュアル", ["casual"], "other", 800),

    # ============ 更多身体部位/细节（万~千级） ============
    ("口", ["mouth", "lips"], "character", 50000),
    ("くちびる", ["lips"], "character", 40000),
    ("歯", ["teeth"], "character", 30000),
    ("まつげ", ["eyelashes"], "character", 25000),
    ("眉毛", ["eyebrows"], "character", 20000),
    ("爪", ["nails", "fingernails"], "character", 15000),
    ("手", ["hands", "hand_focus"], "character", 12000),
    ("足", ["feet", "foot_focus"], "character", 10000),
    ("指", ["fingers"], "character", 8000),
    ("手首", ["wrist"], "character", 6000),
    ("足首", ["ankle"], "character", 5000),
    ("鎖骨", ["collarbone", "clavicle"], "character", 4000),
    ("肩", ["shoulders"], "character", 3500),
    ("背中", ["back"], "character", 3000),
    ("お腹", ["stomach", "belly"], "character", 2500),
    ("へそ", ["navel", "bellybutton"], "character", 2000),
    ("わき", ["armpit", "underarm"], "character", 1500),
    ("太もも", ["thighs", "thigh"], "character", 1200),
    ("ふくらはぎ", ["calves", "calf"], "character", 1000),
    ("二の腕", ["upper_arm"], "character", 800),
    ("えくぼ", ["dimples"], "character", 600),
    ("まぶた", ["eyelids"], "character", 500),
    ("ひとみ", ["pupils", "irises"], "character", 400),
    ("白目", ["sclera", "white_of_eyes"], "character", 300),

    # ============ 更多特殊标签（万~千级） ============
    ("セリフ付き", ["with_text", "speech_bubble"], "other", 20000),
    ("吹き出し", ["speech_bubble", "dialogue_box"], "other", 15000),
    ("男体化", ["genderbend", "male_version"], "character", 8000),
    ("性転換", ["gender_swap", "gender_bend"], "character", 6000),
    ("女装", ["crossdressing", "trap"], "outfit", 5000),
    ("男装", ["crossdressing", "male_clothes_on_female"], "outfit", 4000),
    ("獣化", ["kemonomimi", "animal_transformation"], "character", 3000),
    ("幼児化", ["aged_down", "de-aged"], "character", 2500),
    ("成長", ["aged_up", "grown_up"], "character", 2000),
]


def _load_existing_cache() -> list:
    """加载已有的缓存文件"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data
        except Exception:
            pass
    return []


def _try_fetch_url(url: str, timeout: float = 8.0) -> str | None:
    """尝试抓取一个 URL，返回文本内容或 None"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "ja,en;q=0.9",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read()
            # 尝试多种编码
            for enc in ["utf-8", "utf-8-sig", "shift_jis", "euc-jp"]:
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="replace")
    except Exception:
        return None


def _scrape_pixiv_trending_tags() -> list[dict]:
    """
    从 Pixiv 公开页面/API 抓取热门标签及其作品数。

    尝试多个来源：
    1. Pixiv 搜索建议 JSONP API — 返回热门搜索词
    2. Pixiv 标签自动补全 AJAX API — 返回标签建议，可能含计数
    3. Pixiv 排行榜页面 — 返回日/周/月排行榜标签
    """
    tags = []

    # 源1：Pixiv 搜索建议 JSONP API
    # 返回结构: {candidates: [{tag_name, access_count, illust_count?, ...}]}
    for period in ["day", "week", "month"]:
        url = f"https://www.pixiv.net/rpc/cps.php?keyword=&period={period}"
        text = _try_fetch_url(url, timeout=8.0)
        if not text:
            continue
        try:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                candidates = data.get("candidates", [])
                for item in candidates:
                    tag_name = str(item.get("tag_name", "")).strip()
                    if not tag_name:
                        continue
                    # 获取真实的作品数（如果 API 返回了）
                    illust_count = _safe_int(item.get("illust_count"))
                    if not illust_count:
                        illust_count = _safe_int(item.get("count"))
                    tags.append({
                        "tag": tag_name,
                        "en_keywords": [],
                        "category": "other",
                        "count": illust_count or _estimate_count_by_period(period, len(tags)),
                        "source": f"trending_{period}",
                    })
        except Exception:
            pass
        time.sleep(1.0)

    # 源2：Pixiv 标签自动补全 AJAX API
    # 用假名遍历更多前缀获取标签
    import urllib.parse
    for prefix in ["", "a", "i", "u", "e", "o", "ka", "sa", "ta", "na", "ha", "ma", "ya", "ra", "wa"]:
        auto_url = f"https://www.pixiv.net/ajax/search/autocomplete/{urllib.parse.quote(prefix)}?lang=ja"
        text = _try_fetch_url(auto_url, timeout=6.0)
        if not text:
            continue
        try:
            data = json.loads(text)
            items = []
            if isinstance(data, dict):
                body = data.get("body", {})
                if isinstance(body, dict):
                    items = body.get("candidates", body.get("tags", []))
                elif isinstance(body, list):
                    items = body
            for item in items:
                if isinstance(item, dict):
                    tag_name = str(item.get("tag_name", item.get("tag", item.get("tag_translation", "")))).strip()
                    illust_count = _safe_int(item.get("illust_count", item.get("count", item.get("cnt"))))
                else:
                    tag_name = str(item).strip()
                    illust_count = 0
                if tag_name and tag_name not in {t["tag"] for t in tags}:
                    tags.append({
                        "tag": tag_name,
                        "en_keywords": [],
                        "category": "other",
                        "count": illust_count or _estimate_count_by_index(len(tags)),
                        "source": "autocomplete",
                    })
        except Exception:
            pass
        time.sleep(0.3)

    return tags


def _safe_int(val) -> int:
    """安全转换为整数"""
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _estimate_count_by_period(period: str, index: int) -> int:
    """根据排行榜周期和排名估算作品数"""
    base = {"day": 100000, "week": 500000, "month": 1000000}.get(period, 100000)
    decay = max(1, index + 1)
    return max(10000, base // decay)


def _estimate_count_by_index(index: int) -> int:
    """根据补全列表中的位置估算作品数"""
    return max(5000, 200000 // (index + 1))


def _merge_and_enrich_tags(scraped_tags: list[dict], fallback_tags: list[tuple]) -> list[dict]:
    """
    将在线抓取的标签与内置标签合并去重。
    在线抓取到的真实 count 优先，兜底标签的 count 作为补充/fallback。
    """
    fallback_index: dict[str, tuple[list[str], str, int]] = {}
    for tag, en_keywords, category, count in fallback_tags:
        fallback_index[tag] = (list(en_keywords), category, count)

    merged: OrderedDict[str, dict] = OrderedDict()

    # 先处理在线抓取的标签
    for item in scraped_tags:
        tag = item["tag"]
        if tag in merged:
            # 已存在，取 count 较大的
            if item.get("count", 0) > merged[tag].get("count", 0):
                merged[tag]["count"] = item["count"]
            continue
        fb = fallback_index.get(tag)
        merged[tag] = {
            "tag": tag,
            "en_keywords": list(item.get("en_keywords", [])) or (fb[0] if fb else []),
            "category": item.get("category") or (fb[1] if fb else "other"),
            "count": item.get("count", 0) or (fb[2] if fb else 5000),
        }

    # 再补齐内置标签
    for tag, (en_keywords, category, count) in fallback_index.items():
        if tag not in merged:
            merged[tag] = {
                "tag": tag,
                "en_keywords": list(en_keywords),
                "category": category,
                "count": count,
            }

    # 按 count 降序，同 count 按 tag 排序
    result = sorted(merged.values(), key=lambda x: (-x.get("count", 0), x["tag"]))
    return result


def scrape_pixiv_tags(force_refresh: bool = False) -> list[dict]:
    """
    获取 Pixiv 热门标签列表。
    - 如果本地已有缓存且未强制刷新，直接返回。
    - 否则先尝试在线抓取，再合并兜底标签。
    """
    if not force_refresh:
        cached = _load_existing_cache()
        if cached:
            return cached

    print("[pixiv_tag_scraper] 开始抓取 Pixiv 热门标签...")

    scraped = []
    try:
        import urllib.parse
        scraped = _scrape_pixiv_trending_tags()
        print(f"[pixiv_tag_scraper] 在线抓取到 {len(scraped)} 个标签")
    except Exception as e:
        print(f"[pixiv_tag_scraper] 在线抓取失败: {e}")

    merged = _merge_and_enrich_tags(scraped, _FALLBACK_TAGS)
    print(f"[pixiv_tag_scraper] 合并后共 {len(merged)} 个标签")

    # 保存到 data 目录
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[pixiv_tag_scraper] 已保存到 {CACHE_FILE}")

    # 打印统计
    from collections import Counter
    cat_counts = Counter(t["category"] for t in merged)
    print(f"[pixiv_tag_scraper] 按分类统计：")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    return merged


def load_pixiv_tags() -> list[dict]:
    """加载已缓存的 Pixiv 标签列表。如无缓存，先抓取。"""
    cached = _load_existing_cache()
    if cached:
        return cached
    return scrape_pixiv_tags(force_refresh=True)


if __name__ == "__main__":
    result = scrape_pixiv_tags(force_refresh=True)
    print(f"\n总计 {len(result)} 个标签")
    print(f"\n前 20 个热门标签（按作品数降序）：")
    for item in result[:20]:
        kw_preview = ", ".join(item["en_keywords"][:5]) if item["en_keywords"] else "(无英文关键词)"
        count_str = f"{item['count']:,}" if isinstance(item.get("count"), int) else str(item.get("count", "?"))
        print(f"  {item['tag']}  [{item['category']}]  count: {count_str}  en: {kw_preview}")
    print(f"\n最后 5 个标签：")
    for item in result[-5:]:
        count_str = f"{item['count']:,}" if isinstance(item.get("count"), int) else str(item.get("count", "?"))
        print(f"  {item['tag']}  [{item['category']}]  count: {count_str}")
