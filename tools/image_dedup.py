# -*- coding: utf-8 -*-
"""图片 SHA-256 去重 CLI：删掉同目录里「后缀撒谎」的重复副本，输出清单。

配套 `utils/image_identity.py`（真实格式嗅探）与 `utils/image_dedup.py`（去重规则）。

例：
  # 预演：只列不删（默认）
  python tools/image_dedup.py --dir data/20261006 --dir data/20261007

  # 真的删，并把清单写成 JSON
  python tools/image_dedup.py --dir data/20261007 --apply --json data/20261007/dedup.json

  # 连"后缀都真实、内容相同"的重复也一起收敛
  python tools/image_dedup.py --dir data/20261007 --mode all --apply
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

try:  # 控制台编码兜底（清单里有中文与 emoji）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from utils.image_dedup import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
