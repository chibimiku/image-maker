import os
import sys

# tools/ 下运行时确保项目根在 sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.web_probe_cli import main


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        sys.exit(1)
