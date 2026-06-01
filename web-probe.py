import sys

from utils.web_probe_cli import main


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        sys.exit(1)
