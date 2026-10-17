"""Read-only offline hints: no launch, network, installation or secret reads."""
import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def discover(krita_path=None, bridge_file=None):
    candidates=[]
    found=shutil.which('krita')
    if found:
        candidates.append(Path(found))
    if krita_path:
        candidates.append(Path(krita_path).expanduser())
    if sys.platform=='win32':
        for key in ('ProgramFiles','ProgramFiles(x86)'):
            if os.environ.get(key):
                candidates.extend(Path(os.environ[key])/name/'bin/krita.exe' for name in ('Krita (x64)','Krita'))
    elif sys.platform=='darwin':
        candidates.append(Path('/Applications/krita.app/Contents/MacOS/krita'))
    else:
        candidates.extend(map(Path,('/usr/bin/krita','/usr/local/bin/krita','/snap/bin/krita')))
    unique=list(dict.fromkeys(str(p) for p in candidates))
    config=Path(bridge_file).expanduser() if bridge_file else None
    if config is None and os.environ.get('APPDATA'):
        config=Path(os.environ['APPDATA'])/'krita/krita_mcp_bridge.json'
    return {
        'schema':'drawing-environment-hints/1',
        'os':platform.system(), 'python_version':platform.python_version(),
        'python_command':sys.executable,
        'krita_candidates':[{'path':p,'exists':Path(p).is_file()} for p in unique],
        'bridge_candidate':{'path':str(config) if config else None,'exists':config.is_file() if config else False,'contents_read':False},
        'application_running':'unknown','application_version':'unknown',
        'mcp_tools':'inspect actual session tools','bridge_connection_verified':False,
        'next':'Read ENVIRONMENT.md, verify actual tools and permissions; missing hints do not prove absence.'
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--krita-path')
    parser.add_argument('--bridge-file')
    args=parser.parse_args()
    print(json.dumps(discover(args.krita_path,args.bridge_file),ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
