"""Package the knowledge set without running drawing recipes or connecting apps."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'docs/krita-anime-painting'
ALLOWED={'.md','.py','.json','.png'}
ART=ROOT/'data/test-result/20261017/krita-tutorial-volume/painting/tutorial-volume-eye-edge-checked.kra'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def materials():
    files=[p for p in DOC.rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ALLOWED or p.name=='.gitignore')]
    files.append(ROOT/'tools/krita_studio.py')
    return sorted(files,key=lambda p:p.relative_to(ROOT).as_posix())


def validate(files):
    bundled={p.resolve() for p in files}
    external=[]
    for p in files:
        if p.suffix=='.py':
            ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        elif p.suffix=='.json':
            json.loads(p.read_text(encoding='utf-8'))
        elif p.suffix=='.md':
            for raw in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
                raw=raw.strip().strip('<>')
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',raw) or raw.startswith('#'):
                    continue
                target=(p.parent/unquote(raw.split('#')[0])).resolve()
                if target in bundled:
                    continue
                try:
                    target.relative_to(DOC)
                    in_docs=True
                except ValueError:
                    in_docs=False
                if in_docs and target.suffix not in {'.zip','.bundle'}:
                    raise ValueError(f'Broken knowledge link: {p.relative_to(ROOT)} -> {raw}')
                external.append({'document':p.relative_to(ROOT).as_posix(),'target':raw,'reason':'historical local evidence or separately obtained resource'})
    return external


def verify(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None,'Corrupt ZIP member'
        names=z.namelist()
        assert len(names)==len(set(names)),'Duplicate archive path'
        for name in names:
            posix=PurePosixPath(name)
            assert not posix.is_absolute() and '..' not in posix.parts,'Unsafe archive path'
        manifest=json.loads(z.read('MANIFEST.json'))
        assert set(names)=={x['path'] for x in manifest['files']}|{'MANIFEST.json'}
        for entry in manifest['files']:
            data=z.read(entry['path'])
            assert len(data)==entry['bytes'] and digest(data)==entry['sha256'],entry['path']
    return {'verified':True,'files':len(names),'sha256':digest(Path(path).read_bytes())}


def build(output):
    output.mkdir(parents=True,exist_ok=True)
    archive=output/'krita-anime-knowledge-20261017.zip'
    report_path=output/'package-report.json'
    if archive.exists() or report_path.exists():
        raise FileExistsError('Choose a new output directory; existing packages are not overwritten.')
    files=materials()
    external=validate(files)
    entries={p.relative_to(ROOT).as_posix():p.read_bytes() for p in files}
    entries['START-HERE.md']=('# 二次元绘画知识包\n\n先打开 [完整入口](docs/krita-anime-painting/README.md)，然后阅读 ENVIRONMENT.md。\n\n不要假定接收方装有 Krita、MCP 或 Python；先探索环境。\n\n最新分层画稿在 artwork/latest-head.kra（如本次来源可用）；精选图和独立场景在 docs/krita-anime-painting/examples/。\n\nMANIFEST.json 记录所有归档成员 SHA-256。历史本机 data 路径通常未随包提供。\n').encode('utf-8')
    missing=[]
    if ART.is_file():
        with zipfile.ZipFile(ART) as z:
            assert z.testzip() is None and 'maindoc.xml' in z.namelist()
        entries['artwork/latest-head.kra']=ART.read_bytes()
    else:
        missing.append(ART.relative_to(ROOT).as_posix())
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    commit=revision.stdout.strip() if revision.returncode==0 else None
    manifest={'schema':'anime-knowledge-package/1','source_commit':commit,
              'scope':'All knowledge text, recipes and curated examples; optional latest KRA. Not all historical artifacts.',
              'excluded':['third-party brush binaries','credentials and bridge state','third-party tutorial media','user original reference','full historical KRA and data trees','Python caches'],
              'optional_missing':missing,'historical_external_links':external,
              'files':[{'path':n,'bytes':len(b),'sha256':digest(b)} for n,b in sorted(entries.items())]}
    entries['MANIFEST.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,17,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    result={**verify(archive),'archive':str(archive.resolve()),'bytes':archive.stat().st_size,
            'source_commit':commit,'optional_missing':missing,'historical_external_link_count':len(external)}
    report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path)
    group.add_argument('--verify',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else build(args.output),ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
