"""Read MCP-rendered outputs, prepare equal-scale evidence and check saved files."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/test-result/20261017/krita-face-proportions'
sys.path.insert(0,str(ROOT/'tools'))
from krita_studio import bridge_call


def main():
    code=r'''
from PyQt5.QtGui import QImage,QPainter,QColor,QFont
from PyQt5.QtCore import QRect
from pathlib import Path
out=Path(OUTPUT)
assert Path(doc.fileName()).resolve()==(out/'retained/05-B-retained-no-glaze.kra').resolve()
assert not next(n for n in doc.rootNode().childNodes() if n.name()=='44 Deevad cheek glazing trial').visible()
baseline=QImage(BASELINE);assert not baseline.isNull()
a=QImage(str(out/'color-A-reviewed/04-A-no-glaze.png')).copy(QRect(378,752,693,630))
b=QImage(str(out/'retained/05-B-retained-face.png'))
board=QImage(1840,650,QImage.Format_ARGB32);board.fill(QColor('#f4f1ec'))
p=QPainter(board);p.setFont(QFont('Segoe UI',15));p.setPen(QColor('#424457'));p.setRenderHint(QPainter.SmoothPixmapTransform)
for i,(label,im) in enumerate([('R4 baseline',baseline),('A - original length, new planes',a),('B - shorter lower face, retained',b)]):
    assert not im.isNull()
    x=20+i*610;p.drawText(x,28,label);p.drawImage(QRect(x,45,600,545),im)
p.drawText(20,625,'Same eye line, gaze and light. New brush trial excluded from the retained face.')
p.end();assert board.save(str(out/'comparison-board.png'),'PNG')
clean=QImage(str(out/'color-B/03-color-B-no-glaze.png')).convertToFormat(QImage.Format_ARGB32)
final=QImage(str(out/'retained/05-B-retained-no-glaze.png')).convertToFormat(QImage.Format_ARGB32)
def raw(im):
    q=im.constBits();q.setsize(im.byteCount());return bytes(q)
assert clean.size()==final.size() and raw(clean)==raw(final)
result={'board':str(out/'comparison-board.png'),'retained_matches_no_glaze':True,'active_document':doc.fileName()}
'''.replace('OUTPUT',repr(str(OUT))).replace('BASELINE',repr(str(ROOT/'docs/krita-anime-painting/examples/face-iterations-20261017/round4-face.png')))
    report=bridge_call('run_python',{'code':code})
    checks=[]
    for sub in ['construction/01-proportion-construction.kra','construction-reviewed/01b-proportion-construction-reviewed.kra','grey/02-grey-planes.kra','color-A/03-color-A.kra','color-A-reviewed/04-A-glaze-opacity-corrected.kra','color-B/03-color-B.kra','retained/05-B-retained-no-glaze.kra']:
        f=OUT/sub
        with zipfile.ZipFile(f) as z:assert z.testzip() is None and 'maindoc.xml' in z.namelist()
        checks.append({'file':sub,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'zip_ok':True})
    scene=json.loads((OUT/'final-scene.json').read_text(encoding='utf-8'))
    assert not any(n.get('strokes') for n in scene['layers'])
    report['kra_checks']=checks;report['portable_final_requires_deevad']=False
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'result':report['result']['result'],'kra_checked':len(checks),'portable_final_requires_deevad':False},ensure_ascii=True))


if __name__=='__main__':main()
