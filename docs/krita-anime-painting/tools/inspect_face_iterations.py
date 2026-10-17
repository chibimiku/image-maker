"""Read saved Krita outputs, make a labelled comparison, verify containers."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/test-result/20261017/krita-face-iterations'
sys.path.insert(0,str(ROOT/'tools'))
from krita_studio import bridge_call


def main():
    code=r'''
from PyQt5.QtGui import QImage,QPainter,QColor,QFont
from PyQt5.QtCore import QRect,Qt
from pathlib import Path
import json
out=Path(OUTPUT)
sources=[('BASELINE','00-baseline.png',True),('R1 - rejected nose/lips','02-round1-face.png',False),('R2 - quiet nose/lips','03-round2-face.png',False),('R3 - lids too heavy','04-round3-face.png',False),('R4 - retained local revision','05-round4-face.png',False)]
board=QImage(1840,1250,QImage.Format_ARGB32);board.fill(QColor('#f3f0eb'))
p=QPainter(board);p.setRenderHint(QPainter.SmoothPixmapTransform)
p.setFont(QFont('Segoe UI',16))
sizes=[]
for i,(label,file,whole) in enumerate(sources):
    im=QImage(str(out/'painting'/file));assert not im.isNull(),file
    if whole:im=im.copy(QRect(504,1002,924,840))
    assert [im.width(),im.height()]==[924,840]
    if whole:assert im.save(str(out/'baseline-face.png'),'PNG')
    x=20+(i%3)*610;y=20+(i//3)*615
    p.setPen(QColor('#34354c'));p.drawText(x,y+24,label)
    p.drawImage(QRect(x,y+40,600,545),im)
    sizes.append({'file':file,'source_detail':[im.width(),im.height()]})
p.setFont(QFont('Segoe UI',17));p.setPen(QColor('#34354c'))
p.drawText(QRect(1240,700,560,400),Qt.TextWordWrap,'Fixed camera, iris, gaze and neutral expression.\n\nEach round was viewed before the next was designed.\n\nNose volume, long lower face and flat cheeks remain unresolved.')
p.end();assert board.save(str(out/'comparison-board.png'),'PNG')
result={'board':str(out/'comparison-board.png'),'inputs':sizes}
replay=out/'replay-check/face-iterations-final-snapshot.png'
if replay.is_file():
    import hashlib
    a=QImage(str(out/'painting/05-round4-lid-refined.png')).convertToFormat(QImage.Format_ARGB32)
    b=QImage(str(replay)).convertToFormat(QImage.Format_ARGB32)
    assert not a.isNull() and a.size()==b.size()
    def raw(im):
        ptr=im.constBits();ptr.setsize(im.byteCount());return bytes(ptr)
    aa,bb=raw(a),raw(b)
    assert aa==bb,'Snapshot replay differs from iterative document'
    result['replay']={'pixel_identical':True,'size':[a.width(),a.height()],'argb32_sha256':hashlib.sha256(aa).hexdigest()}
'''.replace('OUTPUT',repr(str(OUT)))
    response=bridge_call('run_python',{'code':code})
    evidence=[]
    for stem in ['01-construction','02-round1-complete','03-round2-quiet-nose-lips','04-round3-complete','05-round4-lid-refined']:
        p=OUT/'painting'/(stem+'.kra')
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None and 'maindoc.xml' in z.namelist()
        evidence.append({'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'zip_ok':True})
    response['kra_checks']=evidence
    (OUT/'verification.json').write_text(json.dumps(response,indent=2),encoding='utf-8')
    print(json.dumps({'comparison':response['result']['result'],'kra_containers_checked':len(evidence)},ensure_ascii=True))


if __name__=='__main__':main()
