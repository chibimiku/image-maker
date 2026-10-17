"""Export native Krita projection details and verify saved containers."""
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path('tools').resolve()))
from krita_studio import bridge_call

out=Path('data/test-result/20261017/krita-tutorial-volume').resolve()
target=out/'painting/tutorial-volume-eye-edge-checked.kra'
code=f'''
from pathlib import Path
assert Path(doc.fileName()).resolve()==Path({str(target)!r}).resolve()
doc.waitForDone()
outputs=[]
for name,rect in [('head-neck-detail.png',(570,650,900,1200)),('ear-detail.png',(1250,820,180,350)),('neck-detail.png',(770,1210,500,500))]:
    path=str(Path({str(out)!r})/name)
    assert doc.projection(*rect).save(path,'PNG')
    outputs.append(path)
result={{'outputs':outputs,'size':[doc.width(),doc.height()]}}
'''
response=bridge_call('run_python',{'code':code})
with zipfile.ZipFile(target) as z:
    bad=z.testzip()
    assert bad is None
    assert 'maindoc.xml' in z.namelist()
response['container_check']={'zip_ok':True,'kra':str(target)}
(out/'inspection.json').write_text(json.dumps(response,indent=2),encoding='utf-8')
print(json.dumps(response,ensure_ascii=True))
