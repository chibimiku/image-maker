"""Export native-resolution detail crops from the explicitly expected head."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tools'))
from krita_studio import bridge_call
out=Path('data/test-result/20261017/krita-planned-head-batch/painting').resolve()
expected=out/('planned-head-reviewed.kra' if '--final' in sys.argv else '04-frontal-gem-eyes.kra')
specs=[('head-detail.png',[570,180,880,1320]),('eyes-detail.png',[695,865,610,230]),('left-tip-detail.png',[100,2350,700,630])]
if '--final' in sys.argv:specs=[(n.replace('.png','-reviewed.png'),r) for n,r in specs]
code='from pathlib import Path\nassert Path(doc.fileName()).resolve()==Path('+repr(str(expected))+').resolve()\npaths=[]\n'
code+='for name,rect in '+repr(specs)+':\n path=str(Path('+repr(str(out))+')/name)\n assert doc.projection(*rect).save(path,"PNG")\n paths.append(path)\n'
code+='result={"exports":paths}\n'
result=bridge_call('run_python',{'code':code})
print(json.dumps(result,ensure_ascii=True))
