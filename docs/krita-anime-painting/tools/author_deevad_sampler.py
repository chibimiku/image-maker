"""Six native brush samples after importing the author-verified local bundle."""
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[3]
    out=root/'data/test-result/20261017/krita-deevad-install'
    out.mkdir(parents=True,exist_ok=True)
    names=[('b4) Basic Size Pressure - deevad 25.01',24),('c2) Mechanical Pencil Detail - deevad 25.01',10),('c4) Thin Brush Regular - deevad 25.01',12),('d) Glazing Round - deevad 25.01',65),('d) Block Brush Glazing - deevad 25.01',65),('k) Blender Textured Soft - deevad 25.01',65)]
    paper='<rect width="1500" height="1280" fill="#f6f2ed"/><text x="45" y="40" font-size="23" fill="#494658">Native Deevad 25.01 samples — pressure curve and layered marks</text>'
    layers=[]
    for i,(brush,size) in enumerate(names):
        y=70+i*195
        paper+=f'<rect x="25" y="{y}" width="1450" height="181" rx="10" fill="#fffaf4"/><text x="45" y="{y+28}" font-size="20" fill="#494658">{brush}; size {size}px</text>'
        strokes=[]
        if brush.startswith('k)'):
            for j,color in enumerate(['#9987aa','#eab5ad','#bdd4d7']):
                strokes.append({'brush':'b2) Basic Size Fill Area - deevad 25.01','color':color,'size':55,'opacity':1,'path':[['M',150+j*260,y+95],['L',600+j*260,y+95]]})
        strokes.append({'brush':brush,'color':'#77657e','size':size,'opacity':1,'path':[['M',80,y+112],['C',230,y+5,400,y+160,650,y+90]],'taper':True,'samples':60})
        for j in range(3):
            strokes.append({'brush':brush,'color':'#bd8f9d','size':size,'opacity':.3,'path':[['M',820,y+75+j*15],['Q',1040,y+150,1370,y+80+j*15]]})
        layers.append({'name':f'{i+1} {brush}','strokes':strokes})
    scene={'name':'Deevad 25.01 native brush sampler','stem':'deevad-native-sampler','width':1500,'height':1280,'scale':1,'layers':[{'name':'Sample paper and labels','svg':'<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1280">'+paper+'</svg>'}]+layers}
    (out/'sampler.scene.json').write_text(json.dumps(scene,indent=2),encoding='utf-8')
    print('Prepared six native brush rows; mixing row has pre-existing color.')


if __name__=='__main__':main()
