"""Remove optional exporter identification and correct document page counts."""
from pathlib import Path
from xml.etree import ElementTree as E
import zipfile

ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'docs/Endterm_Defense.pptx'
with zipfile.ZipFile(p) as z:
    contents={n:z.read(n) for n in z.namelist()}
for name in ['docProps/core.xml','docProps/app.xml']:
    tree=E.fromstring(contents[name])
    for element in list(tree):
        local=element.tag.rsplit('}',1)[-1]
        if local in ['creator','lastModifiedBy','Application']:
            tree.remove(element)
        elif local in ['Slides','Notes']:
            element.text='9'
    contents[name]=E.tostring(tree,encoding='utf-8',xml_declaration=True)
temporary=p.with_suffix('.properties.pptx')
with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in contents.items():z.writestr(name,data)
temporary.replace(p)
print('Optional exporter properties removed; slide content unchanged.')
