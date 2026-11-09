"""Create a slide PDF from the exact reviewed PNGs exported by build_deck.mjs."""
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas

root=Path(__file__).resolve().parents[2]
slides=sorted((Path(__file__).resolve().parent/'.build/rendered').glob('slide-*.png'))
assert len(slides)==9, 'Build and review all nine slides first.'
output=root/'docs/Endterm_Defense.pdf'
c=Canvas(str(output),pagesize=(960,540))
c.setTitle('Video game completion time prediction Endterm defence')
c.setAuthor('Syrym Shadiyarbek and Yerdaulet Akhanov')
for slide in slides:
    c.drawImage(str(slide),0,0,width=960,height=540)
    c.showPage()
c.save()
print('Saved nine-page defence PDF.')
