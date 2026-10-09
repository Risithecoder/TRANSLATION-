from docx import Document
from docx.oxml import parse_xml
import mammoth

doc = Document()
p = doc.add_paragraph('Here is an equation: ')
# Add an OMML equation (a^2 + b^2 = c^2)
math_xml = """
<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
  <m:r><m:t>a</m:t></m:r>
  <m:sup><m:e><m:r><m:t>a</m:t></m:r></m:e><m:supPr><m:ctrlPr/></m:supPr><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sup>
</m:oMath>
"""
p._element.append(parse_xml(math_xml))
doc.save('test_math.docx')

with open('test_math.docx', 'rb') as f:
    result = mammoth.convert_to_html(f)
    print("HTML:", result.value)
