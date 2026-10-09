"""
docx_builder.py — Build formatted .docx output from translated text.

Features:
- Bold question numbers and section labels
- Proper spacing between questions (visual separation)
- Bold "Answer Key:" and "Solution:" lines
- Bold language labels (e.g., "Hindi:")
- Horizontal rule between question blocks
- Markdown table → Word table rendering
"""

import os
import re
import io
import base64
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement



# Maximum image width in the document (inches) — prevents overflow
_MAX_IMAGE_WIDTH = Inches(5.5)


def _insert_image(doc, image_data: dict):
    """
    Insert an image into the document from base64-encoded binary data.
    
    Args:
        doc: The python-docx Document object
        image_data: Dict with keys: data_b64, content_type, alt
    """
    b64 = image_data.get("data_b64", "")
    if not b64:
        # No image data — insert a placeholder text instead
        _add_line(doc, f"[Image: {image_data.get('alt', 'unavailable')}]", 
                  size=10, italic=True, color=RGBColor(0x99, 0x99, 0x99))
        return
    
    try:
        image_bytes = base64.b64decode(b64)
        image_stream = io.BytesIO(image_bytes)
        
        # Determine file extension from content_type
        content_type = image_data.get("content_type", "image/png")
        # python-docx can handle png, jpeg, gif, bmp, tiff etc via Pillow
        
        # Add image with width constraint (preserves aspect ratio)
        doc.add_picture(image_stream, width=_MAX_IMAGE_WIDTH)
        
        # Center the image paragraph
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        last_paragraph.paragraph_format.space_before = Pt(4)
        last_paragraph.paragraph_format.space_after = Pt(4)
        
    except Exception as e:
        print(f"[docx_builder] ⚠️ Failed to insert image: {e}")
        _add_line(doc, f"[Image could not be rendered]", 
                  size=10, italic=True, color=RGBColor(0x99, 0x99, 0x99))


def _add_formatted_runs(p, text, size=11, bold=False, italic=False, color=None):
    """Parse **bold**, «U»underline«U», <sup>, and <sub> and add appropriately formatted runs.
    
    Underline uses «U» markers (not __) to avoid collision with fill-in-the-blank
    underscore sequences (e.g. ______ ).
    """
    tokens = re.split(r'(\*\*|«U»|<sup>|</sup>|<sub>|</sub>)', text)
    is_bold = bold
    is_underline = False
    is_sup = False
    is_sub = False
    
    for token in tokens:
        if not token:
            continue
        if token == '**':
            is_bold = not is_bold
        elif token == '«U»':
            is_underline = not is_underline
        elif token == '<sup>':
            is_sup = True
        elif token == '</sup>':
            is_sup = False
        elif token == '<sub>':
            is_sub = True
        elif token == '</sub>':
            is_sub = False
        else:
            run = p.add_run(token)
            run.font.size = Pt(size)
            run.bold = is_bold
            run.italic = italic
            run.underline = is_underline
            run.font.superscript = is_sup
            run.font.subscript = is_sub
            if color:
                run.font.color.rgb = color

def _add_line(doc, text, size=11, bold=False, italic=False,
              space_before=2, space_after=2, color=None, indent=False):
    """Add a single formatted paragraph."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.left_indent = Inches(0.3)

    if text:
        _add_formatted_runs(p, text, size=size, bold=bold, italic=italic, color=color)
    return p



def _is_table_row(line: str) -> bool:
    """Check if line looks like a markdown table row: | col | col |"""
    return bool(re.match(r'^\s*\|.*\|\s*$', line))


def _is_separator_row(line: str) -> bool:
    """Check if line is a markdown table separator: |---|---|"""
    return bool(re.match(r'^\s*\|[-:\s|]+\|\s*$', line))
 

def _parse_table_row(line: str) -> list:
    """Parse a markdown table row into a list of cell strings."""
    # Strip leading/trailing | and split
    cells = line.strip().strip('|').split('|')
    return [c.strip() for c in cells]


def _set_cell_border(cell):
    """Add thin borders to a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for side in ('top', 'left', 'bottom', 'right'):
        tag = OxmlElement(f'w:{side}')
        tag.set(qn('w:val'), 'single')
        tag.set(qn('w:sz'), '4')
        tag.set(qn('w:color'), 'AAAAAA')
        tcPr.append(tag)


def _add_markdown_table(doc, table_lines: list):
    """
    Render a list of markdown table lines as a proper Word table.
    Header row gets a light grey background; all cells get thin borders.
    """
    # Filter out separator rows and empty rows
    data_rows = []
    header_done = False
    is_header_row = []
    for line in table_lines:
        if not line.strip():
            continue
        if _is_separator_row(line):
            header_done = True  # rows before separator = header
            continue
        cells = _parse_table_row(line)
        if cells:
            data_rows.append(cells)
            is_header_row.append(not header_done)

    if not data_rows:
        return

    # Normalise column count
    col_count = max(len(row) for row in data_rows)
    for row in data_rows:
        while len(row) < col_count:
            row.append('')

    tbl = doc.add_table(rows=len(data_rows), cols=col_count)
    tbl.style = 'Table Grid'

    for r_idx, (row_cells, is_hdr) in enumerate(zip(data_rows, is_header_row)):
        for c_idx, text in enumerate(row_cells):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = ''
            para = cell.paragraphs[0]
            _add_formatted_runs(para, text, size=10, bold=False)
            
            for run in para.runs:
                run.font.name = 'Calibri'

            # No header formatting — plain text only

            _set_cell_border(cell)


def _render_text_block(doc, text: str, size=11, bold=False, italic=False, color=None, space_before=2, space_after=2, indent=False, images=None):
    """Render a block of text that might contain markdown tables or image placeholders."""
    if not text:
        return
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        # ── Markdown table block ─────────────────────────────────────────
        if _is_table_row(line):
            table_lines = []
            while i < len(lines) and (_is_table_row(lines[i].strip()) or _is_separator_row(lines[i].strip())):
                table_lines.append(lines[i].strip())
                i += 1
            _add_markdown_table(doc, table_lines)
            doc.add_paragraph()  # spacing after table
            continue
        
        # ── Check for image placeholders in the line ─────────────────
        if images and re.search(r'\[IMAGE_\d+\]', line):
            # Split line around image placeholders
            parts = re.split(r'(\[IMAGE_\d+\])', line)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                img_match = re.match(r'\[IMAGE_(\d+)\]', part)
                if img_match:
                    img_idx = int(img_match.group(1))
                    # Find the image data by index
                    img_data = None
                    for img in images:
                        if img.get("index") == img_idx:
                            img_data = img
                            break
                    if img_data:
                        _insert_image(doc, img_data)
                    else:
                        # Image data not found — skip placeholder silently
                        pass
                else:
                    _add_line(doc, part, size=size, bold=bold, italic=italic, color=color,
                              space_before=space_before, space_after=space_after, indent=indent)
            i += 1
            continue
            
        _add_line(doc, line, size=size, bold=bold, italic=italic, color=color, 
                  space_before=space_before, space_after=space_after, indent=indent)
        i += 1


    

def _renumber_options(options: list, start: int) -> list:
    """
    Force-renumber a list of option strings to start from `start`.
    e.g. _renumber_options(["(1) Egypt", "(2) China", ...], 5)
      -> ["(5) Egypt", "(6) China", ...]
    """
    result = []
    for i, opt in enumerate(options):
        text = re.sub(r'^\(\d+\)\s*', '', str(opt).strip())
        result.append(f"({start + i}) {text}")
    return result


def build(batch_outputs: list, output_path: str, language: str, images: list = None):
    """
    Build a formatted .docx file from structured batch translation outputs.

    Args:
        batch_outputs (list): List of structured batch dicts (from Pydantic dump).
        output_path (str): Full path to save the output .docx file.
        language (str): Target language name.
        images (list): Optional list of image dicts from text_extractor.
                       Each dict has: index, content_type, data_b64, alt.
    """
    doc = Document()

    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

    first_question = True

    for batch_dict in batch_outputs:
        if not batch_dict:
            continue
            
        questions = batch_dict.get('questions', [])
        for q in questions:
            first_question = False
            
            # 1. Question Number and English Question
            eq = str(q.get('english_question', '')).strip()
            q_no = str(q.get('question_no', ''))
            
            # Strip any leading "N." or "N. " prefix the AI may have included,
            # then re-apply the authoritative question_no cleanly.
            if q_no and q_no != "0":
                # Remove ALL consecutive leading numbers, even if wrapped in bold markers (e.g. "**3. **")
                eq = re.sub(r'^((?:\*+)?\s*\d+[\.\)](?!\d)\s*(?:\*+)?\s*)+', '', eq).strip()
                eq = f"{q_no}. {eq}"
                
            _render_text_block(doc, eq, size=11, bold=False, space_before=16, space_after=4, images=images)
            
            # 2. Language label
            _add_line(doc, f"{language}:", size=11, bold=False, italic=False, 
                      space_before=6, space_after=2)
            
            # 3. Translated Question — strip any leading question number the AI may have added
            tq = str(q.get('translated_question', '')).strip()
            tq = re.sub(r'^\d+[\.\)](?!\d)\s*', '', tq).strip()
            _render_text_block(doc, tq, size=11, images=images)
            
            # 4. English Options — force (1)-(4)
            eng_opts = _renumber_options(q.get('english_options', []), 1)
            for opt in eng_opts:
                _render_text_block(doc, str(opt), size=11, space_before=1, space_after=1, indent=False, images=images)
                
            # 5. Translated Options — force (5)-(8)
            trans_opts = _renumber_options(q.get('translated_options', []), len(eng_opts) + 1)
            for opt in trans_opts:
                _render_text_block(doc, str(opt), size=11, space_before=1, space_after=1, indent=False, images=images)
                
            # 6. Answer Key
            if q.get('answer_key'):
                _add_line(doc, f"Answer Key: {q.get('answer_key')}", size=11, bold=False, 
                          space_before=8, space_after=4)
            
            # 7. English Solution
            if q.get('english_solution'):
                _add_line(doc, "Solution:", size=11, bold=False, 
                          space_before=8, space_after=2)
                _render_text_block(doc, str(q.get('english_solution', '')), size=11, images=images)
            
            # 8. Translated Solution
            if q.get('translated_solution'):
                _add_line(doc, f"{language}:", size=11, bold=False, italic=False, 
                          space_before=6, space_after=2)
                _render_text_block(doc, str(q.get('translated_solution', '')), size=11, images=images)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    print(f"[docx_builder] ✅ Saved: {output_path}")