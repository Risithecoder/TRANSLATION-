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
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


            


def _add_formatted_runs(p, text, size=11, bold=False, italic=False, color=None):
    """Parse **bold**, __underline__, <sup>, and <sub> and add appropriately formatted runs."""
    tokens = re.split(r'(\*\*|__|<sup>|</sup>|<sub>|</sub>)', text)
    is_bold = bold
    is_underline = False
    is_sup = False
    is_sub = False
    
    for token in tokens:
        if not token:
            continue
        if token == '**':
            is_bold = not is_bold
        elif token == '__':
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


def _render_text_block(doc, text: str, size=11, bold=False, italic=False, color=None, space_before=2, space_after=2, indent=False):
    """Render a block of text that might contain markdown tables."""
    if not text:
        return
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        # ── Markdown table block ─────────────────────────────────
        if _is_table_row(line):
            table_lines = []
            while i < len(lines) and (_is_table_row(lines[i].strip()) or _is_separator_row(lines[i].strip())):
                table_lines.append(lines[i].strip())
                i += 1
            _add_markdown_table(doc, table_lines)
            doc.add_paragraph()  # spacing after table
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


def build(batch_outputs: list, output_path: str, language: str):
    """
    Build a formatted .docx file from structured batch translation outputs.

    Args:
        batch_outputs (list): List of structured batch dicts (from Pydantic dump).
        output_path (str): Full path to save the output .docx file.
        language (str): Target language name.
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
            
            # Enforce question number prefix if it's a real question (not 0 or empty)
            if q_no and q_no != "0":
                if not re.match(r'^\d+\.', eq):
                    eq = f"{q_no}. {eq}"
                else:
                    eq = re.sub(r'^\d+\.', f"{q_no}.", eq, count=1)
                
            _render_text_block(doc, eq, size=11, bold=False, space_before=16, space_after=4)
            
            # 2. Language label
            _add_line(doc, f"{language}:", size=11, bold=False, italic=False, 
                      space_before=6, space_after=2)
            
            # 3. Translated Question
            _render_text_block(doc, str(q.get('translated_question', '')), size=11)
            
            # 4. English Options — force (1)-(4)
            eng_opts = _renumber_options(q.get('english_options', []), 1)
            for opt in eng_opts:
                _render_text_block(doc, str(opt), size=11, space_before=1, space_after=1, indent=False)
                
            # 5. Translated Options — force (5)-(8)
            trans_opts = _renumber_options(q.get('translated_options', []), len(eng_opts) + 1)
            for opt in trans_opts:
                _render_text_block(doc, str(opt), size=11, space_before=1, space_after=1, indent=False)
                
            # 6. Answer Key
            if q.get('answer_key'):
                _add_line(doc, f"Answer Key: {q.get('answer_key')}", size=11, bold=False, 
                          space_before=8, space_after=4)
            
            # 7. English Solution
            if q.get('english_solution'):
                _add_line(doc, "Solution:", size=11, bold=False, 
                          space_before=8, space_after=2)
                _render_text_block(doc, str(q.get('english_solution', '')), size=11)
            
            # 8. Translated Solution
            if q.get('translated_solution'):
                _add_line(doc, f"{language}:", size=11, bold=False, italic=False, 
                          space_before=6, space_after=2)
                _render_text_block(doc, str(q.get('translated_solution', '')), size=11)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    print(f"[docx_builder] ✅ Saved: {output_path}")