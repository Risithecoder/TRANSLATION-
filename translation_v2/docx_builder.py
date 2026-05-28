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
from docx.shared import Pt, RGBColor, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def _add_separator(doc):
    """Add a thin horizontal rule between questions."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("─" * 60)
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(180, 180, 180)


def _add_line(doc, text, size=11, bold=False, italic=False,
              space_before=2, space_after=2, color=None, indent=False):
    """Add a single formatted paragraph."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.left_indent = Inches(0.3)

    if text:
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        if color:
            run.font.color.rgb = color
    return p


def _is_question_number(line: str) -> bool:
    """Check if line starts with a question number like '1. ', '23. '"""
    return bool(re.match(r'^\d+\.\s+', line))


def _is_answer_key(line: str) -> bool:
    """Check if line is an Answer Key line."""
    return bool(re.match(r'^Answer\s*Key\s*:', line, re.IGNORECASE))


def _is_solution_label(line: str) -> bool:
    """Check if line is a Solution label."""
    stripped = line.strip()
    return stripped.lower() in ('solution:', 'solution')


def _is_language_label(line: str, language: str) -> bool:
    """Check if line is a language label like 'Hindi:', 'Tamil:', etc."""
    pattern = rf'^{re.escape(language)}\s*:\s*$'
    return bool(re.match(pattern, line.strip(), re.IGNORECASE))


def _is_option_line(line: str) -> bool:
    """Check if line is a numbered option like '(1) ...' or '1. ...'"""
    return bool(re.match(r'^\(\d+\)\s+', line))


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
            run = para.add_run(text)
            run.font.size = Pt(10)
            run.font.name = 'Calibri'
            run.bold = is_hdr

            if is_hdr:
                # Light grey header background
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement('w:shd')
                shd.set(qn('w:val'), 'clear')
                shd.set(qn('w:color'), 'auto')
                shd.set(qn('w:fill'), 'E8E8E8')
                tcPr.append(shd)

            _set_cell_border(cell)


def build(batch_outputs: list, output_path: str, language: str):
    """
    Build a formatted .docx file from batch translation outputs.

    Args:
        batch_outputs (list): List of raw text strings, one per batch.
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

    for batch_text in batch_outputs:
        if not batch_text:
            continue

        lines = batch_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip completely empty lines (we handle spacing ourselves)
            if not line:
                i += 1
                continue

            # ── Markdown table block ─────────────────────────────────
            if _is_table_row(line):
                # Collect all consecutive table rows
                table_lines = []
                while i < len(lines) and (_is_table_row(lines[i].strip()) or _is_separator_row(lines[i].strip())):
                    table_lines.append(lines[i].strip())
                    i += 1
                _add_markdown_table(doc, table_lines)
                doc.add_paragraph()  # spacing after table
                continue

            # ── Question number line ────────────────────────────────
            if _is_question_number(line):
                if not first_question:
                    pass  # _add_separator(doc)
                first_question = False
                _add_line(doc, line, size=12, bold=True,
                          space_before=16, space_after=4)
                i += 1
                continue

            # ── Answer Key line ─────────────────────────────────────
            if _is_answer_key(line):
                _add_line(doc, line, size=11, bold=True,
                          space_before=8, space_after=4,
                          color=RGBColor(0, 120, 60))
                i += 1
                continue

            # ── Solution label ──────────────────────────────────────
            if _is_solution_label(line):
                _add_line(doc, line, size=11, bold=True,
                          space_before=8, space_after=2,
                          color=RGBColor(0, 90, 160))
                i += 1
                continue

            # ── Language label (e.g., "Hindi:") ─────────────────────
            if _is_language_label(line, language):
                _add_line(doc, line, size=11, bold=True, italic=True,
                          space_before=6, space_after=2,
                          color=RGBColor(180, 80, 0))
                i += 1
                continue

            # ── Option lines ────────────────────────────────────────
            if _is_option_line(line):
                _add_line(doc, line, size=11, space_before=1,
                          space_after=1, indent=True)
                i += 1
                continue

            # ── Regular content line ────────────────────────────────
            _add_line(doc, line, size=11, space_before=2, space_after=2)
            i += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    print(f"[docx_builder] ✅ Saved: {output_path}")