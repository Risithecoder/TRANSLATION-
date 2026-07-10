import os
import subprocess
import shutil

def convert(input_path: str, output_pdf_path: str):
    """
    Convert a .docx or .html/.htm file to PDF.
    Output is always written to output_pdf_path.

    Args:
        input_path (str): Full path to the .docx or .html file.
        output_pdf_path (str): Full path where the output .pdf should be saved.

    Raises:
        ValueError: If the file extension is not supported.
        RuntimeError: If the conversion fails.
    """
    ext = os.path.splitext(input_path)[-1].lower()

    if ext == ".docx":
        _docx_to_pdf(input_path, output_pdf_path)
    elif ext in (".html", ".htm"):
        _html_to_pdf(input_path, output_pdf_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .docx and .html are supported.")


def _docx_to_pdf(docx_path: str, output_pdf_path: str):
    """
    Convert .docx → PDF using LibreOffice headless.
    LibreOffice always saves the PDF in the same directory as the input file,
    so we convert in a temp location and then move to the desired output path.
    """
    input_dir = os.path.dirname(os.path.abspath(docx_path))
    input_filename = os.path.basename(docx_path)
    expected_pdf = os.path.join(
        input_dir,
        os.path.splitext(input_filename)[0] + ".pdf"
    )

    try:
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", input_dir,
                docx_path
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice conversion failed:\n{result.stderr}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "LibreOffice is not installed or not in PATH. "
            "Install with: sudo apt install libreoffice"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("LibreOffice conversion timed out after 120 seconds.")

    if not os.path.exists(expected_pdf):
        raise RuntimeError(
            f"LibreOffice ran but output PDF not found at: {expected_pdf}"
        )

    # Move to the desired output path if different
    if os.path.abspath(expected_pdf) != os.path.abspath(output_pdf_path):
        shutil.move(expected_pdf, output_pdf_path)


def _html_to_pdf(html_path: str, output_pdf_path: str):
    """
    Convert .html → PDF using wkhtmltopdf.
    wkhtmltopdf is already installed on the server.
    Falls back to WeasyPrint if wkhtmltopdf is not found.
    """
    # Try wkhtmltopdf first (already on server)
    if shutil.which("wkhtmltopdf"):
        try:
            result = subprocess.run(
                [
                    "wkhtmltopdf",
                    "--enable-local-file-access",
                    "--quiet",
                    html_path,
                    output_pdf_path
                ],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"wkhtmltopdf failed:\n{result.stderr}"
                )
            if not os.path.exists(output_pdf_path):
                raise RuntimeError("wkhtmltopdf ran but output PDF not found.")
            return
        except subprocess.TimeoutExpired:
            raise RuntimeError("wkhtmltopdf timed out after 120 seconds.")

    # Fallback: WeasyPrint
    try:
        from weasyprint import HTML
        HTML(filename=html_path).write_pdf(output_pdf_path)
        if not os.path.exists(output_pdf_path):
            raise RuntimeError("WeasyPrint ran but output PDF not found.")
    except ImportError:
        raise RuntimeError(
            "Neither wkhtmltopdf nor WeasyPrint is available.\n"
            "Install WeasyPrint with: pip install weasyprint"
        )