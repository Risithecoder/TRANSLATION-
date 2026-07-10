import os
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError

# DPI 150 is sufficient for Gemini vision and uses ~4x less RAM than 300 DPI
DPI = 150

# Process this many pages at a time to keep memory usage flat
CHUNK_SIZE = 20


def convert_pdf(pdf_path, output_folder, poppler_path=None):
    """
    Converts each page of a PDF into a JPEG image.
    Processes pages in chunks to avoid OOM on large PDFs.

    Args:
        pdf_path (str): Path to the input PDF.
        output_folder (str): Directory to save the output images.
        poppler_path (str): Optional path to poppler binaries (Windows only).

    Returns:
        list: Sorted list of saved image file paths.
    """
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"Starting conversion for: {pdf_path}...")

    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
        total_pages = info["Pages"]
        print(f"  Total pages: {total_pages} — processing in chunks of {CHUNK_SIZE}")
    except Exception as e:
        print(f"  Could not read page count ({e}), falling back to single-pass conversion")
        total_pages = None

    image_paths = []

    try:
        if total_pages is None:
            # Fallback: convert all at once (small PDFs)
            images = convert_from_path(
                pdf_path,
                dpi=DPI,
                fmt='jpeg',
                thread_count=2,
                poppler_path=poppler_path
            )
            for i, image in enumerate(images):
                path = _save_image(image, output_folder, base_name, i)
                image_paths.append(path)
                image.close()
        else:
            # Chunked conversion — keeps RAM usage constant regardless of PDF size
            for chunk_start in range(1, total_pages + 1, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE - 1, total_pages)
                images = convert_from_path(
                    pdf_path,
                    dpi=DPI,
                    fmt='jpeg',
                    thread_count=2,
                    first_page=chunk_start,
                    last_page=chunk_end,
                    poppler_path=poppler_path
                )
                for i, image in enumerate(images):
                    page_index = chunk_start - 1 + i  # 0-based global index
                    path = _save_image(image, output_folder, base_name, page_index)
                    image_paths.append(path)
                    image.close()  # free memory immediately
                del images  # release the list too

        image_paths.sort()
        print(f"✅ Saved {len(image_paths)} page(s) to: {output_folder}")
        return image_paths

    except PDFInfoNotInstalledError:
        print("❌ ERROR: Poppler is not installed or not in PATH.")
        return []
    except Exception as e:
        print(f"❌ ERROR during PDF conversion: {e}")
        return []


def _save_image(image, folder, base_name, index):
    """Save a single PIL image as JPEG and return its path."""
    filename = f"{base_name}_page_{index:04d}.jpg"
    path = os.path.join(folder, filename)
    image.save(path, "JPEG", quality=85)
    return path