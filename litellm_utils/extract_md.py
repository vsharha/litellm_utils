import base64
import io
import logging

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat, DocumentStream
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
        TableFormerMode,
        EasyOcrOptions
    )
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

logging.getLogger("docling").setLevel(logging.ERROR)

def extract_structured_md(filename: str, encoded_data: str, ocr_threshold: float = 0.1) -> str:
    if not DOCLING_AVAILABLE:
        raise ImportError(
            "Docling is not installed (used for local file processing). Install it with: pip install litellm_utils[docling]"
        )

    # mime_type, _ = mimetypes.guess_type(filename)

    table_opts = TableStructureOptions(
        mode=TableFormerMode.ACCURATE,
        do_cell_matching=True
    )

    ocr_opts = EasyOcrOptions(
        lang=["en"],
        force_full_page_ocr=False,
        bitmap_area_threshold=ocr_threshold
    )

    pipeline_opts = PdfPipelineOptions(
        do_table_structure=True,
        table_structure_options=table_opts,
        ocr_options=ocr_opts
    )

    format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}

    converter = DocumentConverter(
        format_options=format_options
    )

    file_bytes = base64.b64decode(encoded_data)
    doc_stream = DocumentStream(name=filename, stream=io.BytesIO(file_bytes))
    result = converter.convert(doc_stream)
    md = result.document.export_to_markdown()
    return md
