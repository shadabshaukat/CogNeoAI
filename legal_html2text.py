"""
DEPRECATED: legal_html2text.py has been renamed to html2text_utils.py to remove domain-specific naming.
This shim re-exports the public API to avoid breaking existing imports.
"""

# Re-export all public functions from the generic module
from html2text_utils import (  # noqa: F401
    generate_canonical_url,
    extract_jurisdiction_and_court,
    reformat_date,
    parse_title,
    generate_doc_header,
    parse_case,
    convert_html_file,
    convert_pdf_file,
    log_html_conversion,
    streamlit_conversion_runner,
)

# Optional: expose module-level constants that were previously imported
from html2text_utils import (  # noqa: F401
    BASE_DOC_PATH,
    OUTPUT_PATH,
    BASE_URL_PREFIX,
)
