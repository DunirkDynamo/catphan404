# -----------------------------
# File: catphan404/reporting/builder.py
# -----------------------------
"""Simple HTML report builder that embeds module PNGs and numeric results."""
import os
import json
from typing import Any


def build_html_report(analyzer, outfile: str = 'catphan_report.html', title: str = 'Catphan 404 Report') -> str:
    """Generate an HTML report for the given analyzer.

    The function calls ``analyzer.plot_all(outdir=tmpdir)`` to create PNGs and then
    creates a simple HTML file embedding those images alongside the analyzer.results.

    Args:
        analyzer: Catphan404Analyzer instance with `.results` populated.
        outfile: Path to HTML file to write.
        title: Document title.

    Returns:
        str: The full HTML content (also written to ``outfile``).
    """
    import tempfile
    from html import escape

    tmpdir = tempfile.mkdtemp(prefix='catphan_report_')
    saved = analyzer.plot_all(outdir=tmpdir)

    html_parts = [f"<html><head><title>{escape(title)}</title></head><body>"]
    html_parts.append(f"<h1>{escape(title)}</h1>")
    html_parts.append('<h2>Summary</h2>')
    html_parts.append('<pre>')
    html_parts.append(escape(json.dumps(analyzer.results, indent=2)))
    html_parts.append('</pre>')

    for module, path in saved.items():
        html_parts.append(f"<h3>{escape(module)}</h3>")
        rel = os.path.basename(path)
        # copy file to same dir as outfile
        outdir = os.path.dirname(os.path.abspath(outfile)) or '.'
        dest = os.path.join(outdir, rel)
        try:
            with open(path, 'rb') as fr, open(dest, 'wb') as fw:
                fw.write(fr.read())
            img_path = rel
        except Exception:
            img_path = path
        html_parts.append(f"<img src=\"{img_path}\" style=\"max-width:800px;\"><br>")

    html_parts.append('</body></html>')
    html = '
'.join(html_parts)
    with open(outfile, 'w') as f:
        f.write(html)
    return html

# -----------------------------
"""Command-line interface for Catphan 404 analysis."""
import argparse
import json
from .io import load_image
from .analysis import Catphan404Analyzer


def main():
    """Run Catphan 404 analysis from the command line."""
    parser = argparse.ArgumentParser(description='Minimal Catphan 404 analysis')
    parser.add_argument('image', help='Path to single-slice image (DICOM, PNG, TIFF, JPG)')
    parser.add_argument('-o', '--out', help='JSON output path (optional)')
    args = parser.parse_args()

    img, meta = load_image(args.image)
    analyzer = Catphan404Analyzer(img, spacing=meta.get('Spacing'))
    results = analyzer.analyze()

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()