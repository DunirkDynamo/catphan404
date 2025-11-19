# -----------------------------
# File: catphan404/cli.py
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
    results = analyzer.run_all()

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
