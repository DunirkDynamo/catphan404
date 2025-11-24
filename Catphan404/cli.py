# File: catphan404/cli.py
import argparse
import json
from pathlib import Path
from .io import load_image
from .analysis import Catphan404Analyzer

# List of available analysis modules
AVAILABLE_MODULES = ['uniformity', 'low_contrast', 'high_contrast', 'slice_thickness', 'geometry']

def parse_args():
    """Parse command-line arguments for the Catphan 404 CLI."""
    parser = argparse.ArgumentParser(description="Catphan 404 analysis CLI")
    parser.add_argument('image', help="Path to single-slice image (DICOM, PNG, TIFF, JPG)")
    parser.add_argument(
        '--modules', '-m',
        nargs='+',
        choices=AVAILABLE_MODULES + ['all'],
        default=['all'],
        help="Which analysis modules to run (default: all)"
    )
    parser.add_argument(
        '--out', '-o',
        type=str,
        default=None,
        help="JSON output path (default: derived from module names)"
    )
    return parser.parse_args()

def run_cli(args):
    """Run analysis based on parsed arguments."""
    # Load image
    img, meta = load_image(args.image)
    analyzer = Catphan404Analyzer(img, spacing=meta.get('Spacing'))

    # Determine which modules to run
    if 'all' in args.modules:
        modules_to_run = AVAILABLE_MODULES
    else:
        modules_to_run = args.modules

    # Run selected modules
    for m in modules_to_run:
        run_method = f'run_{m}'
        getattr(analyzer, run_method)()

    # Determine output filename
    if args.out:
        out_path = Path(args.out)
    else:
        filename = "_".join(modules_to_run) + ".json"
        out_path = Path(filename)

    # Write results to JSON
    with open(out_path, 'w') as f:
        json.dump(analyzer.results, f, indent=2)

    print(f"Results written to {out_path.resolve()}")
    print("Modules run:", ", ".join(modules_to_run))
    print("Result keys in JSON:", ", ".join(analyzer.results.keys()))

def main():
    args = parse_args()
    run_cli(args)

if __name__ == '__main__':
    main()
