#!/usr/bin/env python3
# -----------------------------
# File: catphan404_cli.py
# -----------------------------
import argparse
import numpy as np
import imageio.v2 as iio
from catphan404.analysis import Catphan404Analyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Run Catphan 404 analysis modules.")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--module', type=str, default='all',
                        choices=['all', 'uniformity', 'low_contrast', 'high_contrast', 'slice_thickness', 'geometry'],
                        help='Module to run.')
    parser.add_argument('--spacing', type=float, nargs=2, default=None,
                        help='Pixel spacing in mm: dx dy')
    parser.add_argument('--radius', type=float, default=None, help='Radius for uniformity ROIs.')
    return parser.parse_args()

def main():
    args = parse_args()
    # Load image
    image = iio.imread(args.image)
    if image.ndim > 2:
        # Convert to grayscale if needed
        image = np.mean(image, axis=2)

    # Instantiate analyzer
    analyzer = Catphan404Analyzer(image, spacing=tuple(args.spacing) if args.spacing else None)

    # Run selected module(s)
    if args.module == 'all':
        results = analyzer.run_all()
    else:
        run_method = f'run_{args.module}'
        if not hasattr(analyzer, run_method):
            print(f"Module {args.module} not implemented.")
            return
        getattr(analyzer, run_method)()
        results = analyzer.results

    # Print results
    import pprint
    pprint.pprint(results)

if __name__ == '__main__':
    main()
