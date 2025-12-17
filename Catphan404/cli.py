# File: catphan404/cli.py
# File: catphan404/cli.py
import argparse
import json
from pathlib import Path
from .io import load_image
from .analysis import Catphan404Analyzer
from .plots.plotters import HighContrastPlotter, UniformityPlotter, CTP401Plotter, CTP515Plotter
from matplotlib import pyplot as plt

AVAILABLE_MODULES = [
    'uniformity',
    'low_contrast',
    'high_contrast',
    'slice_thickness',
    'geometry',
    'ctp401',
    'ctp515'
]

# Mapping of module -> analyzer attribute on Catphan404Analyzer
ANALYZER_ATTRS = {
    'high_contrast': '_high_contrast_analyzer',
    'uniformity'   : '_uniformity_analyzer',
    'ctp401'       : '_ctp401_analyzer',
    'ctp515'       : '_ctp515_analyzer'
}

# Mapping of module -> corresponding plotter class
PLOTTER_CLASSES = {
    'high_contrast': HighContrastPlotter,
    'uniformity'   : UniformityPlotter,
    'ctp401'       : CTP401Plotter,
    'ctp515'       : CTP515Plotter
}


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
    parser.add_argument(
        '--no-save',
        action='store_true',
        help="Do not save results to JSON (useful for dry-run/debug)"
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help="Generate diagnostic plots using separate plotter classes"
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help="Directory or file prefix to save plots (PNG)"
    )
    return parser.parse_args()


def run_cli(args):
    """Run analysis and optionally generate plots using separate plotter classes."""
    # Load image and metadata
    img, meta = load_image(args.image)
    analyzer = Catphan404Analyzer(img, spacing=meta.get('Spacing'))

    # Determine which modules to run
    modules_to_run = AVAILABLE_MODULES if 'all' in args.modules else args.modules

    # Run selected modules
    for m in modules_to_run:
        run_method = f'run_{m}'
        if hasattr(analyzer, run_method):
            getattr(analyzer, run_method)()
            print(f"✅ Ran: {run_method}()")
        else:
            print(f"⚠️  Analyzer has no method '{run_method}'. Skipping.")

    # Decide JSON output path
    out_path = Path(args.out) if args.out else Path("_".join(modules_to_run) + ".json")

    # Save results to JSON
    if not args.no_save:
        try:
            if hasattr(analyzer, "save_results_json"):
                analyzer.save_results_json(out_path)
            else:
                with open(out_path, 'w') as f:
                    json.dump(analyzer.results, f, indent=2)
            print(f"📁 Results saved to: {out_path.resolve()}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    else:
        print("⚠️  Results NOT saved (--no-save)")

    # Plotting using separate plotters
    if args.plot:
        save_plot_path = Path(args.save_plot) if args.save_plot else None

        for m in modules_to_run:
            analyzer_attr = ANALYZER_ATTRS.get(m)
            plotter_class = PLOTTER_CLASSES.get(m)

            if analyzer_attr and hasattr(analyzer, analyzer_attr) and plotter_class:
                # Instantiate plotter with the corresponding analyzer object
                plotter_obj = plotter_class(getattr(analyzer, analyzer_attr))

                try:
                    fig = plotter_obj.plot()  # Always display plot

                    # Save plot externally if requested
                    if save_plot_path:
                        if save_plot_path.is_dir():
                            target_path = save_plot_path / f"{m}.png"
                        else:
                            p = save_plot_path
                            suffix = p.suffix if p.suffix else ".png"
                            target_path = p.with_name(p.stem + f"_{m}" + suffix)
                        fig.savefig(target_path)
                        plt.show()
                        print(f"📈 Saved plot for {m} -> {target_path.resolve()}")

                except Exception as e:
                    print(f"❌ Plotting failed for module {m}: {e}")
            else:
                print(f"⚠️ No plotter available for module '{m}'")

    # Summary
    print("\nModules run:", ", ".join(modules_to_run))
    if analyzer.results:
        print("Result keys in JSON:", ", ".join(analyzer.results.keys()))
    else:
        print("No results present in analyzer.results (may be empty).")


def main():
    args = parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
