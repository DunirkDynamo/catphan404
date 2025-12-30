# -----------------------------
# File: catphan404/plots/plotters.py
# -----------------------------
"""Matplotlib-based plotting helpers for Catphan modules.

Each function returns a matplotlib.Figure object. The analyzer will save these
figures when requested by ``plot_all()``.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



class UniformityPlotter:
    """
    Plotter for UniformityAnalyzer results.

    Creates a comprehensive 3x2 figure layout showing:
      - Image with ROI overlays and statistics
      - Overlaid ROI histograms
      - Statistics table with mean, std, and SEM for each ROI
      - Boxplots of ROI intensities
      - Center line profiles (vertical and horizontal)
      - Uniformity metric display

    Args:
        analyzer (UniformityAnalyzer): Completed analyzer instance with results.
    """

    def __init__(self, analyzer):
        """
        Args:
            analyzer (UniformityAnalyzer): Completed analyzer instance.
        """
        self.analyzer = analyzer
        self.results = analyzer.analyze()   # dict already JSON-compatible

    # ------------------------------------------------------------------
    def _add_roi_box(self, ax, center_xy, size, label, color="yellow", above=True):
        """Internal helper to draw a square ROI with text."""

        cx, cy = center_xy
        half   = size / 2

        # Rectangle uses (x, y) = (col, row) for matplotlib
        rect = patches.Rectangle(
            (cx - half, cy - half),
            size, 
            size,
            linewidth=1.5,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        # text slightly above the ROI
        stats = self.results[label.lower()]
        text = f"{stats['mean']:.1f} ± {stats['std']:.1f}"
        if above:
            ax.text(
                cx, cy - 2 * half,   # above the ROI
                text,
                color=color,
                ha="center", va="bottom",
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.4, pad=2)
            )
        else:
            ax.text(
                cx, cy + 2* half,   # below the ROI
                text,
                color=color,
                ha="center", va="top",
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.4, pad=2)
            )

    # ------------------------------------------------------------------
    def plot(self):
        """Generate the uniformity figure with image, overlaid ROI histograms, errorbar plot, and boxplot."""
        
        img    = self.analyzer.image
        cx, cy = self.analyzer.center
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        ax_img    = axes[0, 0]
        ax_hist   = axes[0, 1]
        ax_bar    = axes[1, 0]
        ax_box    = axes[1, 1]
        ax_prof   = axes[2, 0]
        ax_metric = axes[2, 1]
        
        # Left top: image with ROIs
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title("Uniformity Analysis")
        ax_img.set_axis_off()
        
        # self.analyzer.center is (x, y) = (col, row), so unpack as cx, cy
        cx, cy = self.analyzer.center
        size   = self.analyzer.roi_size
        offset = self.analyzer.roi_offset
        
        # ROI centers
        centers = {
            "centre" : (cx, cy),
            "north"  : (cx, cy - offset),  # Above center (smaller row)
            "south"  : (cx, cy + offset),  # Below center (larger row)
            "east"   : (cx + offset, cy),  # Right of center (larger column)
            "west"   : (cx - offset, cy),  # Left of center (smaller column)
        }
        
        # Colors for ROIs (matching the image annotations)
        roi_colors = {
            "centre" : "purple",
            "north"  : "blue",
            "south"  : "orange",
            "east"   : "green",
            "west"   : "red",
        }
        
        legend_handles = []
        labels         = list(centers.keys())
        means          = []
        sems           = []
        roi_datas      = []
        
        # Draw ROIs on image and collect data for histograms and bar plot
        for label, coord in centers.items():
            color = roi_colors.get(label, "white")
            if label == "centre" or label == "south":
                self._add_roi_box(ax_img, coord, size, label, color, above=False)
            else:
                self._add_roi_box(ax_img, coord, size, label, color)
            
            # For histogram: extract ROI data
            cx_roi, cy_roi = coord
            half           = size / 2
            roi_data       = img[int(cy_roi - half):int(cy_roi + half), int(cx_roi - half):int(cx_roi + half)].flatten()
            
            roi_datas.append(roi_data)
            
            # Plot step histogram (unfilled)
            ax_hist.hist(roi_data, histtype='step', color=color, linewidth=3, label=label)
            
            # Add mean line
            mean_val = self.results[label.lower()]['mean']
            ax_hist.axvline(mean_val, color=color, linestyle='--', linewidth=2)
            
            # Legend handle
            legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=f'{label} (mean: {mean_val:.1f})'))
            
            # Collect for bar plot
            means.append(mean_val)
            roi_attr = getattr(self.analyzer, f'm{label[0]}')  # mc, mn, ms, me, mw
            n        = roi_attr.size
            std_val  = self.results[label.lower()]['std']
            sem      = std_val / np.sqrt(n)
            sems.append(sem)
        
        # Top right: overlaid step histograms
        ax_hist.set_title("ROI Histograms (Overlaid)")
        ax_hist.set_xlabel('HU', fontsize=14)
        ax_hist.set_ylabel('Counts', fontsize=14)
        ax_hist.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_facecolor('gray')
        
        # Bottom left: statistics table
        ax_bar.axis('off')
        
        # Prepare table data
        table_data = [['ROI', 'Mean (HU)', 'Std (HU)', 'SEM (HU)']]
        for label, mean_val, sem in zip(labels, means, sems):
            std_val = self.results[label.lower()]['std']
            table_data.append([
                label.title(),
                f"{mean_val:.1f}",
                f"{std_val:.1f}",
                f"{sem:.2f}"
            ])
        
        # Create table
        table = ax_bar.table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(table_data[0])):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternate row colors and apply ROI colors
        for i in range(1, len(table_data)):
            roi_label = labels[i-1]
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                # Color-code the ROI name column
                if j == 0:
                    cell.set_text_props(color=roi_colors[roi_label], weight='bold')
        
        ax_bar.set_title('ROI Statistics', fontsize=12, weight='bold', pad=10)
        
        # Middle right: boxplot
        ax_box.boxplot(roi_datas, labels=labels, patch_artist=True)
        ax_box.set_title('ROI Boxplots')
        ax_box.set_ylabel('HU')
        
        # Color the boxes
        for patch, color in zip(ax_box.patches, [roi_colors[l] for l in labels]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Bottom left: center profiles (central 300 pixels)
        central_range = 360
        half_range = central_range//2
        center_y = img.shape[0] // 2
        start_y = max(0, center_y - half_range)
        end_y = min(img.shape[0], center_y + half_range)
        vertical_profile = img[start_y:end_y, int(cx)]
        
        center_x = img.shape[1] // 2
        start_x = max(0, center_x - half_range)
        end_x = min(img.shape[1], center_x + half_range)
        horizontal_profile = img[int(cy), start_x:end_x]
        
        ax_prof.plot(vertical_profile, label='Vertical (central {}px)'.format(central_range), color='blue')
        ax_prof.plot(horizontal_profile, label='Horizontal (central {}px)'.format(central_range), color='red')
        
        # Add vertical lines at true centers
        vert_center_idx = int(cy) - start_y
        horiz_center_idx = int(cx) - start_x
        ax_prof.axvline(vert_center_idx, color='blue', linestyle='--', linewidth=2, label='Vertical center (y={:.0f})'.format(cy))
        ax_prof.axvline(horiz_center_idx, color='red', linestyle='--', linewidth=2, label='Horizontal center (x={:.0f})'.format(cx))
        
        ax_prof.set_title('Center Profiles (Central 300 Pixels)')
        ax_prof.set_xlabel('Pixel position (relative)')
        ax_prof.set_ylabel('HU')
        ax_prof.legend()
        ax_prof.grid(True, alpha=0.3)
        
        # Bottom right: uniformity metric
        uni = self.results["uniformity"]
        ax_metric.text(0.5, 0.5, f"Uniformity: {uni:.2f} %", ha='center', va='center', fontsize=16, transform=ax_metric.transAxes)
        ax_metric.axis('off')
        
        fig.tight_layout()
        return fig





class HighContrastPlotter:
    """
    Plotter for HighContrastAnalyzer results.

    Creates a multi-panel figure showing:
      - Image with line pair centers and sampling segments
      - Normalized MTF curve with MTF10/30/50/80 markers
      - Stacked color-coded intensity profiles from line pair samples

    Args:
        analyzer (HighContrastAnalyzer): Completed analyzer instance with results.
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.results = analyzer.to_dict()  # assumes HighContrastAnalyzer has a to_dict() method
        # unpack attributes needed for plotting
        self.image = analyzer.image
        self.lpx = analyzer.lpx           # x-coordinates of centers
        self.lpy = analyzer.lpy           # y-coordinates of centers
        self.lp_x = analyzer.lp_x         # list of tuples for segment x-coords
        self.lp_y = analyzer.lp_y         # list of tuples for segment y-coords
        self.lp_axis = analyzer.lp_axis   # x-axis for MTF curve
        self.nMTF = analyzer.nMTF         # normalized MTF values
        self.mtf_points = analyzer.mtf_points  # dict of specific MTF points, e.g. {'MTF10': x, 'MTF50': y}

    def plot(self, savefile: str = None, vmin: float = None, vmax: float = None):
        """
        Plot:
          - left: image with segments (top) and MTF curve (bottom)
          - right: all profiles stacked vertically with color coding
        """
        # Count profiles to determine layout
        n_profiles = len(self.analyzer.profiles) if hasattr(self.analyzer, 'profiles') and self.analyzer.profiles else 0
        
        # Create figure with GridSpec for flexible layout
        from matplotlib.gridspec import GridSpec
        n_to_show = n_profiles if n_profiles > 0 else 4
        n_rows = max(n_to_show, 2)
        fig = plt.figure(figsize=(16, max(10, n_to_show * 2)))
        gs = GridSpec(n_rows, 2, figure=fig, width_ratios=[1, 1])
        
        # Left top: image with centers and sampled segments (spans half of rows)
        ax_img = fig.add_subplot(gs[:n_rows//2, 0])
        ax_img.imshow(self.image, cmap='gray', vmin=vmin, vmax=vmax)
        ax_img.plot(self.lpx, self.lpy, 'ro', markersize=4)
        for xs, ys in zip(self.lp_x, self.lp_y):
            ax_img.plot([xs[0], xs[1]], [ys[0], ys[1]], '-r', linewidth=0.8)
        ax_img.set_title("CTP528 centers & sampling segments")
        ax_img.axis('off')

        # Left bottom: normalized MTF curve (spans remaining rows)
        ax_mtf = fig.add_subplot(gs[n_rows//2:, 0])
        if self.lp_axis is not None and self.nMTF is not None:
            ax_mtf.plot(self.lp_axis, self.nMTF, label='nMTF', linewidth=2)
            for k, v in self.mtf_points.items():
                ax_mtf.plot(v, float(k[3:]) / 100, 'o', mfc='none', markersize=8, label=k)
            ax_mtf.set_xlabel('lp/mm')
            ax_mtf.set_ylabel('Normalized MTF')
            ax_mtf.grid(True)
            ax_mtf.legend()
            ax_mtf.set_title('Aggregated normalized MTF')

        # Right: all profiles stacked vertically with shared x-axis
        if n_to_show > 0:
            # Generate distinct colors for profiles
            if n_to_show <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))
            else:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))
            
            ax_profiles = []
            for i in range(n_to_show):
                if i == 0:
                    ax_prof = fig.add_subplot(gs[i, 1])
                else:
                    ax_prof = fig.add_subplot(gs[i, 1], sharex=ax_profiles[0])
                
                profile = self.analyzer.profiles[i]
                ax_prof.plot(range(len(profile)), profile, color=colors[i], linewidth=1.5)
                ax_prof.set_ylabel('HU', fontsize=9)
                ax_prof.grid(True, alpha=0.3)
                ax_prof.text(0.02, 0.98, f'Profile {i}', transform=ax_prof.transAxes, 
                            fontsize=9, va='top', ha='left', 
                            bbox=dict(facecolor='white', alpha=0.7, pad=2),
                            color=colors[i], weight='bold')
                
                # Only show x-label on bottom plot
                if i == n_to_show - 1:
                    ax_prof.set_xlabel('Sample Index')
                else:
                    ax_prof.tick_params(labelbottom=False)
                
                ax_profiles.append(ax_prof)

        fig.tight_layout()
        return fig
        return fig




class CTP401Plotter:
    """
    Plotter for AnalyzerCTP401 results.

    Creates a comprehensive display with:
      - Main image showing ROI locations as color-coded circles
      - Per-ROI histograms showing intensity distributions with mean/median markers
      - 2D heatmaps of pixel intensities within each ROI
      - Color-coded legend matching ROI names to materials (LDPE, Air, Teflon, Acrylic)

    Args:
        analyzer (AnalyzerCTP401): Completed analyzer instance.
        vmin (float, optional): Minimum intensity for display windowing.
        vmax (float, optional): Maximum intensity for display windowing.
    """

    def __init__(self, analyzer, vmin: float = None, vmax: float = None):
        """
        Args:
            analyzer: AnalyzerCTP401 object (already analyzed)
            vmin: optional min intensity for display
            vmax: optional max intensity for display
        """
        self.analyzer = analyzer
        self.results = analyzer.results
        self.vmin = vmin
        self.vmax = vmax

        # Define ROI mapping and colors
        self.roi_angles = {
            "ROI0_LDPE"    : 0,
            "ROI90_Air"    : 90,
            "ROI180_Teflon": 180,
            "ROI270_Acrylic": -90
        }
        self.roi_colors = {
            "ROI0_LDPE"    : "blue",
            "ROI90_Air"    : "orange",
            "ROI180_Teflon": "green",
            "ROI270_Acrylic": "red"
        }
    def plot(self):
        """Display the CTP401 image with ROIs, per-ROI histograms, and 2D heatmaps.

        Layout: left column shows the image with ROI overlays; middle column
        contains histograms; right column contains 2D heatmaps of ROI pixels.
        """
        image = getattr(self.analyzer, "image", None)
        if image is None:
            raise ValueError("Analyzer has no image to plot.")

        # Create a 4x3 GridSpec: image spans left column; middle has histograms; right has heatmaps
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(4, 3, width_ratios=[3, 1, 1], wspace=0.3, hspace=0.6)
        ax_img = fig.add_subplot(gs[:, 0])
        ax_img.imshow(image, cmap="gray", vmin=self.vmin, vmax=self.vmax)
        ax_img.set_title("CTP401 ROIs and HU Values")
        ax_img.axis("off")

        # Draw ROIs as unfilled circles and collect legend handles
        legend_handles = []
        rois = self.analyzer.results.get("ROIs", {})
        ny, nx = image.shape[:2]

        # Use the ordered keys from roi_angles to keep order stable
        roi_names = list(self.roi_angles.keys())

        for i, name in enumerate(roi_names):
            stats = rois.get(name)
            if stats is None:
                # create empty axes if ROI missing
                ax_hist = fig.add_subplot(gs[i, 1])
                ax_hist.text(0.5, 0.5, "no data", ha='center', va='center')
                ax_hist.set_title(name, fontsize=9)
                ax_hist.axis('off')
                ax_heat = fig.add_subplot(gs[i, 2])
                ax_heat.text(0.5, 0.5, "no data", ha='center', va='center')
                ax_heat.set_title(f"{name} Heatmap", fontsize=9)
                ax_heat.axis('off')
                continue

            angle_deg = self.roi_angles[name]
            color = self.roi_colors.get(name, "white")

            radius_px = 3.5 / self.analyzer.pixel_spacing
            cx = self.analyzer.center[0] + np.cos(np.radians(angle_deg)) * 58.5 / self.analyzer.pixel_spacing
            cy = self.analyzer.center[1] + np.sin(np.radians(angle_deg)) * 58.5 / self.analyzer.pixel_spacing

            # Add circle patch on main image
            circle = patches.Circle(
                (cx, cy),
                radius=radius_px,
                edgecolor=color,
                facecolor="none",
                linewidth=2,
                alpha=0.6
            )
            ax_img.add_patch(circle)

            # For legend
            label = f"{name}: {stats['mean']:.1f} ± {stats['std']:.1f}"
            legend_handles.append(patches.Circle((0, 0), radius=radius_px, edgecolor=color, facecolor="none", alpha=0.6, label=label))

            # Histogram for this ROI
            ax_hist = fig.add_subplot(gs[i, 1])
            try:
                mask = self.analyzer.create_circular_mask(ny, nx, center=(cx, cy), radius=radius_px)
                data = image[mask]
            except Exception:
                data = np.array([])

            if data.size > 0:
                # Compute histogram
                counts, bins = np.histogram(data.flatten(), bins=30)
                bin_centers = (bins[:-1] + bins[1:]) / 2.0
                # Add darker edgecolor to bars
                import matplotlib.colors as mcolors
                darker_color = mcolors.to_rgba(color, alpha=1.0)
                darker_color = tuple(max(0, c - 0.3) for c in darker_color[:3]) + (1.0,)
                ax_hist.bar(bin_centers, counts, width=(bins[1] - bins[0]), color=color, alpha=0.75, align='center', edgecolor=darker_color, linewidth=0.5)

                mean_val = float(stats.get('mean', np.mean(data)))
                std_val = float(stats.get('std', np.std(data)))
                median_val = float(np.median(data))

                # y position for markers
                ypos = counts.max() * 0.9 if counts.size else 1.0

                # Mean line (dashed)
                ax_hist.axvline(mean_val, color='k', linestyle='--', linewidth=1)

                # Median line (dotted)
                ax_hist.axvline(median_val, color='k', linestyle=':', linewidth=1)

                # Horizontal error bar
                ax_hist.errorbar([mean_val], [ypos], xerr=[std_val], fmt='none', ecolor='k', capsize=3, linewidth=1)
            else:
                ax_hist.text(0.5, 0.5, "no data", ha='center', va='center')

            # Add horizontal gridlines
            ax_hist.grid(axis='y', linestyle='--', alpha=0.7)

            # Make frame edges more pronounced
            for spine in ax_hist.spines.values():
                spine.set_linewidth(2)
                spine.set_color('black')

            ax_hist.set_title(name, fontsize=9)
            # Set xlabel only on the bottom histogram
            if i == 3:
                ax_hist.set_xlabel('HU', fontsize=8)
            ax_hist.set_ylabel('Counts', fontsize=8)
            ax_hist.tick_params(axis='both', which='major', labelsize=8)

            # 2D Heatmap for this ROI
            ax_heat = fig.add_subplot(gs[i, 2])
            if data.size > 0:
                # Create a small 2D array from the masked data (assume circular, pad to square)
                # For simplicity, reshape to a square if possible, or use imshow on the mask
                roi_img = np.zeros((int(radius_px * 2), int(radius_px * 2)))
                y_indices, x_indices = np.ogrid[:roi_img.shape[0], :roi_img.shape[1]]
                dist = np.sqrt((x_indices - radius_px) ** 2 + (y_indices - radius_px) ** 2)
                circle_mask = dist <= radius_px
                roi_img[circle_mask] = data[:circle_mask.sum()]  # Fill with data
                ax_heat.imshow(roi_img, cmap='gray', vmin=self.vmin, vmax=self.vmax)
            else:
                ax_heat.text(0.5, 0.5, "no data", ha='center', va='center')
            ax_heat.set_title(f"{name} Heatmap", fontsize=9)
            ax_heat.axis('off')

        # Add color-coded legend on the image axis
        if legend_handles:
            ax_img.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

        # Add a single legend for mean/median lines
        from matplotlib.lines import Line2D
        mean_line = Line2D([0], [0], color='k', linestyle='--', linewidth=1, label='Mean')
        median_line = Line2D([0], [0], color='k', linestyle=':', linewidth=1, label='Median')
        fig.legend(handles=[mean_line, median_line], loc='lower center', ncol=2, fontsize=10, framealpha=0.9)

        return fig


class CTP515Plotter:
    """
    Plotter for AnalyzerCTP515 (low-contrast detectability) results.

    Creates a 2x2 layout displaying:
      - Image with color-coded ROI circles and background ROI
      - Dual-axis plot of CNR and Contrast vs. ROI diameter
      - Statistics table showing mean, std, CNR, and contrast for each ROI

    ROIs are color-coded by diameter size with adaptive contrast windowing
    to enhance visibility of low-contrast features.

    Args:
        analyzer (AnalyzerCTP515): Completed analyzer instance with results.
    """

    def __init__(self, analyzer):
        """
        Args:
            analyzer (AnalyzerCTP515): Completed analyzer instance with results.
        """
        self.analyzer = analyzer
        self.results  = analyzer.results
        self.image    = analyzer.image
        self.center   = analyzer.center

    def plot(self):
        """
        Generate visualization of low-contrast ROI analysis.
        
        Layout:
          - Top left: Image with ROI overlays
          - Top right: CNR and Contrast vs. ROI Diameter (dual y-axes)
          - Bottom: Statistics table
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Create subplots
        ax_img = plt.subplot2grid((2, 2), (0, 0))
        ax_plot = plt.subplot2grid((2, 2), (0, 1))
        ax_table = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        
        # Left: Image with ROIs (no cropping)
        display_image = self.image
        
        # Apply adaptive contrast focused on background values
        # Sample center region to determine background intensity
        center_crop_size = 100
        h, w             = display_image.shape
        cy_center        = h // 2
        cx_center        = w // 2
        center_sample    = display_image[
            cy_center - center_crop_size//2 : cy_center + center_crop_size//2,
            cx_center - center_crop_size//2 : cx_center + center_crop_size//2
        ]
        bg_mean = np.mean(center_sample)
        bg_std  = np.std(center_sample)
        
        # Set contrast window centered on background: mean ± 3*std
        vmin = bg_mean - 3 * bg_std
        vmax = bg_mean + 3 * bg_std
        
        ax_img.imshow(display_image, cmap='gray', vmin=vmin, vmax=vmax)
        ax_img.set_title(f"CTP515 Low-Contrast ROIs (n={self.results['n_detected']})")
        ax_img.axis('off')
        
        # Plot phantom center (no crop adjustment needed)
        # self.center is (x, y) = (col, row) matching codebase convention
        center_col = self.center[0]
        center_row = self.center[1]
        ax_img.plot(center_col, center_row, 'r+', markersize=12, markeredgewidth=2)
        
        # Define color map for different ROI diameters
        color_map = {
            15: '#00ffff',  # cyan
            9:  '#00ff00',  # green
            8:  '#ffff00',  # yellow
            7:  '#ff8800',  # orange
            6:  '#ff00ff',  # magenta
            5:  '#ff0000',  # red
        }
        
        # Overlay ROIs and collect data for plotting
        diameters       = []
        cnrs            = []
        contrasts       = []
        legend_handles  = []
        legend_labels   = []
        
        # Draw background ROI first (in white/gray)
        # Background ROI: 38mm from center at first angle
        bg_dist_mm   = 35
        bg_radius_mm = 5
        bg_angle_deg = self.analyzer.roi_angles[0] + self.analyzer.angle_offset
        bg_angle_rad = np.radians(bg_angle_deg)
        spacing      = self.analyzer.pixel_spacing
        
        bg_dist_px   = bg_dist_mm / spacing
        bg_radius_px = bg_radius_mm / spacing
        
        # Calculate background ROI position
        cy, cx = int(self.center[1]), int(self.center[0])
        bg_x   = cx + bg_dist_px * np.cos(bg_angle_rad)
        bg_y   = cy - bg_dist_px * np.sin(bg_angle_rad)
        
        # Draw background ROI circle in white with dashed line
        bg_color = 'red'
        bg_circle = patches.Circle((bg_x, bg_y), bg_radius_px, edgecolor=bg_color, 
                                   facecolor='none', linewidth=3, linestyle='-')
        ax_img.add_patch(bg_circle)
        
        for roi_name, roi_data in self.results['blobs'].items():
            x        = roi_data['x']
            y        = roi_data['y']
            r        = roi_data['r']
            cnr      = roi_data['cnr']
            contrast = roi_data['contrast']
            
            # Extract diameter from roi name (e.g., 'roi_15mm' -> 15)
            diameter = float(roi_name.split('_')[1].replace('mm', ''))
            diameters.append(diameter)
            cnrs.append(cnr)
            contrasts.append(contrast)
            
            # Get color for this diameter
            color = color_map.get(int(diameter), 'cyan')
            
            # Draw ROI circle
            circle = patches.Circle((x, y), r, edgecolor=color, facecolor='none', linewidth=2)
            ax_img.add_patch(circle)
            
            # Add to legend (only once per diameter)
            if int(diameter) not in [int(label.split('mm')[0]) for label in legend_labels]:
                legend_handles.append(patches.Patch(facecolor='none', edgecolor=color, linewidth=2))
                legend_labels.append(f'{int(diameter)}mm')
        
        # Add background to legend after ROIs
        legend_handles.append(patches.Patch(facecolor='none', edgecolor=bg_color, linewidth=3, linestyle='-'))
        legend_labels.append('Background')
        
        # Add legend to image
        if legend_handles:
            ax_img.legend(legend_handles, legend_labels, loc='upper right', fontsize=10, 
                         framealpha=0.8, title='ROI Diameter')
        
        # Right: CNR and Contrast vs. Diameter
        # Sort by diameter for proper line plotting
        sorted_indices   = np.argsort(diameters)
        diameters_sorted = [diameters[i] for i in sorted_indices]
        cnrs_sorted      = [cnrs[i] for i in sorted_indices]
        contrasts_sorted = [contrasts[i] for i in sorted_indices]
        
        # Primary axis: CNR
        ax_plot.plot(diameters_sorted, cnrs_sorted, 'bo-', linewidth=3, markersize=8, label='CNR', alpha=0.5)
        #ax_plot.scatter(diameters_sorted, cnrs_sorted, c='blue', s=80, zorder=3, alpha=0.5)
        ax_plot.set_xlabel('ROI Diameter (mm)', fontsize=12)
        ax_plot.set_ylabel('CNR', fontsize=12, color='blue')
        ax_plot.tick_params(axis='y', labelcolor='blue')
        ax_plot.grid(True, alpha=0.3)
        
        # Secondary axis: Contrast
        ax_contrast = ax_plot.twinx()
        ax_contrast.plot(diameters_sorted, contrasts_sorted, 'ro-', linewidth=2, markersize=8, label='Contrast (%)', alpha=0.5)
        #ax_contrast.scatter(diameters_sorted, contrasts_sorted, c='red', s=80, zorder=3, alpha=0.5, marker='x')
        ax_contrast.set_ylabel('Contrast (%)', fontsize=12, color='red')
        ax_contrast.tick_params(axis='y', labelcolor='red')
        
        ax_plot.set_title('CNR and Contrast vs. ROI Diameter')
        
        # Add legends
        lines1, labels1 = ax_plot.get_legend_handles_labels()
        lines2, labels2 = ax_contrast.get_legend_handles_labels()
        ax_plot.legend(lines1 + lines2, labels1 + labels2, loc='lower right', bbox_to_anchor=(1, 0))
        
        # Bottom: Statistics table
        ax_table.axis('off')
        
        # Prepare table data
        table_data = [['Diameter (mm)', 'Mean (HU)', 'Std (HU)', 'CNR', 'Contrast (%)']]
        
        # Sort ROIs by diameter for table
        roi_list = [(float(name.split('_')[1].replace('mm', '')), name, data) 
                    for name, data in self.results['blobs'].items()]
        roi_list.sort(key=lambda x: x[0])
        
        for diameter, roi_name, roi_data in roi_list:
            table_data.append([
                f"{diameter:.0f}",
                f"{roi_data['mean']:.1f}",
                f"{roi_data['std']:.1f}",
                f"{roi_data['cnr']:.2f}",
                f"{roi_data['contrast']:.2f}"
            ])
        
        # Add background row (get from first ROI's bg_mean and bg_std)
        if roi_list:
            bg_data = roi_list[0][2]  # Get first ROI data
            table_data.append([
                'Background',
                f"{bg_data['bg_mean']:.1f}",
                f"{bg_data['bg_std']:.1f}",
                '—',
                '—'
            ])
        
        # Create table
        table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                              colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(table_data[0])):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
        
        ax_table.set_title('ROI Statistics', fontsize=12, weight='bold', pad=10)
        
        fig.tight_layout()
        return fig
