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

    This class visualizes:
      - The uniformity slice image.
      - The five ROIs (center, north, south, east, west).
      - Annotated mean and std in each ROI.
    """

    def __init__(self, analyzer):
        """
        Args:
            analyzer (UniformityAnalyzer): Completed analyzer instance.
        """
        self.analyzer = analyzer
        self.results = analyzer.analyze()   # dict already JSON-compatible

    # ------------------------------------------------------------------
    def _add_roi_box(self, ax, center_xy, size, label, color="yellow"):
        """Internal helper to draw a square ROI with text."""

        cx, cy = center_xy
        half = size / 2

        # Rectangle uses (x_min, y_min) in image coordinates
        rect = patches.Rectangle(
            (cy - half, cx - half),
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
        ax.text(
            cy, cx - half - 5,   # above the ROI
            text,
            color=color,
            ha="center", va="bottom",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.4, pad=2)
        )

    # ------------------------------------------------------------------
    def plot(self):
        """Generate the uniformity figure with image, overlaid ROI histograms, errorbar plot, and boxplot."""
        
        img    = self.analyzer.image
        cy, cx = self.analyzer.center
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        ax_img = axes[0, 0]
        ax_hist = axes[0, 1]
        ax_bar = axes[1, 0]
        ax_box = axes[1, 1]
        ax_prof = axes[2, 0]
        ax_metric = axes[2, 1]
        
        # Left top: image with ROIs
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title("Uniformity Analysis")
        ax_img.set_axis_off()
        
        size   = self.analyzer.roi_size
        offset = self.analyzer.roi_offset
        
        # ROI centers
        centers = {
            "centre" : (cx, cy),
            "north"  : (cx, cy + offset),
            "south"  : (cx, cy - offset),
            "east"   : (cx + offset, cy),
            "west"   : (cx - offset, cy),
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
        labels = list(centers.keys())
        means = []
        sems = []
        roi_datas = []
        
        # Draw ROIs on image and collect data for histograms and bar plot
        for label, coord in centers.items():
            color = roi_colors.get(label, "white")
            self._add_roi_box(ax_img, coord, size, label, color)
            
            # For histogram: extract ROI data
            cx_roi, cy_roi = coord
            half = size / 2
            roi_data = img[int(cy_roi - half):int(cy_roi + half), int(cx_roi - half):int(cx_roi + half)].flatten()
            
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
            n = roi_attr.size
            std_val = self.results[label.lower()]['std']
            sem = std_val / np.sqrt(n)
            sems.append(sem)
        
        # Top right: overlaid step histograms
        ax_hist.set_title("ROI Histograms (Overlaid)")
        ax_hist.set_xlabel('HU', fontsize=14)
        ax_hist.set_ylabel('Counts', fontsize=14)
        ax_hist.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_facecolor('gray')
        
        # Bottom left: scatter plot with SEM error bars
        x_pos = np.arange(len(labels))
        for i, (x, m, s, l) in enumerate(zip(x_pos, means, sems, labels)):
            ax_bar.errorbar(x, m, yerr=s, fmt='o', markersize=10, capsize=5, color=roi_colors[l], ecolor='black', linewidth=2, alpha=0.5)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(labels)
        ax_bar.set_ylabel('Mean HU')
        ax_bar.set_title('ROI Means with Standard Error of the Mean')
        
        # Condense y-axis to range of largest error bars
        y_min = min(m - s for m, s in zip(means, sems))
        y_max = max(m + s for m, s in zip(means, sems))
        ax_bar.set_ylim(y_min, y_max)
        
        # Bottom middle: boxplot
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
          - left column (2 rows): image with segments (top) and MTF curve (bottom)
          - right column: all profiles stacked vertically with shared x-axis
        """
        # Count profiles to determine layout
        n_profiles = len(self.analyzer.profiles) if hasattr(self.analyzer, 'profiles') and self.analyzer.profiles else 0
        
        # Create figure with GridSpec for flexible layout
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(max(n_profiles, 2), 2, figure=fig, width_ratios=[1, 1], height_ratios=[1] * max(n_profiles, 2))
        
        # Left top: image with centers and sampled segments
        ax0 = fig.add_subplot(gs[:max(n_profiles, 2)//2, 0])
        ax0.imshow(self.image, cmap='gray', vmin=vmin, vmax=vmax)
        ax0.plot(self.lpx, self.lpy, 'ro', markersize=4)
        for xs, ys in zip(self.lp_x, self.lp_y):
            ax0.plot([xs[0], xs[1]], [ys[0], ys[1]], '-r', linewidth=0.8)
        ax0.set_title("CTP528 centers & sampling segments")
        ax0.axis('off')

        # Left bottom: normalized MTF curve
        ax1 = fig.add_subplot(gs[max(n_profiles, 2)//2:, 0])
        if self.lp_axis is not None and self.nMTF is not None:
            ax1.plot(self.lp_axis, self.nMTF, label='nMTF')
            for k, v in self.mtf_points.items():
                ax1.plot(v, float(k[3:]) / 100, 'o', mfc='none', label=k)
            ax1.set_xlabel('lp/mm')
            ax1.set_ylabel('Normalized MTF')
            ax1.grid(True)
            ax1.legend()
            ax1.set_title('Aggregated normalized MTF')

        # Right: all profiles stacked vertically with shared x-axis
        if n_profiles > 0:
            ax_profiles = []
            for i in range(n_profiles):
                if i == 0:
                    ax_prof = fig.add_subplot(gs[i, 1])
                else:
                    ax_prof = fig.add_subplot(gs[i, 1], sharex=ax_profiles[0])
                
                profile = self.analyzer.profiles[i]
                ax_prof.plot(range(len(profile)), profile, 'b-')
                ax_prof.set_ylabel('HU', fontsize=9)
                ax_prof.grid(True, alpha=0.3)
                ax_prof.text(0.02, 0.98, f'Profile {i}', transform=ax_prof.transAxes, 
                            fontsize=9, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, pad=2))
                
                # Only show x-label on bottom plot
                if i == n_profiles - 1:
                    ax_prof.set_xlabel('Sample Index')
                else:
                    ax_prof.tick_params(labelbottom=False)
                
                ax_profiles.append(ax_prof)

        fig.tight_layout()
        return fig




class CTP401Plotter:
    """
    Plotter for AnalyzerCTP401 results.
    Displays the ROIs overlaid on the image with their mean and std HU values.
    Color-coded, with a legend beside the plot.
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
    
    Displays detected blobs overlaid on the image with annotations for CNR,
    and provides visual summary of blob statistics.
    """

    def __init__(self, analyzer):
        """
        Args:
            analyzer (AnalyzerCTP515): Completed analyzer instance.
        """
        self.analyzer = analyzer
        self.results = analyzer.analyze()
        self.image = analyzer.image
        self.center = analyzer.center

    def plot(self):
        """
        Generate visualization of low-contrast blob detection.
        
        Shows the central 400x400 pixels with detected blobs overlaid.
        Uses adaptive contrast: clip extremes, then display 60th-100th percentiles.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Crop to central 400 pixels
        h, w = self.image.shape
        crop_size = 200
        start_y = max(0, (h - crop_size) // 2)
        start_x = max(0, (w - crop_size) // 2)
        end_y = min(h, start_y + crop_size)
        end_x = min(w, start_x + crop_size)
        cropped_image = self.image[start_y:end_y, start_x:end_x]
        
        # Apply adaptive contrast: clip extremes (1st-99th), then focus on 60th-100th percentiles
        pixels = cropped_image.flatten().astype(np.float32)
        
        # Clip extremes: remove outside 1st-99th percentiles
        vmin_clip = np.percentile(pixels, 1)
        vmax_clip = np.percentile(pixels, 99)
        clipped_pixels = np.clip(pixels, vmin_clip, vmax_clip)
        
        # Display contrast over 60th to 100th percentiles of clipped data
        vmin = np.percentile(clipped_pixels, 60)
        vmax = np.percentile(clipped_pixels, 100)
        
        # Image with detected blobs
        ax.imshow(cropped_image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"CTP515 Low-Contrast Detection ({self.results['n_detected']} blobs, central 400px)")
        ax.axis('off')
        
        # Plot phantom center (adjusted for crop)
        cy, cx = self.center[1] - start_y, self.center[0] - start_x
        ax.plot(cx, cy, 'r+', markersize=12, markeredgewidth=2)
        
        # Overlay detected blobs (adjusted for crop)
        for blob_name, blob_data in self.results['blobs'].items():
            x, y = blob_data['x'] - start_x, blob_data['y'] - start_y
            r = blob_data['r']
            cnr = blob_data['cnr']
            
            # Only draw if blob is within cropped region
            if 0 <= x < crop_size and 0 <= y < crop_size:
                # Draw circle around blob
                circle = patches.Circle((x, y), r, edgecolor='cyan', facecolor='none', linewidth=2)
                ax.add_patch(circle)
                
                # Annotate with CNR
                ax.text(x, y - r - 5, f"CNR={cnr:.1f}", color='cyan', fontsize=8, 
                       ha='center', va='top', bbox=dict(facecolor='black', alpha=0.6, pad=2))
        
        fig.tight_layout()
        return fig
