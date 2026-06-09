"""
Supernova Spectrum Analyzer
============================
Interactive tool for measuring pEW, FWHM,
and expansion velocities from supernova spectra.

Usage
-----
1. Set the configuration variables at the bottom of this file.
2. Run the script. For each spectrum file an interactive window opens.
3. Left-click twice  → define Region 1 (absorption wing)
   Right-click twice → define Region 2 (emission wing)
4. Press 'd' → fit the continuum through the two regions
5. Press 'c' → compute pEW, FWHM, velocity and save results
6. Press 'z' → write a row of zeros (no line)
7. Press 'n' → write a row of NaNs  (spectrum too noisy/ line not measurable)
8. Press 'b' → quit

Configuration
-------------
Edit the block at the bottom of this file:
    SAMPLE_DIRECTORY      – folder containing per-supernova sub-directories
    RESULTS_BASE_DIRECTORY – where results and images are written
    REDSHIFT              – redshift used to de-redshift the spectra
    REST_WAVELENGTH       – rest wavelength (Å) of the line being measured
    FILE_DELIMITER        – delimiter used in the spectrum files (',', '\\t', …)
"""

import os
import sys
import math

import numpy as np
import pylab as plt
from matplotlib import rc
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error

import astropy.units as u
from specutils import Spectrum1D
from specutils.analysis import gaussian_fwhm

# ── Font configuration ───────────────────────────────────────────────────────
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Georgia'], 'size': '16'})

ANGSTROM = "\u00c5"


# ── Helper functions ──────────────────────────────────────────────────────────

def smooth(y, box_pts):
    """Boxcar-smooth an array."""
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')


def zero_crossings(x, y):
    """Return indices where y changes sign."""
    crossings = np.where(np.diff(np.sign(y)))[0]
    if len(crossings) >= 1:
        return crossings
    derivative_changes = np.abs(np.diff(np.diff(y)))
    return [np.argmax(derivative_changes)]


def find_transition_points(nflux):
    """Return indices where the normalised flux crosses 1."""
    transitions = []
    for i in range(1, len(nflux) - 1):
        if (nflux[i - 1] < 1 and nflux[i] >= 1) or (nflux[i - 1] >= 1 and nflux[i] < 1):
            transitions.append(i)
    return transitions


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / (2 * stddev)) ** 2)


# ── Main class ────────────────────────────────────────────────────────────────

class SpectrumAnalyzer:
    """
    Interactive spline-fitting routine for supernova spectral analysis.

    Parameters
    ----------
    wave : array-like
        Observed wavelength array (Å).
    flux : array-like
        Flux array.
    supernova_name : str
        Name shown in the plot title.
    redshift : float
        Redshift used to de-redshift the spectrum.
    rest_wavelength : float
        Rest wavelength (Å) of the spectral line being measured.
    filename : str, optional
        Base name for output files (no extension).
    """

    def __init__(self, wave, flux, supernova_name,
                 redshift=0.0, rest_wavelength=6563,
                 filename=None):

        wave = wave / (1 + redshift)
        mask = ~np.isnan(flux)
        self.wave = wave[mask]
        self.flux = flux[mask]
        self.rest_wavelength = rest_wavelength

        print("""
        ╔══════════════════════════════════════════════════════════════════════╗
        ║ Spectral User-Guided Absorption-emission Analyzer Redshift-corrected ║
        ║                              (SUGAAR.py)                             ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  SMOOTHING:                                                          ║
        ║    Type degree in box + click Enter/Apply                            ║
        ║  REGIONS:                                                            ║
        ║    Left-click  × 2  →  Region 1 (blue wing)                          ║
        ║    Right-click × 2  →  Region 2 (red wing)                           ║
        ║    (include both absorption and emission components)                 ║
        ║  KEYS:                                                               ║
        ║    d  →  fit continuum through the two regions                       ║
        ║    r  →  reset regions and start again                               ║
        ║    c  →  compute pEW, FWHM, velocity + save                          ║
        ║    z  →  save row of zeros (no line)                                 ║
        ║    n  →  save row of NaNs  (too noisy)                               ║
        ║    b  →  quit/kill the program                                       ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """)

        self.supernova_name = supernova_name
        self.filename = filename
        print(f"  Processing: {os.path.basename(self.filename)}\n")

        # Continuum
        self.cont_coords = None
        self.continuum_slope = None

        # Region boundaries (set by mouse clicks)
        self.start_region = self.end_region = None
        self.start_region2 = self.end_region2 = None
        self.region = self.region2 = None

        # Intermediate results
        self.max_max_slope_values = []
        self.max_max_slope_values2 = []
        self.max_positive_idx = None
        self.max_negative_idx = None
        self.start_idx_negative = self.end_idx_negative = None
        self.start_idx_positive = self.end_idx_positive = None

        self.x = self.y = None
        self.true1 = self.true2 = None
        self.trueFWHM1 = self.trueFWHM2 = None
        self.pEW_values = self.FWHM_values = None
        self.velocity1 = self.velocity2 = None
        self.x01 = self.x02 = None

        self.smoothed_flux = None
        self.error_smooth = None
        self.mean_error_smooth = None
        self.degree_of_smoothing = None
        self.apply_button_clicked = False

        self.spectrum = Spectrum1D(
            spectral_axis=self.wave * u.Unit('Angstrom'),
            flux=self.flux * u.Unit('erg / (cm2 s Angstrom)')
        )

        # Build the figure
        fig = plt.figure()
        self.fig = fig
        self.ax = plt.gca()

        fig.canvas.mpl_connect('key_press_event', self.ontype)
        fig.canvas.mpl_connect('button_press_event', self.onclick2)
        fig.canvas.mpl_connect('pick_event', self.onpick)

        plt.title(f"Supernova: {self.supernova_name}")
        self.ax.plot(self.wave, self.flux, 'k-', label='spectrum')
        self._setup_input_dialog()
        plt.show()

    # ── GUI helpers ───────────────────────────────────────────────────────────

    def _setup_input_dialog(self):
        axbox = self.fig.add_axes([0.55, 0.005, 0.25, 0.05])
        self.text_box = TextBox(axbox, 'Degree of Smoothing:')
        self.text_box.on_submit(self.handle_input)

        axbutton = self.fig.add_axes([0.8, 0.005, 0.1, 0.05])
        self.button = Button(axbutton, 'Apply')
        self.button.on_clicked(self.handle_input)


    def handle_input(self, event):
        if self.apply_button_clicked:
            return
        text = self.text_box.text
        try:
            self.degree_of_smoothing = int(float(text))
            self.smoothed_flux = smooth(self.flux, box_pts=self.degree_of_smoothing)
            self.ax.plot(self.wave, self.smoothed_flux, color='skyblue')

            self.error_smooth = np.abs(self.flux - self.smoothed_flux)
            self.mean_error_smooth = self.error_smooth.mean()

            self.flux = self.smoothed_flux
            self.apply_button_clicked = True
        except ValueError:
            print("Invalid smoothing degree — please enter an integer.")

        self.start_region = None
        self.fig.canvas.draw_idle()

    # ── Mouse interaction ─────────────────────────────────────────────────────

    def onclick2(self, event):
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode != '':
            return

        if event.button == 1:
            if self.start_region is None:
                self.start_region = event.xdata
            else:
                self.end_region = event.xdata
                self.region = [self.start_region, self.end_region]
                rect = Rectangle(
                    (self.start_region, np.min(self.flux)),
                    self.end_region - self.start_region,
                    np.max(self.flux) - np.min(self.flux),
                    alpha=0.3, facecolor='green', label='Region 1'
                )
                self.ax.add_patch(rect)
                self.ax.legend(handles=[rect])

        elif event.button == 3:
            if self.start_region2 is None:
                self.start_region2 = event.xdata
            else:
                self.end_region2 = event.xdata
                self.region2 = [self.start_region2, self.end_region2]
                rect2 = Rectangle(
                    (self.start_region2, np.min(self.flux)),
                    self.end_region2 - self.start_region2,
                    np.max(self.flux) - np.min(self.flux),
                    alpha=0.3, facecolor='red', label='Region 2'
                )
                self.ax.add_patch(rect2)
                self.ax.legend(handles=[rect2])

        plt.draw()

    def onpick(self, event):
        if event.mouseevent.button == 2:
            if hasattr(event.artist, 'get_label') and event.artist.get_label() == 'cont_pnt':
                event.artist.remove()

    # ── Analysis methods ──────────────────────────────────────────────────────

    def derivative(self, start_wavelength, end_wavelength):
        """Fit continuum anchor points via the derivative method."""
        mask = (self.wave >= start_wavelength) & (self.wave <= end_wavelength)
        xvals, yvals = self.wave[mask], self.flux[mask]

        deriv = np.gradient(yvals, xvals)
        delta = end_wavelength - start_wavelength
        max_idx = np.where(np.abs(xvals - end_wavelength) < delta)[0]
        crossings = zero_crossings(xvals[max_idx], deriv[max_idx])

        if len(crossings) == 1:
            i = crossings[0]
            slope_wavelengths = [xvals[max_idx][i], xvals[max_idx][i]]
            slope_values = [yvals[max_idx][i], yvals[max_idx][i]]
        else:
            diffs = np.diff(yvals[max_idx][crossings])
            best = np.argmax(diffs)
            slope_wavelengths = [
                xvals[max_idx][crossings[best]],
                xvals[max_idx][crossings[best + 1]]
            ]
            slope_values = [
                yvals[max_idx][crossings[best]],
                yvals[max_idx][crossings[best + 1]]
            ]

        if not self.max_max_slope_values:
            self.max_max_slope_values = (slope_wavelengths, slope_values)
        else:
            self.max_max_slope_values2 = (slope_wavelengths, slope_values)

    def continuum(self, wave, flux):
        """Fit a linear continuum between the two anchor regions."""
        wavelength1, flux1 = self.max_max_slope_values
        wavelength2, flux2 = self.max_max_slope_values2

        self.continuum_slope = (flux1[1] - flux2[1]) / (wavelength1[1] - wavelength2[1])
        b = flux1[1] - self.continuum_slope * wavelength1[1]

        mask = (self.wave >= self.start_region) & (self.wave <= self.end_region2)
        xvals, yvals = self.wave[mask], self.flux[mask]
        x_cont = np.linspace(wavelength1[1], wavelength2[1], num=len(xvals))
        y_cont = self.continuum_slope * x_cont + b

        self.cont_coords = np.vstack((x_cont, y_cont))

        plt.figure()
        plt.plot(xvals, yvals, color='k')
        plt.plot(x_cont, y_cont, color='r', linestyle='--')
        plt.scatter(wavelength1[1], flux1[1], color='purple')
        plt.scatter(wavelength2[1], flux2[1], color='purple')
        plt.show()

    def pEW(self, wave, flux):
        """Compute pseudo-equivalent widths and FWHM for absorption/emission."""
        wavelength1, flux1 = self.max_max_slope_values
        wavelength2, flux2 = self.max_max_slope_values2

        mask = (self.wave >= wavelength1[1]) & (self.wave <= wavelength2[1])
        xvals, yvals = self.wave[mask], self.flux[mask]
        self.x = xvals
        self.y = yvals

        nflux = yvals / (self.continuum_slope * (xvals - xvals[0]) + yvals[0])
        transition_points_flux = find_transition_points(nflux)

        tran_point_cont_idx = [0, len(self.x) - 1]
        transition_points = np.unique(
            np.concatenate((tran_point_cont_idx, transition_points_flux))
        ).astype(int)

        plt.figure()
        plt.plot(xvals, nflux)
        plt.axhline(y=1.0, color='red', linestyle='--')
        plt.scatter(xvals[transition_points], nflux[transition_points],
                    color='k', marker='o', label='Transition Points')

        pEW_values = []
        FWHM_values = []

        if len(transition_points_flux) < 2:
            start_idx = np.searchsorted(xvals, wavelength1[1])
            end_idx = np.searchsorted(xvals, wavelength2[1])
            region_wavelength = xvals[start_idx:end_idx]
            region_flux = nflux[start_idx:end_idx]

            norm_spec = Spectrum1D(
                spectral_axis=region_wavelength * u.Unit('Angstrom'),
                flux=region_flux * u.Unit('erg / (cm2 s Angstrom)')
            )
            fwhm_val = gaussian_fwhm(norm_spec, regions=None).value

            region_pEW = 0
            for j in range(start_idx, end_idx):
                delta = xvals[j + 1] - xvals[j]
                region_pEW += delta * (1 - nflux[j]) if nflux[j] < 1 else -delta * (nflux[j] - 1)

            pEW_values.append(region_pEW)
            FWHM_values.append(fwhm_val)

        else:
            for i in range(1, len(transition_points)):
                start_idx = transition_points[i - 1]
                end_idx = transition_points[i]
                region_wavelength = xvals[start_idx:end_idx]
                region_flux = nflux[start_idx:end_idx]

                norm_spec = Spectrum1D(
                    spectral_axis=region_wavelength * u.Unit('Angstrom'),
                    flux=region_flux * u.Unit('erg / (cm2 s Angstrom)')
                )
                fwhm_val = gaussian_fwhm(norm_spec, regions=None).value

                region_pEW = 0
                for j in range(start_idx, end_idx):
                    delta = xvals[j + 1] - xvals[j]
                    if nflux[j] < 1:
                        region_pEW -= delta * (1 - nflux[j])
                    else:
                        region_pEW += delta * (nflux[j] - 1)

                pEW_values.append(region_pEW)
                FWHM_values.append(fwhm_val)

        self.pEW_values = pEW_values
        self.FWHM_values = FWHM_values

        # Identify absorption (most negative) and emission (most positive) regions
        if any(p > 0 for p in pEW_values):
            self.max_positive_idx = int(np.argmax(pEW_values))
        else:
            self.max_positive_idx = float('nan')

        self.max_negative_idx = int(np.argmin(pEW_values))

        max_negative_pEW = pEW_values[self.max_negative_idx]
        max_negativeFWHM = FWHM_values[self.max_negative_idx]

        if np.isnan(self.max_positive_idx):
            max_positive_pEW = float('nan')
            max_positiveFWHM = float('nan')
            self.start_idx_positive = self.end_idx_positive = float('nan')
            self.start_idx_negative = start_idx
            self.end_idx_negative = end_idx
            plt.fill_between(xvals[start_idx:end_idx], 1.0, nflux[start_idx:end_idx],
                             color='orange', alpha=0.5, label='Absorption region')
        else:
            max_positive_pEW = pEW_values[self.max_positive_idx]
            max_positiveFWHM = FWHM_values[self.max_positive_idx]

            s = transition_points[self.max_positive_idx]
            e = transition_points[self.max_positive_idx + 1]
            plt.fill_between(xvals[s:e], 1.0, nflux[s:e],
                             color='lightblue', alpha=0.5, label='Emission region')
            self.start_idx_positive, self.end_idx_positive = s, e

            sa = transition_points[self.max_negative_idx]
            ea = transition_points[self.max_negative_idx + 1]
            plt.fill_between(xvals[sa:ea], 1.0, nflux[sa:ea],
                             color='lightgreen', alpha=0.5, label='Absorption region')
            self.start_idx_negative, self.end_idx_negative = sa, ea

        if self.filename:
            img_path = os.path.join(images_directory, os.path.basename(self.filename) + '.png')
            plt.savefig(img_path)
        plt.show()

        self.true1 = max_negative_pEW
        self.true2 = max_positive_pEW
        self.trueFWHM1 = max_negativeFWHM
        self.trueFWHM2 = max_positiveFWHM

        return max_positive_pEW, max_negative_pEW, max_positiveFWHM, max_negativeFWHM

    def _safe_region(self, wave, flux, w_start, w_end):
        """
        Extract a wavelength/flux sub-array, clamping boundaries to the
        actual spectrum range and ensuring start < end.
        Returns (xvals, yvals) or (None, None) if the region is empty.
        """
        w_start = max(w_start, wave[0])
        w_end   = min(w_end,   wave[-1])
        if w_start >= w_end:
            return None, None
        mask = (wave >= w_start) & (wave <= w_end)
        xvals, yvals = wave[mask], flux[mask]
        if len(xvals) < 2:
            return None, None
        return xvals, yvals

    def _region_pEW_fwhm(self, xvals, yvals):
        """Normalise a region and return (pEW, fwhm). Returns (nan, nan) on failure."""
        try:
            nflux = yvals / (self.continuum_slope * (xvals - xvals[0]) + yvals[0])
            norm_spec = Spectrum1D(
                spectral_axis=xvals * u.Unit('Angstrom'),
                flux=nflux * u.Unit('erg / (cm2 s Angstrom)')
            )
            fwhm_val = gaussian_fwhm(norm_spec, regions=None).value
            pEW_val = 0.0
            for j in range(len(xvals) - 1):
                delta = xvals[j + 1] - xvals[j]
                pEW_val += -delta * (1 - nflux[j]) if nflux[j] < 1 else delta * (nflux[j] - 1)
            return pEW_val, fwhm_val
        except Exception:
            return float('nan'), float('nan')

    def _fit_velocity(self, xvals, nflux, is_absorption=True):
        """
        Fit a Gaussian to a normalised region and return the expansion velocity.
        Returns nan if the fit fails or there are too few points.
        """
        def gaus(x, y0, a, x0, sigma):
            return y0 + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        if len(xvals) <= 4:
            return float('nan')

        try:
            norm_spec = Spectrum1D(
                spectral_axis=xvals * u.Unit('Angstrom'),
                flux=nflux * u.Unit('erg / (cm2 s Angstrom)')
            )
            fwhm_val = gaussian_fwhm(norm_spec, regions=None).value

            if is_absorption:
                peak_idx = np.argmin(nflux)
                amplitude = nflux[peak_idx] - 1.0
            else:
                peak_idx = np.argmax(nflux)
                amplitude = nflux[peak_idx] - 1.0

            p0 = [1.0, amplitude, xvals[peak_idx], max(fwhm_val / 2.355, 1.0)]
            popt, _ = curve_fit(gaus, xvals, nflux, p0=p0, maxfev=5000)
            x0 = popt[2]
        except (RuntimeError, TypeError, Exception):
            return float('nan')

        z = (x0 - self.rest_wavelength) / self.rest_wavelength
        return 299792.458 * ((z + 1) ** 2 - 1) / ((z + 1) ** 2 + 1)

    def _mc_errors(self, wave, flux, n_trials=200):
        """
        Estimate measurement uncertainties by randomly perturbing the integration
        boundaries within ±20% of the measured FWHM on each side.

        This captures the uncertainty from where exactly the boundaries are placed —
        i.e. how much the result would change if a different user clicked slightly
        differently. Returns (e_pEW1, e_pEW2, e_fwhm1, e_fwhm2, mean_v1, mean_v2,
        e_vel1, e_vel2).
        """
        wave_st1  = self.x[self.start_idx_negative]
        wave_end1 = self.x[self.end_idx_negative]

        has_emission = not np.isnan(self.max_positive_idx)
        if has_emission:
            wave_st2  = self.x[self.start_idx_positive]
            wave_end2 = self.x[self.end_idx_positive]
        else:
            wave_st2 = wave_end2 = float('nan')

        # Scatter scale: 20% of each line's FWHM, minimum 2 Å
        scatter1 = max(0.20 * self.trueFWHM1, 2.0) if not np.isnan(self.trueFWHM1) else 2.0
        scatter2 = max(0.20 * self.trueFWHM2, 2.0) if (has_emission and not np.isnan(self.trueFWHM2)) else 2.0

        pEW1_trials  = []
        fwhm1_trials = []
        vel1_trials  = []
        pEW2_trials  = []
        fwhm2_trials = []
        vel2_trials  = []

        for _ in range(n_trials):
            # Perturb absorption boundaries
            st1 = wave_st1  + np.random.uniform(-scatter1, scatter1)
            en1 = wave_end1 + np.random.uniform(-scatter1, scatter1)
            xv1, yv1 = self._safe_region(wave, flux, st1, en1)
            if xv1 is not None:
                p, f = self._region_pEW_fwhm(xv1, yv1)
                nf1 = yv1 / (self.continuum_slope * (xv1 - xv1[0]) + yv1[0])
                v = self._fit_velocity(xv1, nf1, is_absorption=True)
            else:
                p = f = v = float('nan')
            pEW1_trials.append(p)
            fwhm1_trials.append(f)
            vel1_trials.append(v)

            # Perturb emission boundaries
            if has_emission:
                st2 = wave_st2  + np.random.uniform(-scatter2, scatter2)
                en2 = wave_end2 + np.random.uniform(-scatter2, scatter2)
                xv2, yv2 = self._safe_region(wave, flux, st2, en2)
                if xv2 is not None:
                    p2, f2 = self._region_pEW_fwhm(xv2, yv2)
                    nf2 = yv2 / (self.continuum_slope * (xv2 - xv2[0]) + yv2[0])
                    v2 = self._fit_velocity(xv2, nf2, is_absorption=False)
                else:
                    p2 = f2 = v2 = float('nan')
                pEW2_trials.append(p2)
                fwhm2_trials.append(f2)
                vel2_trials.append(v2)

        e_pEW1  = np.nanstd(pEW1_trials)
        e_fwhm1 = np.nanstd(fwhm1_trials)
        mean_v1 = np.nanmean(vel1_trials)
        e_vel1  = np.nanstd(vel1_trials)

        if has_emission:
            e_pEW2  = np.nanstd(pEW2_trials)
            e_fwhm2 = np.nanstd(fwhm2_trials)
            mean_v2 = np.nanmean(vel2_trials)
            e_vel2  = np.nanstd(vel2_trials)
        else:
            e_pEW2 = e_fwhm2 = mean_v2 = e_vel2 = float('nan')

        # Plot the fit for the absorption velocity (for inspection)
        xv1_ref, yv1_ref = self._safe_region(wave, flux, wave_st1, wave_end1)
        if xv1_ref is not None:
            def gaus(x, y0, a, x0, sigma):
                return y0 + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
            nf1_ref = yv1_ref / (self.continuum_slope * (xv1_ref - xv1_ref[0]) + yv1_ref[0])
            plt.figure()
            plt.plot(xv1_ref, nf1_ref, 'k-', label='Absorption region')
            plt.axhline(1.0, color='red', linestyle='--')
            peak_idx = np.argmin(nf1_ref)
            fwhm_ref = gaussian_fwhm(
                Spectrum1D(spectral_axis=xv1_ref * u.Unit('Angstrom'),
                           flux=nf1_ref * u.Unit('erg / (cm2 s Angstrom)')),
                regions=None
            ).value
            p0 = [1.0, nf1_ref[peak_idx] - 1.0, xv1_ref[peak_idx], max(fwhm_ref / 2.355, 1.0)]
            try:
                popt, _ = curve_fit(gaus, xv1_ref, nf1_ref, p0=p0, maxfev=5000)
                plt.plot(xv1_ref, gaus(xv1_ref, *popt), 'b--', label='Gaussian fit')
                plt.axvline(popt[2], color='blue', linestyle=':', label=f'x0 = {popt[2]:.1f} Å')
            except Exception:
                pass
            plt.title('Absorption velocity fit')
            plt.legend()
            plt.show()
           
        # Plot the fit for the emission velocity (for inspection)
        if has_emission:
            xv2_ref, yv2_ref = self._safe_region(wave, flux, wave_st2, wave_end2)
            if xv2_ref is not None:
                def gaus(x, y0, a, x0, sigma):
                    return y0 + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
                nf2_ref = yv2_ref / (self.continuum_slope * (xv2_ref - xv2_ref[0]) + yv2_ref[0])
                plt.figure()
                plt.plot(xv2_ref, nf2_ref, 'k-', label='Emission region')
                plt.axhline(1.0, color='red', linestyle='--')
                peak_idx = np.argmax(nf2_ref)
                fwhm_ref = gaussian_fwhm(
                    Spectrum1D(spectral_axis=xv2_ref * u.Unit('Angstrom'),
                               flux=nf2_ref * u.Unit('erg / (cm2 s Angstrom)')),
                    regions=None
                ).value
                p0 = [1.0, nf2_ref[peak_idx] - 1.0, xv2_ref[peak_idx], max(fwhm_ref / 2.355, 1.0)]
                try:
                    popt, _ = curve_fit(gaus, xv2_ref, nf2_ref, p0=p0, maxfev=5000)
                    plt.plot(xv2_ref, gaus(xv2_ref, *popt), 'b--', label='Gaussian fit')
                    plt.axvline(popt[2], color='blue', linestyle=':', label=f'x0 = {popt[2]:.1f} Å')
                except Exception:
                    pass
                plt.title('Emission velocity fit')
                plt.legend()
                plt.show()
        
        return e_pEW1, e_pEW2, e_fwhm1, e_fwhm2, mean_v1, mean_v2, e_vel1, e_vel2

    def _st_error(self, true, measurements):
        """Root-mean-square deviation used as uncertainty estimate.
        Returns nan if the reference value or all measurements are nan."""
        measurements = np.asarray(measurements, dtype=float)
        if np.isnan(true):
            return float('nan')
        valid = measurements[~np.isnan(measurements)]
        if len(valid) == 0:
            return float('nan')
        return np.sqrt(np.sum((valid - true) ** 2) / (len(valid) * 2))

    # ── Keyboard events ───────────────────────────────────────────────────────

    def ontype(self, event):

        if event.key == 'd':
            self.derivative(self.start_region, self.end_region)
            self.derivative(self.start_region2, self.end_region2)
            self.continuum(self.wave, self.flux)

        elif event.key == 'r':
            self.start_region = self.end_region = None
            self.start_region2 = self.end_region2 = None
            self.max_max_slope_values = []
            self.max_max_slope_values2 = []
            for artist in self.ax.artists:
                if hasattr(artist, 'get_label') and artist.get_label() in ('Region 1', 'Region 2'):
                    artist.remove()
            self.fig.canvas.mpl_connect('button_press_event', self.onclick2)

        elif event.key == 'c':
            pew1, pew2, max_positiveFWHM, max_negativeFWHM = self.pEW(self.wave, self.flux)

            # Continuum error contribution
            flux_err = self.mean_error_smooth if self.mean_error_smooth is not None else 0.0
            pEW_cont_err = np.abs(self.cont_coords[0, 0] - self.cont_coords[0, 1]) * flux_err
            pEW_err = math.hypot(flux_err, pEW_cont_err)

            e_pEW1, e_pEW2, e_fwhm1, e_fwhm2, mean_v1, mean_v2, e_vel1, e_vel2 = \
                self._mc_errors(self.wave, self.flux)

            # Combine continuum error with MC boundary error in quadrature
            err1_pEW  = np.sqrt(pEW_err ** 2 + e_pEW1 ** 2)
            err2_pEW  = np.sqrt(pEW_err ** 2 + e_pEW2 ** 2)
            err1_FWHM = np.sqrt(pEW_err ** 2 + e_fwhm1 ** 2)
            err2_FWHM = np.sqrt(pEW_err ** 2 + e_fwhm2 ** 2)
            err1_vel  = np.sqrt(pEW_err ** 2 + e_vel1 ** 2)
            err2_vel  = np.sqrt(pEW_err ** 2 + e_vel2 ** 2)

            print("\n─── Results ───────────────────────────────────────")
            print(f"  pEW  absorption : {pew2:10.3f} ± {err1_pEW:.3f} {ANGSTROM}")
            print(f"  FWHM absorption : {max_negativeFWHM:10.3f} ± {err1_FWHM:.3f} {ANGSTROM}")
            print(f"  v    absorption : {mean_v1:10.1f} ± {err1_vel:.1f} km/s")
            print(f"  pEW  emission   : {pew1:10.3f} ± {err2_pEW:.3f} {ANGSTROM}")
            print(f"  FWHM emission   : {max_positiveFWHM:10.3f} ± {err2_FWHM:.3f} {ANGSTROM}")
            print(f"  v    emission   : {mean_v2:10.1f} ± {err2_vel:.1f} km/s")
            print("───────────────────────────────────────────────────\n")

            if self.filename:
                result_path = os.path.join(results_directory, f"{self.filename}.txt")
                with open(result_path, 'a') as f:
                    f.write("pEW_abs pEW_abs_err FWHM_abs FWHM_abs_err "
                            "v_abs v_abs_err pEW_em pEW_em_err "
                            "FWHM_em FWHM_em_err v_em v_em_err\n")
                    f.write(f"{pew2} {err1_pEW} {max_negativeFWHM} {err1_FWHM} "
                            f"{mean_v1} {err1_vel} {pew1} {err2_pEW} "
                            f"{max_positiveFWHM} {err2_FWHM} {mean_v2} {err2_vel}\n")

        elif event.key == 'z':
            self._write_result_row(['0'] * 12)

        elif event.key == 'n':
            self._write_result_row(['NAN'] * 12)

        elif event.key == 'b':
            sys.exit(0)

    def _write_result_row(self, values):
        label = 'ZEROS' if values[0] == '0' else 'NaNs'
        print(f"\n─── Results ({label}) ───────────────────────────────────────")
        print(f"  pEW  absorption : {values[0]} ± {values[1]} {ANGSTROM}")
        print(f"  FWHM absorption : {values[2]} ± {values[3]} {ANGSTROM}")
        print(f"  v    absorption : {values[4]} ± {values[5]} km/s")
        print(f"  pEW  emission   : {values[6]} ± {values[7]} {ANGSTROM}")
        print(f"  FWHM emission   : {values[8]} ± {values[9]} {ANGSTROM}")
        print(f"  v    emission   : {values[10]} ± {values[11]} km/s")
        print("───────────────────────────────────────────────────\n")

        if self.filename:
            result_path = os.path.join(results_directory, f"{self.filename}.txt")
            with open(result_path, 'a') as f:
                f.write("pEW_abs pEW_abs_err FWHM_abs FWHM_abs_err "
                        "v_abs v_abs_err pEW_em pEW_em_err "
                        "FWHM_em FWHM_em_err v_em v_em_err\n")
                f.write(' '.join(str(v) for v in values) + '\n')

### ============================================================================
# ── Configuration ─────────────────────────────────────────────────────────────
### ============================================================================
# Edit these variables before running.


SAMPLE_DIRECTORY = '/path/to/spectra'
RESULTS_BASE_DIRECTORY = '/path/to/results'
REDSHIFT = 0.0               # 0.0 by default
REST_WAVELENGTH = 6563       # Å  — H-alpha 6563 by default
FILE_DELIMITER = ','         # delimiter used in your spectrum files
FILENAME_PREFIX = 'NEURKETAK'

### ============================================================================
### ============================================================================

# ── Main loop ─────────────────────────────────────────────────────────────────


num_files_processed = 0

for supernova_dir in sorted(os.listdir(SAMPLE_DIRECTORY)):
    supernova_path = os.path.join(SAMPLE_DIRECTORY, supernova_dir)
    if not os.path.isdir(supernova_path):
        continue

    results_directory = os.path.join(RESULTS_BASE_DIRECTORY, f'RESULTS_{supernova_dir}')
    if os.path.exists(results_directory):
        print(f"Skipping {supernova_dir} — results directory already exists.")
        continue

    os.makedirs(results_directory)
    images_directory = results_directory

    for root, dirs, files in os.walk(supernova_path):
        files = sorted(files, key=lambda x: (x.split('_')[0], x.split('_')[1]))
        for filename in files:
            if filename.startswith(FILENAME_PREFIX):
                continue

            file_path = os.path.join(root, filename)
            data = np.loadtxt(file_path, delimiter=FILE_DELIMITER,
                              skiprows=1, usecols=(0, 1)).T
            wavelength, flux = data
            supernova_name = filename.split('_')[0]

            output_name = os.path.join(results_directory, f'{FILENAME_PREFIX}_{filename}')
            SpectrumAnalyzer(
                wavelength, flux, supernova_name,
                redshift=REDSHIFT,
                rest_wavelength=REST_WAVELENGTH,
                filename=output_name
            )
            num_files_processed += 1

print(f"\nDone. Total spectra processed: {num_files_processed}")






