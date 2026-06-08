# SUGAAR
### Spectral User-Guided Absorption-emission Analyzer Redshift-corrected

Interactive Python tool for measuring spectral features of supernova spectra. Given a 1D spectrum (wavelength, flux), the user interactively selects two continuum anchor regions; SUGAAR fits a linear continuum via a derivative method, normalises the profile, and automatically identifies and measures both the absorption trough and emission peak of a spectral line. Uncertainties are estimated via Monte Carlo boundary perturbation (200 trials scaled to the measured FWHM). Results are saved to text files for further analysis.

---

## Features

- Interactive matplotlib GUI — region selection by mouse click, smoothing widget
- Redshift correction at load time
- Linear continuum fitting via derivative-based anchor detection
- Automatic separation of absorption and emission components
- pEW, FWHM and relativistic Doppler velocity for both components
- MC uncertainty estimation by randomly perturbing integration boundaries within ±20% of the line FWHM
- Graceful handling of emission-free spectra (returns NaN for emission quantities)
- Batch processing of multiple supernova directories
- Results and diagnostic plots saved automatically

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/SUGAAR.git
cd SUGAAR
pip install numpy scipy matplotlib astropy specutils scikit-learn
```

---

## Usage

### 1. Configure
Edit the configuration block at the bottom of `SUGAAR.py`:

```python
SAMPLE_DIRECTORY       = '/path/to/spectra'
RESULTS_BASE_DIRECTORY = '/path/to/results'
REDSHIFT               = 0.0      # redshift of the target
REST_WAVELENGTH        = 6563     # Å — H-alpha by default
FILE_DELIMITER         = ','      # delimiter used in your spectrum files
FILENAME_PREFIX        = 'MEASUREMENTS'
```

### 2. Run

```bash
python SUGAAR.py
```

An interactive window opens for each spectrum file found in `SAMPLE_DIRECTORY`.

### 3. Interact

| Action | Effect |
|--------|--------|
| Left-click × 2 | Define Region 1 (blue wing) |
| Right-click × 2 | Define Region 2 (red wing) |
| `d` | Fit continuum through the two regions |
| `c` | Compute pEW, FWHM, velocity and save results |
| `r` | Reset regions and start again |
| `z` | Save a row of zeros (line not measurable) |
| `n` | Save a row of NaNs (spectrum too noisy) |
| `b` | Quit |


## Output

For each spectrum, a `.txt` file is saved in `RESULTS_<supernova>/` with the following 12 columns:

| Column | Description |
|--------|-------------|
| `pEW_abs` | Pseudo-equivalent width of absorption (Å) |
| `pEW_abs_err` | Uncertainty on pEW absorption (Å) |
| `FWHM_abs` | Full width at half maximum of absorption (Å) |
| `FWHM_abs_err` | Uncertainty on FWHM absorption (Å) |
| `v_abs` | Expansion velocity from absorption (km/s) |
| `v_abs_err` | Uncertainty on absorption velocity (km/s) |
| `pEW_em` | Pseudo-equivalent width of emission (Å) |
| `pEW_em_err` | Uncertainty on pEW emission (Å) |
| `FWHM_em` | Full width at half maximum of emission (Å) |
| `FWHM_em_err` | Uncertainty on FWHM emission (Å) |
| `v_em` | Expansion velocity from emission (km/s) |
| `v_em_err` | Uncertainty on emission velocity (km/s) |

---

## Method

### Continuum
Two anchor regions are selected interactively (blue and red wings of the line). Within each region, the point of maximum derivative change is identified and used as the continuum anchor. A straight line is fit between the two anchors.

### pEW
The spectrum is normalised by the continuum. The normalised profile is split into absorption (flux < 1) and emission (flux > 1) segments. The pseudo-equivalent width is computed by numerical integration:

$$\mathrm{pEW} = -\int \left(1 - \frac{F_\lambda}{F_{\rm cont}}\right) d\lambda$$

### Velocity
A Gaussian is fit to the absorption trough of the normalised profile. The centroid $\lambda_0$ is converted to an expansion velocity via the relativistic Doppler formula:

$$v = c \cdot \frac{\left(\lambda_0 / \lambda_{\rm rest}\right)^2 - 1}{\left(\lambda_0 / \lambda_{\rm rest}\right)^2 + 1}$$

### Uncertainties
Uncertainties are estimated by combining two contributions in quadrature:
- **Continuum placement error**: propagated from the smoothing residuals
- **Boundary placement error**: 200 Monte Carlo trials in which the integration boundaries are randomly perturbed within ±20% of the measured FWHM; the standard deviation of the resulting measurements is taken as the uncertainty

---

## Dependencies

- Python 3.8+
- `numpy`
- `scipy`
- `matplotlib`
- `astropy`
- `specutils`
- `scikit-learn`

---

## Citation

If you use SUGAAR in your research, please cite this repository:

```
@software{SUGAAR,
  author = {Your Name},
  title  = {SUGAAR: Spectral User-Guided Absorption-emission Analyzer Redshift-corrected},
  year   = {2026},
  url    = {https://github.com/yourusername/SUGAAR}
}
```
