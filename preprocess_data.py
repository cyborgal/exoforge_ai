import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

from astropy.io import fits
from astropy.table import Table
import requests

# Download sample light curve (e.g., for a known exoplanet like Kepler-10b)
url = "https://archive.stsci.edu/missions/kepler/light_curves/0114/011467064/kplr011467064-2010000000000_llc.fits"
response = requests.get(url)
with open('sample_light_curve.fits', 'wb') as f:
    f.write(response.content)

# Load and extract flux/time (brightness over time)
with fits.open('sample_light_curve.fits') as hdul:
    light_curve = Table(hdul[1].data)
time = light_curve['TIME']  # Days
flux = light_curve['PDCSAP_FLUX']  # Brightness
print("Sample Light Curve Loaded: Time shape", time.shape, "Flux shape", flux.shape)

# Load the three CSV files (adjust skiprows based on your line counts)
kepler_data = pd.read_csv('cumulative.csv', skiprows=53)  # Use your exact number, e.g., 53
tess_data = pd.read_csv('TOI.csv', skiprows=69)  # Adjust to your count
k2_data = pd.read_csv('k2pandc.csv', skiprows=98)  # Adjust to your count

# Print basic info to explore
print("Kepler Data Shape:", kepler_data.shape)
print(kepler_data.columns.tolist())
print("\nTESS Data Shape:", tess_data.shape)
print(tess_data.columns.tolist())
print("\nK2 Data Shape:", k2_data.shape)
print(k2_data.columns.tolist())

# Standardize labels for each dataset
# Kepler: 'CONFIRMED' -> 2, 'CANDIDATE' -> 1, 'FALSE POSITIVE' -> 0
kepler_data['label'] = kepler_data['koi_disposition'].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})

# TESS: 'CP'/'KP' -> 2 (confirmed/known), 'PC'/'APC' -> 1 (candidate), 'FP' -> 0 (false positive)
tess_data['label'] = tess_data['tfopwg_disp'].map({'CP': 2, 'KP': 2, 'PC': 1, 'APC': 1, 'FP': 0})

# K2: 'CONFIRMED' -> 2, 'CANDIDATE' -> 1, 'FALSE POSITIVE' -> 0 (or similar; data uses uppercase)
k2_data['label'] = k2_data['disposition'].map({'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})

# Drop rows with missing or unknown labels
kepler_data = kepler_data.dropna(subset=['label'])
tess_data = tess_data.dropna(subset=['label'])
k2_data = k2_data.dropna(subset=['label'])

# Print label counts to check balance
print("\nKepler Labels:\n", kepler_data['label'].value_counts())
print("\nTESS Labels:\n", tess_data['label'].value_counts())
print("\nK2 Labels:\n", k2_data['label'].value_counts())

# Common features (these are columns that exist in all 3 datasets, based on your column prints)
# We picked ones related to the planet (like orbit period, size, temperature) and star (like temperature, size)
# Note: Transit duration/depth aren't in K2 columns, so we skipped them to avoid errors
common_features = [
    'pl_orbper',   # Orbital Period (how long the planet takes to orbit the star, in days)
    'pl_rade',     # Planetary Radius (size of the planet, in Earth sizes)
    'pl_insol',    # Insolation Flux (how much heat/light the planet gets from its star)
    'pl_eqt',      # Equilibrium Temperature (estimated temperature of the planet)
    'st_teff',     # Stellar Effective Temperature (temperature of the star)
    'st_rad',      # Stellar Radius (size of the star, in Sun sizes)
    'st_logg'      # Stellar Surface Gravity (gravity on the star's surface; helps with star type)
]

# Rename columns to standardize (make names match across datasets)
kepler_data = kepler_data.rename(columns={
    'koi_period': 'pl_orbper',
    'koi_prad': 'pl_rade',
    'koi_insol': 'pl_insol',
    'koi_teq': 'pl_eqt',
    'koi_steff': 'st_teff',
    'koi_srad': 'st_rad',
    'koi_slogg': 'st_logg'
})

# TESS and K2 already have matching names, so no big changes needed
# If K2 or TESS has slight variants, add them here (e.g., if K2 has 'pl_eqterr' instead, but they match)
tess_data = tess_data.rename(columns={})  # Placeholder, no changes
k2_data = k2_data.rename(columns={})      # Placeholder, no changes

# Select only the common features + the label column for each dataset
kepler_data = kepler_data[common_features + ['label']]
tess_data = tess_data[common_features + ['label']]
k2_data = k2_data[common_features + ['label']]

# Merge all three datasets into one big table called all_data
all_data = pd.concat([kepler_data, tess_data, k2_data], ignore_index=True)

# Print the merged table's shape (rows x columns) and column names to check
print("\nMerged Data Shape:", all_data.shape)
print(all_data.columns.tolist())

# Handle missing values: Fill with median (middle value) for numerical columns
all_data = all_data.fillna(all_data.median(numeric_only=True))

# Separate features (X = input data for AI) and labels (y = what to predict)
X = all_data.drop('label', axis=1)
y = all_data['label']

# Normalize features (scale numbers to a standard range so AI treats them equally)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance classes with SMOTE (create extra samples for underrepresented labels like candidates/false positives)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Combine back into a DataFrame and save the prepared data to a new CSV (for use in later steps)
prepared_data = pd.DataFrame(X_balanced, columns=X.columns)
prepared_data['label'] = y_balanced
prepared_data.to_csv('prepared_exoplanet_data.csv', index=False)

# Print final prepared shape and balanced label counts
print("\nPrepared Data Shape:", prepared_data.shape)
print("Prepared Labels:\n", prepared_data['label'].value_counts())