import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # For saving the model
from tsfresh import extract_features
import numpy as np
from astropy.io import fits
from astropy.table import Table

if __name__ == '__main__':
    # Load the prepared data
    data = pd.read_csv('prepared_exoplanet_data.csv')

    # Limit to 1000 rows for quick testing (remove for full run)
    data = data.iloc[:1000]

    # Load the local sample light curve FITS file (outside loop)
    with fits.open('sample_light_curve.fits') as hdul:
        light_curve = Table(hdul[1].data)
        print("Available Columns in Light Curve:", light_curve.columns)  # Print once

    time_col = light_curve['TIME']  # Days
    flux = light_curve['LC_DETREND']  # Use this from your file; it's the detrended flux data
    flux = np.nan_to_num(flux, nan=np.nanmean(flux))  # Replace NaNs with mean

    # For demo, repeat sample flux for all rows (replace with real curves per example)
    data['id'] = range(len(data))
    data['flux'] = [list(flux[:100]) for _ in range(len(data))]  # Take first 100 points

    # Create long format for tsfresh
    long_df = pd.DataFrame()
    for idx, row in data.iterrows():
        valid_flux = [f for f in row['flux'] if not np.isnan(f)]  # Remove NaNs if any
        temp_df = pd.DataFrame({
            'id': [row['id']] * len(valid_flux),
            'time': range(len(valid_flux)),
            'flux': valid_flux
        })
        long_df = pd.concat([long_df, temp_df])

    # Extract features
    extracted_features = extract_features(long_df, column_id='id', column_sort='time', column_value='flux')

    # Fill NaNs in extracted features (fix for KNN)
    extracted_features = extracted_features.fillna(0)

    # Separate features (X) and labels (y)
    X = data.drop(['label', 'id', 'flux'], axis=1, errors='ignore')  # Drop non-features
    y = data['label']

    # Add extracted features to X
    X = pd.concat([X.reset_index(drop=True), extracted_features.reset_index(drop=True)], axis=1)

    print("X with Curve Features Shape:", X.shape)

    # Split into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print shapes to check
    print("Training Data Shape:", X_train.shape)
    print("Test Data Shape:", X_test.shape)

    # Define base models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 decision trees
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)  # Gradient boosting
    knn = KNeighborsClassifier(n_neighbors=5)  # 5 nearest neighbors

    # Stack them (meta-model uses RandomForest to combine)
    estimators = [('rf', rf), ('xgb', xgb_model), ('knn', knn)]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=50, random_state=42))

    # Train the model
    stacking_model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(stacking_model, 'exoplanet_model.pkl')
    print("Model trained and saved as 'exoplanet_model.pkl'")

    # Make predictions on test data
    y_pred = stacking_model.predict(X_test)

    # Print evaluation metrics
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))