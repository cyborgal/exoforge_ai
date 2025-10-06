# ExoForge AI: Exoplanet Hunter

A Streamlit application that classifies potential exoplanets, visualizes light-curve signals, and lets you generatively "imagine" your discoveries. The project combines NASA catalog data, tsfresh feature extraction, and a stacking ensemble model to deliver an educational, interactive mission experience.

## Features
- **Mission Tour**: guided onboarding with stage progress tracking (Gather Inputs ? Explore Signals ? Classify & Explain).
- **Manual & CSV Predictions**: enter planet/star parameters or upload a CSV to classify multiple candidates.
- **NASA Data Fetch**: pull the latest discoveries directly from the NASA Exoplanet Archive, explore them in 2D/3D, and run predictions.
- **Generative Planet Builder**: after any prediction, a custom Plotly visualization, story, and downloadable planet card are generated based on the planet’s characteristics.
- **Retraining Workflow**: optional retraining with community data captured during app usage.

## Local Setup
1. Create and activate a virtual environment (recommended):
   `ash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # macOS/Linux
   `
2. Install dependencies:
   `ash
   pip install -r requirements.txt
   `
3. Run the app:
   `ash
   streamlit run app.py
   `

## Project Structure
`
app.py                     # Main Streamlit application
requirements.txt           # Python dependencies
exoplanet_model.pkl        # Saved stacking model
prepared_exoplanet_data.csv# Balanced training data for SHAP/fallback
sample_light_curve.fits    # Sample light-curve used for feature extraction
preprocess_data.py         # Data preparation script (optional for provenance)
train_model.py             # Model training script (optional for provenance)
test_upload.csv            # Example CSV for the upload workflow
`

## Deployment (Streamlit Community Cloud)
1. Push the files above to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Click **Create app**, select your repository/branch, and set the main file to pp.py.
4. Click **Deploy**. The cloud build installs dependencies from equirements.txt; the app will run exactly like your local version.

### Secrets
If you need API keys or private configuration, add them via Streamlit Cloud’s “Secrets” manager rather than committing them to GitHub.

## Data Notes
- prepared_exoplanet_data.csv was generated via preprocess_data.py and includes SMOTE-balanced features.
- sample_light_curve.fits is required for tsfresh feature extraction during retraining/NASA prediction flows.
- Large raw catalogs (cumulative.csv, k2pandc.csv, TOI.csv) are used only during preprocessing/training and aren’t required at runtime.

## License
Add your preferred license information here (e.g., MIT License).

## Acknowledgements
- NASA Exoplanet Archive for publicly available catalog data.
- Streamlit Community Cloud for easy deployment.
- tsfresh, shap, and numerous open-source libraries powering feature extraction and explainability.
