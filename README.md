# ExoForge AI: Exoplanet Hunter

A Streamlit application that classifies potential exoplanets, visualizes light-curve signals, and lets you generatively "imagine" your discoveries. The project combines NASA catalog data, tsfresh feature extraction, and a stacking ensemble model to deliver an educational, interactive mission experience.

## Features
- **Mission Tour**: guided onboarding with stage progress tracking (Gather Inputs ? Explore Signals ? Classify & Explain).
- **Manual & CSV Predictions**: enter planet/star parameters or upload a CSV to classify multiple candidates.
- **NASA Data Fetch**: pull the latest discoveries directly from the NASA Exoplanet Archive, explore them in 2D/3D, and run predictions.
- **Generative Planet Builder**: after any prediction, a custom Plotly visualization, story, and downloadable planet card are generated based on the planet’s characteristics.
- **Retraining Workflow**: optional retraining with community data captured during app usage.

