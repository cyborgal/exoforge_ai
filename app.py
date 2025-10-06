import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap  # For explainability
import numpy as np  # For simulations
from tsfresh import extract_features  # For light curve features
import colorsys
import random
from pathlib import Path
from datetime import datetime, timezone
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from astropy.io import fits
from astropy.table import Table

# File paths
MODEL_PATH = Path("exoplanet_model.pkl")
PREPARED_DATA_PATH = Path("prepared_exoplanet_data.csv")
USER_DATA_PATH = Path("user_contributions.csv")
SAMPLE_LIGHT_CURVE_PATH = Path("sample_light_curve.fits")
MAX_BASE_ROWS = 1000  # Mirror the training script default

# NASA API configuration
NASA_API_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
NASA_API_QUERY_TEMPLATE = (
    "select top {limit} pl_name, pl_orbper, pl_rade, pl_insol, pl_eqt, st_teff, st_rad, st_logg, disc_year "
    "from pscomppars order by disc_year desc"
)
NASA_DEFAULT_LIMIT = 50

# Hyperparameter defaults and presets
DEFAULT_HYPERPARAMS = {
    "rf_n_estimators": 100,
    "rf_max_depth": 0,  # 0 -> automatically determined by sklearn
    "xgb_n_estimators": 100,
    "xgb_learning_rate": 0.1,
    "knn_n_neighbors": 5,
    "stack_rf_n_estimators": 50,
}

HYPERPARAMETER_PRESETS = {
    "Default (balanced)": DEFAULT_HYPERPARAMS,
    "Fast prototyping": {
        "rf_n_estimators": 50,
        "rf_max_depth": 0,
        "xgb_n_estimators": 50,
        "xgb_learning_rate": 0.2,
        "knn_n_neighbors": 3,
        "stack_rf_n_estimators": 25,
    },
    "Accuracy focus": {
        "rf_n_estimators": 200,
        "rf_max_depth": 12,
        "xgb_n_estimators": 150,
        "xgb_learning_rate": 0.05,
        "knn_n_neighbors": 7,
        "stack_rf_n_estimators": 100,
    },


}
MISSION_STAGES = [
    {"id": 0, "title": "Stage 1 - Gather Inputs", "description": "Set mission parameters from NASA catalogs or manual readings."},
    {"id": 1, "title": "Stage 2 - Explore Signals", "description": "Inspect light curves and visualize how planets dim their stars."},
    {"id": 2, "title": "Stage 3 - Classify & Explain", "description": "Run the model and interpret the discovery."},
]

GUIDED_TOUR_STEPS = [
    {
        "title": "Welcome to your exoplanet mission!",
        "body": "We will gather planet and star measurements, explore how transits look, and let ExoForge AI label potential worlds.",
    },
    {
        "title": "Stage 1 - Gather Inputs",
        "body": "Orbital period tells us how fast a planet circles its star. Radius and insolation hint at size and climate. Plug in values or load NASA data to begin.",
    },
    {
        "title": "Stage 2 - Explore Signals",
        "body": "Watch how a star's light dips during a transit. Adjust the sliders to see how depth, duration, and impact parameter change the curve.",
    },
    {
        "title": "Stage 3 - Classify & Explain",
        "body": "Run a prediction, review the confidence gauge, and learn why the model chose the class via SHAP feature explanations.",
    },
]


def hsl_to_hex(h: float, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return '#{0:02x}{1:02x}{2:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def classify_temperature(eq_temp: float | None) -> str:
    if eq_temp is None or np.isnan(eq_temp):
        return 'temperature unknown'
    if eq_temp < 230:
        return 'a frigid, icy world'
    if eq_temp < 320:
        return 'temperate with potential oceans'
    if eq_temp < 500:
        return 'a warm greenhouse world'
    return 'an ultra-hot planet with molten skies'


def size_descriptor(radius: float | None) -> str:
    if radius is None or np.isnan(radius):
        return 'mystery-sized'
    if radius < 0.8:
        return 'compact and likely rocky'
    if radius < 1.5:
        return 'Earth-like in scale'
    if radius < 4:
        return 'a super-Earth or mini-Neptune'
    return 'a majestic gas giant'


def radiation_descriptor(insolation: float | None) -> str:
    if insolation is None or np.isnan(insolation):
        return 'mysterious energy levels'
    if insolation < 0.5:
        return 'gentle starlight bathing frozen continents'
    if insolation < 1.5:
        return "balanced stellar warmth akin to Earth's"
    if insolation < 5:
        return 'a toasty world with intense sunshine'
    return 'blistering stellar bombardment'


def find_similar_exoplanet(features: dict[str, float | None]) -> str | None:
    nasa_df = st.session_state.get('nasa_data')
    if nasa_df is None or len(nasa_df) == 0:
        return None
    comparison_cols = [col for col in ('pl_orbper', 'pl_rade', 'st_teff') if col in nasa_df.columns]
    if not comparison_cols:
        return None
    df = nasa_df.dropna(subset=comparison_cols)
    if df.empty:
        return None
    target = np.array([features.get(col) for col in comparison_cols], dtype=float)
    if np.isnan(target).any():
        return None
    diff = df[comparison_cols].to_numpy(dtype=float) - target
    distances = np.sqrt((diff ** 2).sum(axis=1))
    best_idx = int(np.argmin(distances))
    best_row = df.iloc[best_idx]
    name = best_row.get('exoplanet_name')
    if pd.isna(name):
        return None
    return str(name)


def generate_planet_name(seed: int) -> str:
    rng = random.Random(seed)
    prefixes = ['Xeo', 'Astra', 'Nova', 'Kepla', 'Orb', 'Halo', 'Myth', 'Echo']
    suffixes = ['-Prime', '-LX', '-9', '-Tau', '-Rho', '-Zen', '-Aur']
    base = rng.choice(prefixes)
    tail = rng.choice(suffixes)
    number = rng.randint(100, 999)
    return f"{base}{number}{tail}"


def generate_planet_palette(seed: int, mood_shift: float = 0.0) -> tuple[str, str, str]:
    rng = random.Random(seed)
    base_h = (rng.random() + mood_shift) % 1.0
    s = 0.55 + 0.25 * rng.random()
    l = 0.45 + 0.2 * rng.random()
    primary = hsl_to_hex(base_h, s, l)
    secondary = hsl_to_hex((base_h + 0.08) % 1.0, min(0.95, s + 0.1), max(0.15, l - 0.2))
    highlight = hsl_to_hex((base_h + 0.18) % 1.0, min(1.0, s + 0.2), min(1.0, l + 0.2))
    return primary, secondary, highlight


def render_planet_creator(
    features: dict[str, float | None],
    prediction: int,
    confidence: float,
    source_label: str,
) -> None:
    st.subheader('Generative Planet Builder')

    safe_features = {
        key: (float(value) if value is not None else np.nan) for key, value in features.items()
    }
    seed_base = abs(hash(tuple(sorted(safe_features.items())))) % (2 ** 32)
    widget_prefix = f"{source_label}_{seed_base}"

    vibe_options = {
        'Aurora': 0.08,
        'Stormy': -0.12,
        'Crystal': 0.24,
        'Desert': -0.28,
    }

    with st.expander('Customize your world', expanded=False):
        mood = st.radio(
            'Atmospheric vibe',
            list(vibe_options.keys()),
            index=0,
            key=f'{widget_prefix}_mood',
        )
        add_rings = st.checkbox(
            'Add planetary rings',
            value=bool(safe_features.get('pl_rade', np.nan) and safe_features.get('pl_rade', np.nan) >= 3),
            key=f'{widget_prefix}_rings',
        )
        cloud_swirl = st.slider(
            'Cloud swirl intensity', 0.0, 1.0, 0.5, 0.1, key=f'{widget_prefix}_swirl'
        )

    primary_color, secondary_color, highlight_color = generate_planet_palette(
        seed_base, mood_shift=vibe_options[mood]
    )

    phi = np.linspace(0, 2 * np.pi, 400)
    base_radius = 1.0
    swirl = 1 + 0.05 * cloud_swirl * np.sin(3 * phi + (seed_base % 360) * np.pi / 180)
    x = base_radius * swirl * np.cos(phi)
    y = base_radius * swirl * np.sin(phi)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            fill='toself',
            line=dict(color=secondary_color, width=2),
            fillcolor=primary_color,
            hoverinfo='skip',
        )
    )

    if add_rings:
        ring_radius = 1.25
        ring_x = ring_radius * np.cos(phi)
        ring_y = 0.35 * ring_radius * np.sin(phi)
        fig.add_trace(
            go.Scatter(
                x=ring_x,
                y=ring_y,
                mode='lines',
                line=dict(color=highlight_color, width=1),
                fill='toself',
                fillcolor=highlight_color,
                opacity=0.25,
                hoverinfo='skip',
            )
        )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='#050915',
        paper_bgcolor='#050915',
        title=dict(text='Imagined Appearance', font=dict(color='white', size=16)),
    )

    st.plotly_chart(fig, use_container_width=True)

    orbit_days = safe_features.get('pl_orbper', np.nan)
    radius = safe_features.get('pl_rade', np.nan)
    insolation = safe_features.get('pl_insol', np.nan)
    eq_temp = safe_features.get('pl_eqt', np.nan)
    star_temp = safe_features.get('st_teff', np.nan)

    size_text = size_descriptor(radius)
    temp_text = classify_temperature(eq_temp)
    radiation_text = radiation_descriptor(insolation)
    star_text = describe_star_temperature(star_temp)

    similar = find_similar_exoplanet(safe_features)
    planet_name = generate_planet_name(seed_base)

    def format_metric(value: float | None, suffix: str, digits: int = 1) -> str:
        if value is None or np.isnan(value):
            return 'unknown'
        fmt = '{:.' + str(digits) + 'f}'
        return fmt.format(value) + suffix

    prediction_labels = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
    verdict = prediction_labels.get(int(prediction), 'Unknown')

    story_lines = [
        f"**{planet_name}** is {size_text} with {temp_text}.",
        f"It receives {radiation_text} and circles its star every {format_metric(orbit_days, ' days', 0)}.",
        f"Host star outlook: {star_text}.",
    ]
    if similar:
        story_lines.append(f"Closest NASA sibling: **{similar}**.")
    story_lines.append(
        f"Model verdict: class {int(prediction)} ({verdict}) with {confidence * 100:.0f}% confidence."
    )

    st.markdown('\n'.join(story_lines))

    planet_card = '\n'.join([
        f"Name: {planet_name}",
        f"Orbit: {format_metric(orbit_days, ' days', 0)}",
        f"Radius: {format_metric(radius, ' Earth radii')}",
        f"Insolation: {format_metric(insolation, ' \u00d7 Earth flux')}",
        f"Equilibrium temperature: {format_metric(eq_temp, ' K', 0)}",
        f"Stellar temperature: {format_metric(star_temp, ' K', 0)}",
        f"Verdict: {verdict} (confidence {confidence * 100:.0f}%)",
        f"Kindred world: {similar if similar else 'uncharted'}",
    ])

    st.download_button(
        'Download planet card',
        planet_card,
        file_name=f"{planet_name.lower().replace(' ', '_')}_card.txt",
        key=f'{widget_prefix}_card',
    )

    tweet_text = (
        f"Discovered {planet_name}: {size_text}, {temp_text}. Model sees class {int(prediction)} "
        f"with {confidence * 100:.0f}% confidence!"
    )
    st.text_area('Share your discovery', tweet_text, key=f'{widget_prefix}_share', height=80)


def ensure_onboarding_state() -> None:
    state = st.session_state
    state.setdefault("mission_stage", 0)
    state.setdefault("onboarding_step", 0)
    state.setdefault("onboarding_complete", False)
    state.setdefault("prediction_counter", 0)


def update_mission_stage(target_stage: int) -> None:
    max_stage = len(MISSION_STAGES) - 1
    current = st.session_state.get("mission_stage", 0)
    st.session_state["mission_stage"] = min(max(current, target_stage), max_stage)


def render_onboarding_tour() -> None:
    if st.session_state.get("onboarding_complete", False):
        return
    step = st.session_state.get("onboarding_step", 0)
    step = max(0, min(step, len(GUIDED_TOUR_STEPS) - 1))
    info = GUIDED_TOUR_STEPS[step]
    with st.container():
        st.info(f"**{info['title']}**\n\n{info['body']}")

        col_next, col_skip = st.columns([1, 1])
        with col_next:
            if st.button("Next tip", key=f"tour_next_{step}"):
                if step + 1 < len(GUIDED_TOUR_STEPS):
                    st.session_state["onboarding_step"] = step + 1
                else:
                    st.session_state["onboarding_complete"] = True
                st.rerun()
        with col_skip:
            if st.button("Skip tour", key=f"tour_skip_{step}"):
                st.session_state["onboarding_complete"] = True
                st.rerun()


def render_mission_banner() -> None:
    stage = st.session_state.get("mission_stage", 0)
    progress_value = (stage + 1) / len(MISSION_STAGES)
    st.progress(progress_value)
    st.caption(f"Mission progress: {MISSION_STAGES[stage]['title']}")
    cols = st.columns(len(MISSION_STAGES))
    for idx, stage_info in enumerate(MISSION_STAGES):
        container = cols[idx]
        title = stage_info["title"]
        description = stage_info["description"]
        if idx == stage:
            container.markdown("**{}**<br>:rocket: {}".format(title, description), unsafe_allow_html=True)
        elif idx < stage:
            container.markdown("{}<br>[completed] {}".format(title, description), unsafe_allow_html=True)
        else:
            container.markdown("{}<br>{}".format(title, description), unsafe_allow_html=True)



def describe_orbital_period(days: float) -> str:
    if days <= 0:
        return "Set a positive value to compare with known exoplanets."
    if days < 10:
        return "Ultra-short orbit, similar to hot Jupiters hugging their stars."
    if days < 100:
        return "Comparable to many Kepler finds with close orbits."
    if days < 400:
        return "Similar to Earth-like years around Sun-sized stars."
    return "A long, wide orbit that may indicate a colder world."


def describe_planet_radius(radius: float) -> str:
    if radius <= 0:
        return "Enter a radius to estimate planet type."
    if radius < 1:
        return "Smaller than Earth and likely rocky."
    if radius < 3:
        return "Super-Earth or mini-Neptune size range."
    return "Gas giant territory similar to Jupiter or Saturn."


def describe_star_temperature(teff: float) -> str:
    if teff <= 0:
        return "Add a stellar temperature to gauge host star type."
    if teff < 4000:
        return "Cool K or M dwarf star with gentle light."
    if teff < 6500:
        return "Sun-like star delivering balanced radiation."
    return "Hot F-type star that can scorch nearby planets."




def get_model_features(fitted_model):
    if hasattr(fitted_model, "feature_names_in_"):
        return list(fitted_model.feature_names_in_)
    if hasattr(fitted_model.estimators_[0], "feature_names_in_"):
        return list(fitted_model.estimators_[0].feature_names_in_)
    return []


def normalize_max_depth(value: int | float) -> int | None:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return None if numeric <= 0 else numeric


def ensure_hyperparam_state() -> None:
    if "hp_values" not in st.session_state:
        st.session_state["hp_values"] = dict(DEFAULT_HYPERPARAMS)
    if "hp_preset_choice" not in st.session_state:
        st.session_state["hp_preset_choice"] = "Default (balanced)"
    if "hp_preset_applied" not in st.session_state:
        st.session_state["hp_preset_applied"] = False
    for key, value in st.session_state["hp_values"].items():
        state_key = f"hp_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = value


@st.cache_data(show_spinner=False)
def load_base_feature_matrix(feature_names):
    feature_names = list(feature_names)
    base_df = pd.read_csv(PREPARED_DATA_PATH)
    if MAX_BASE_ROWS:
        base_df = base_df.iloc[:MAX_BASE_ROWS]
    base_df = base_df.reset_index(drop=True)

    extracted = pd.DataFrame(index=base_df.index)
    if SAMPLE_LIGHT_CURVE_PATH.exists():
        with fits.open(SAMPLE_LIGHT_CURVE_PATH) as hdul:
            light_curve = Table(hdul[1].data)
        flux = light_curve["LC_DETREND"] if "LC_DETREND" in light_curve.colnames else light_curve["PDCSAP_FLUX"]
        flux = np.nan_to_num(np.array(flux, dtype=float), nan=float(np.nanmean(flux)))
        flux = flux[: min(len(flux), 100)]  # Keep it lightweight

        long_records = []
        for idx in base_df.index:
            long_records.append(
                pd.DataFrame(
                    {
                        "id": idx,
                        "time": range(len(flux)),
                        "flux": flux,
                    }
                )
            )
        long_df = pd.concat(long_records, ignore_index=True)
        extracted = extract_features(long_df, column_id="id", column_sort="time", column_value="flux")
        extracted = extracted.sort_index().reset_index(drop=True).fillna(0)
    else:
        extracted = pd.DataFrame(index=base_df.index)

    features = base_df.drop(columns=["label"], errors="ignore").reset_index(drop=True)
    features = pd.concat([features, extracted], axis=1)

    if feature_names:
        missing = [col for col in feature_names if col not in features.columns]
        if missing:
            zero_pad = pd.DataFrame(0.0, index=features.index, columns=missing)
            features = pd.concat([features, zero_pad], axis=1)
        features = features[feature_names]
    return features, base_df["label"].reset_index(drop=True)


def prepare_features(base_df, extracted_df, feature_names):
    extracted_df = extracted_df if extracted_df is not None else pd.DataFrame(index=base_df.index)
    extracted_df = extracted_df.reindex(base_df.index, fill_value=0)
    combined = pd.concat([base_df, extracted_df], axis=1)
    missing = [col for col in feature_names if col not in combined.columns]
    if missing:
        zero_pad = pd.DataFrame(0.0, index=combined.index, columns=missing)
        combined = pd.concat([combined, zero_pad], axis=1)
    return combined[feature_names]


def append_user_contribution(features_df, labels, source, predictions, confidences):
    records = features_df.copy()
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels, index=records.index)
    if not isinstance(predictions, pd.Series):
        predictions = pd.Series(predictions, index=records.index)
    if not isinstance(confidences, pd.Series):
        confidences = pd.Series(confidences, index=records.index)

    records["label"] = labels
    records["model_prediction"] = predictions
    records["model_confidence"] = confidences
    records["source"] = source
    records["timestamp"] = datetime.now(timezone.utc).isoformat()

    if USER_DATA_PATH.exists():
        existing = pd.read_csv(USER_DATA_PATH)
        combined = pd.concat([existing, records], ignore_index=True)
    else:
        combined = records
    combined.to_csv(USER_DATA_PATH, index=False)


def load_user_contributions_for_training(feature_names):
    if not USER_DATA_PATH.exists():
        return pd.DataFrame(columns=feature_names), pd.Series(dtype=float)
    user_df = pd.read_csv(USER_DATA_PATH)
    if "label" not in user_df.columns:
        return pd.DataFrame(columns=feature_names), pd.Series(dtype=float)
    labeled = user_df.dropna(subset=["label"])
    if labeled.empty:
        return pd.DataFrame(columns=feature_names), pd.Series(dtype=float)

    labeled["label"] = labeled["label"].astype(float)
    features = labeled.reindex(columns=feature_names, fill_value=0.0)
    return features, labeled["label"].reset_index(drop=True)


def get_contribution_counts():
    if not USER_DATA_PATH.exists():
        return 0, 0
    data = pd.read_csv(USER_DATA_PATH)
    total = len(data)
    labeled = int(data["label"].notna().sum()) if "label" in data.columns else 0
    return total, labeled


def retrain_model_with_contributions(feature_names, hyperparams):
    params = dict(DEFAULT_HYPERPARAMS)
    if hyperparams:
        params.update({k: hyperparams.get(k, params[k]) for k in params})

    base_features, base_labels = load_base_feature_matrix(tuple(feature_names))
    user_features, user_labels = load_user_contributions_for_training(feature_names)

    X = base_features.copy()
    y = base_labels.copy()
    if not user_features.empty:
        X = pd.concat([X, user_features], ignore_index=True)
        y = pd.concat([y, user_labels], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_max_depth = normalize_max_depth(params.get("rf_max_depth", 0))

    rf = RandomForestClassifier(
        n_estimators=int(params.get("rf_n_estimators", 100)),
        max_depth=rf_max_depth,
        random_state=42,
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=int(params.get("xgb_n_estimators", 100)),
        learning_rate=float(params.get("xgb_learning_rate", 0.1)),
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    knn = KNeighborsClassifier(n_neighbors=int(params.get("knn_n_neighbors", 5)))

    estimators = [("rf", rf), ("xgb", xgb_model), ("knn", knn)]
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=RandomForestClassifier(
            n_estimators=int(params.get("stack_rf_n_estimators", 50)),
            random_state=42,
        ),
    )
    stacking_model.fit(X_train, y_train)

    y_pred = stacking_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    joblib.dump(stacking_model, MODEL_PATH)
    load_base_feature_matrix.clear()

    return {
        "model": stacking_model,
        "accuracy": accuracy,
        "report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": len(X),
        "user_samples": len(user_labels),
        "hyperparameters": params,
    }


@st.cache_data(show_spinner=False)
def fetch_nasa_exoplanet_data(limit=NASA_DEFAULT_LIMIT):
    limit = max(1, min(int(limit), 500))
    query = NASA_API_QUERY_TEMPLATE.format(limit=limit)
    params = {"query": query, "format": "json"}

    response = None
    last_exc = None
    for attempt in range(2):
        try:
            response = requests.get(NASA_API_URL, params=params, timeout=45)
            response.raise_for_status()
            break
        except requests.Timeout as exc:
            last_exc = exc
            if attempt == 0:
                continue
            raise requests.Timeout("NASA API request timed out. Try reducing the number of rows or retry later.") from exc
        except requests.ConnectionError as exc:
            last_exc = exc
            if attempt == 0:
                continue
            raise requests.ConnectionError("Could not reach NASA servers. Check your connection and retry.") from exc
    else:
        if last_exc is not None:
            raise last_exc

    data = response.json() if response is not None else []
    if not data:
        return pd.DataFrame(
            columns=[
                "exoplanet_name",
                "pl_orbper",
                "pl_rade",
                "pl_insol",
                "pl_eqt",
                "st_teff",
                "st_rad",
                "st_logg",
                "disc_year",
            ]
        )

    df = pd.DataFrame(data)
    expected_cols = ["pl_name", "pl_orbper", "pl_rade", "pl_insol", "pl_eqt", "st_teff", "st_rad", "st_logg", "disc_year"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = ["pl_orbper", "pl_rade", "pl_insol", "pl_eqt", "st_teff", "st_rad", "st_logg", "disc_year"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns={"pl_name": "exoplanet_name"})

    ordered_cols = ["exoplanet_name", "pl_orbper", "pl_rade", "pl_insol", "pl_eqt", "st_teff", "st_rad", "st_logg", "disc_year"]
    df = df[ordered_cols]
    return df


# Load the trained model
model = joblib.load(MODEL_PATH)
model_features = get_model_features(model)

try:
    sample_matrix, _ = load_base_feature_matrix(tuple(model_features))
    sample_data = sample_matrix.iloc[:100]
except Exception:  # pragma: no cover - fallback if feature build fails
    fallback = pd.read_csv(PREPARED_DATA_PATH).drop("label", axis=1)
    sample_data = fallback.iloc[:100]
    sample_matrix = fallback

if "retrain_summary" not in st.session_state:
    st.session_state["retrain_summary"] = None
if "nasa_data" not in st.session_state:
    st.session_state["nasa_data"] = None

ensure_onboarding_state()
ensure_hyperparam_state()
render_onboarding_tour()
render_mission_banner()

# Title and description
st.title("ExoForge AI: Exoplanet Hunter")
st.write("Upload data or enter values to classify exoplanets. Labels: 2=Confirmed, 1=Candidate, 0=False Positive.")

st.header("Join the Hunt for New Worlds!")
st.write(
    """
Inspired by NASA's Kepler, K2, and TESS missions, ExoForge AI automates exoplanet detection using machine learning.
The transit method detects dips in starlight as planets pass by - your inputs could uncover hidden worlds!
Tutorial: Enter planet/star data (e.g., from NASA archives) or upload a CSV. Get predictions and insights.
    """
)

# Sidebar for user input
st.sidebar.header("Input Options")
input_type = st.sidebar.radio("Choose input:", ("Manual Entry", "Upload CSV"))

st.sidebar.header("Community Contribution")
share_data = st.sidebar.checkbox(
    "I agree to share my inputs anonymously to improve the model",
    help="When enabled, your entries (with optional labels) are stored and can be used in future retraining runs.",
)
if share_data:
    st.sidebar.caption("Thank you! Only the scientific features are stored - no personal information.")
else:
    st.sidebar.caption("Sharing is optional. Leave unchecked if you prefer not to contribute data.")

total_contrib, labeled_contrib = get_contribution_counts()
st.sidebar.write(f"Community samples collected: {total_contrib} (labeled: {labeled_contrib})")

# Hyperparameter tuning controls
with st.sidebar.expander("Hyperparameter Tuning", expanded=False):
    st.write("Adjust model settings before retraining. Presets provide quick starting points.")
    preset_choice = st.selectbox(
        "Preset",
        list(HYPERPARAMETER_PRESETS.keys()),
        index=list(HYPERPARAMETER_PRESETS.keys()).index(st.session_state.get("hp_preset_choice", "Default (balanced)")),
        key="hp_preset_choice",
        help="Select a starting configuration. You can tweak individual values afterward.",
    )
    if st.session_state.get("hp_last_preset") != preset_choice:
        preset_values = HYPERPARAMETER_PRESETS[preset_choice]
        for key, value in preset_values.items():
            st.session_state[f"hp_{key}"] = value
        st.session_state["hp_values"] = dict(preset_values)
        st.session_state["hp_last_preset"] = preset_choice
    rf_trees = st.number_input(
        "Random Forest trees (base)",
        min_value=10,
        max_value=500,
        step=10,
        key="hp_rf_n_estimators",
    )
    rf_depth = st.number_input(
        "Random Forest max depth (0 = auto)",
        min_value=0,
        max_value=30,
        step=1,
        key="hp_rf_max_depth",
    )
    xgb_trees = st.number_input(
        "XGBoost trees",
        min_value=20,
        max_value=400,
        step=10,
        key="hp_xgb_n_estimators",
    )
    xgb_lr = st.number_input(
        "XGBoost learning rate",
        min_value=0.01,
        max_value=0.5,
        step=0.01,
        format="%0.2f",
        key="hp_xgb_learning_rate",
    )
    knn_neighbors = st.number_input(
        "KNN neighbors",
        min_value=1,
        max_value=15,
        step=1,
        key="hp_knn_n_neighbors",
    )
    stack_trees = st.number_input(
        "Stacking final Random Forest trees",
        min_value=10,
        max_value=500,
        step=10,
        key="hp_stack_rf_n_estimators",
    )

    st.caption(
        "Tip: Fewer trees and higher learning rates train faster but may reduce accuracy."
    )

current_hyperparams = {
    "rf_n_estimators": int(st.session_state["hp_rf_n_estimators"]),
    "rf_max_depth": int(st.session_state["hp_rf_max_depth"]),
    "xgb_n_estimators": int(st.session_state["hp_xgb_n_estimators"]),
    "xgb_learning_rate": float(st.session_state["hp_xgb_learning_rate"]),
    "knn_n_neighbors": int(st.session_state["hp_knn_n_neighbors"]),
    "stack_rf_n_estimators": int(st.session_state["hp_stack_rf_n_estimators"]),
}
st.session_state["hp_values"] = dict(current_hyperparams)

st.sidebar.header("Model Maintenance")
if st.sidebar.button("Refresh Model with Community Data"):
    with st.spinner("Retraining model with community contributions..."):
        retrain_info = retrain_model_with_contributions(model_features, current_hyperparams)
    model = retrain_info["model"]
    model_features = get_model_features(model)
    sample_matrix, _ = load_base_feature_matrix(tuple(model_features))
    sample_data = sample_matrix.iloc[:100]
    fetch_nasa_exoplanet_data.clear()
    st.session_state["retrain_summary"] = retrain_info
    hp_summary = retrain_info["hyperparameters"]
    rf_depth_text = "auto" if normalize_max_depth(hp_summary["rf_max_depth"]) is None else int(hp_summary["rf_max_depth"])
    st.success(
        "Model retrained on {total} samples (including {user} community entries). Test accuracy: {acc:.2f}."
        .format(total=retrain_info["total_samples"], user=retrain_info["user_samples"], acc=retrain_info["accuracy"])
    )
    st.caption(
        "Settings used: RF trees {rf}, RF depth {depth}, XGB trees {xgb}, XGB lr {lr:.2f}, KNN {knn}, Final RF trees {frf}."
        .format(
            rf=hp_summary["rf_n_estimators"],
            depth=rf_depth_text,
            xgb=hp_summary["xgb_n_estimators"],
            lr=hp_summary["xgb_learning_rate"],
            knn=hp_summary["knn_n_neighbors"],
            frf=hp_summary["stack_rf_n_estimators"],
        )
    )
    st.text(retrain_info["report"])
elif st.session_state["retrain_summary"]:
    info = st.session_state["retrain_summary"]
    st.sidebar.info(
        "Last refresh accuracy {acc:.2f} using {user} contributed samples."
        .format(acc=info["accuracy"], user=info["user_samples"])
    )

label_map = {
    "Unknown / Not provided": np.nan,
    "Confirmed (2)": 2,
    "Candidate (1)": 1,
    "False Positive (0)": 0,
}

if input_type == "Manual Entry":
    # Form for manual input (match our features)
    pl_orbper = st.number_input("Orbital Period (days)", value=0.0, help="Time for one orbit around the star. Earth takes 365 days; hot Jupiters can orbit in just a few days.")
    pl_rade = st.number_input("Planetary Radius (Earth radii)", value=0.0, help="Compare the planet size to Earth. Values <1 suggest rocky worlds; >3 points to gas giants.")
    pl_insol = st.number_input("Insolation Flux (Earth flux)", value=0.0, help="How much stellar energy hits the planet compared to Earth. Higher numbers mean hotter climates.")
    pl_eqt = st.number_input("Equilibrium Temperature (K)", value=0.0, help="Estimated planet temperature assuming no atmosphere. 300K is roughly Earth-like.")
    st_teff = st.number_input("Stellar Temperature (K)", value=0.0, help="Surface temperature of the host star. The Sun is about 5778 K.")
    st_rad = st.number_input("Stellar Radius (Solar radii)", value=0.0, help="Size of the star compared to the Sun. Values <1 indicate smaller stars like K/M dwarfs.")
    st_logg = st.number_input("Stellar log(g) (cm/s^2)", value=0.0, help="Surface gravity of the star. Sun-like stars sit near log(g)=4.3.")

    user_label_value = np.nan
    if share_data:
        label_choice = st.selectbox("Known classification (optional)", list(label_map.keys()), key="manual_label_choice")
        user_label_value = label_map[label_choice]

    flux_str = st.text_input(
        "Light Curve Flux (comma-separated values, e.g., 1.0,0.99,1.0 for a dip)",
        help="Optional. If provided, time-series features are generated with tsfresh.",
    )

    st.caption(
        f"Mission hint: {describe_orbital_period(pl_orbper)} | Planet size: {describe_planet_radius(pl_rade)} | Star type: {describe_star_temperature(st_teff)}"
    )

    with st.expander("Feature cheat sheet", expanded=False):
        st.markdown("- **Orbital Period** - time for one full orbit.")
        st.markdown("- **Planet Radius** - compare to Earth (1.0).")
        st.markdown("- **Insolation Flux** - higher means hotter worlds.")
        st.markdown("- **Equilibrium Temperature** - no-atmosphere estimate.")
        st.markdown("- **Stellar Temperature & Radius** - describe the host star.")
        st.markdown("- **log(g)** - surface gravity; values around 4 indicate Sun-like stars.")

    with st.expander("Need an example?", expanded=False):
        st.markdown("Try orbital period 280, radius 1.5, and stellar temperature 5700 K to mimic a hot Jupiter from NASA catalogs.")
        st.markdown("Want a rocky world? Set period around 20 days and radius 0.9.")

    manual_result = st.session_state.get("manual_result")

    if st.button("Predict", key="manual_predict"):
        feature_columns = [
            "pl_orbper",
            "pl_rade",
            "pl_insol",
            "pl_eqt",
            "st_teff",
            "st_rad",
            "st_logg",
        ]
        feature_values = [
            pl_orbper,
            pl_rade,
            pl_insol,
            pl_eqt,
            st_teff,
            st_rad,
            st_logg,
        ]
        input_df = pd.DataFrame({col: [value] for col, value in zip(feature_columns, feature_values)})

        extracted = pd.DataFrame()
        if flux_str:
            try:
                flux_list = [float(f) for f in flux_str.split(",") if f.strip()]
                if flux_list:
                    lc_df = pd.DataFrame(
                        {
                            "id": [0] * len(flux_list),
                            "time": range(len(flux_list)),
                            "flux": flux_list,
                        }
                    )
                    extracted = extract_features(lc_df, column_id="id", column_sort="time", column_value="flux").fillna(0)
                    extracted.index = input_df.index
            except ValueError:
                st.error("Could not parse the flux values. Please ensure they are numeric and comma-separated.")
                extracted = pd.DataFrame()

        prepared = prepare_features(input_df, extracted, model_features)

        prediction = model.predict(prepared)[0]
        prob = model.predict_proba(prepared)[0]
        confidence = float(np.max(prob))

        update_mission_stage(2)

        result_id = st.session_state.get("manual_result_id", 0) + 1
        st.session_state["manual_result_id"] = result_id

        feature_snapshot = {col: feature_values[idx] for idx, col in enumerate(feature_columns)}
        manual_result = {
            "result_id": result_id,
            "features": feature_snapshot,
            "feature_columns": feature_columns,
            "feature_values": feature_values,
            "prediction": int(prediction),
            "confidence": confidence,
            "message": f"Prediction: {prediction} (Confidence: {confidence:.2f})",
            "share_info": None,
        }

        if share_data:
            append_user_contribution(
                prepared,
                labels=pd.Series([user_label_value], index=prepared.index),
                source="manual",
                predictions=pd.Series([prediction], index=prepared.index),
                confidences=pd.Series([confidence], index=prepared.index),
            )
            if np.isnan(user_label_value):
                manual_result["share_info"] = "Your input was saved for future review. Add a label next time to help supervised retraining."
            else:
                manual_result["share_info"] = "Thanks for sharing labeled data! It will be available for the next model refresh."

        st.session_state["manual_result"] = manual_result
        manual_result = manual_result

    if manual_result:
        st.success(manual_result["message"])

        confidence = manual_result["confidence"]
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                number={"suffix": "%"},
                title={"text": "Prediction Confidence"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#4b9cd3"},
                    "steps": [
                        {"range": [0, 50], "color": "#f8d7da"},
                        {"range": [50, 80], "color": "#fff3cd"},
                        {"range": [80, 100], "color": "#d4edda"},
                    ],
                },
            )
        )
        gauge_fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(gauge_fig, use_container_width=True)

        feature_columns = manual_result["feature_columns"]
        feature_values = manual_result["feature_values"]
        baseline_lookup = {}
        if 'sample_matrix' in globals() and sample_matrix is not None:
            for col in feature_columns:
                if col in sample_matrix.columns:
                    baseline_lookup[col] = float(sample_matrix[col].median())
        comparison_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "User Input": feature_values,
                "Training Median": [baseline_lookup.get(col, 0.0) for col in feature_columns],
            }
        )
        comparison_long = comparison_df.melt(id_vars="feature", var_name="source", value_name="value")
        comparison_fig = px.bar(
            comparison_long,
            x="feature",
            y="value",
            color="source",
            barmode="group",
            title="Your Inputs vs Training Median",
        )
        comparison_fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Value",
            legend_title="Source",
            xaxis_tickangle=-40,
        )
        st.plotly_chart(comparison_fig, use_container_width=True)

        render_planet_creator(
            manual_result["features"],
            manual_result["prediction"],
            manual_result["confidence"],
            source_label=f"manual_{manual_result['result_id']}",
        )

        sample_size = len(sample_matrix) if 'sample_matrix' in globals() and sample_matrix is not None else 1
        st.markdown(
            f"**Mission debrief:** Compared with {sample_size} reference systems, this profile most closely matched class {manual_result['prediction']}."
        )

        if manual_result.get("share_info"):
            st.info(manual_result["share_info"])

        quiz_key = f"quiz_manual_{manual_result['result_id']}"
        quiz_options = [
            "Choose your answer",
            "It measures dips in a star's light when a planet passes in front.",
            "It tracks star wobble caused by gravitational pulls.",
            "It listens for radio messages from intelligent life.",
        ]
        quiz_choice = st.selectbox(
            "Quick quiz: What does the transit method measure?",
            quiz_options,
            index=0,
            key=quiz_key,
        )
        if quiz_choice != quiz_options[0]:
            if quiz_choice.startswith("It measures dips"):
                st.success("Correct! The transit method watches for brightness dips caused by orbiting planets.")
            else:
                st.warning("Not quite. The transit method focuses on starlight dips - try again next prediction!")

        if st.button("Share on X", key="share_manual"):
            tweet_text = (
                f"I just classified an exoplanet with ExoForge AI! Prediction: {manual_result['prediction']} #NASA #Exoplanets #SpaceApps"
            )
            tweet_url = f"https://twitter.com/intent/tweet?text={tweet_text}"
            st.markdown(f"[Tweet your discovery!]({tweet_url})")

else:  # Upload CSVelse:  # Upload CSV
    uploaded_file = st.file_uploader(
        "Upload CSV (columns: pl_orbper, pl_rade, pl_insol, pl_eqt, st_teff, st_rad, st_logg)",
        type="csv",
    )

    default_upload_label = np.nan
    if share_data:
        upload_label_choice = st.selectbox(
            "Known label for uploaded rows (used when CSV lacks a label column)",
            list(label_map.keys()),
            key="upload_label_choice",
        )
        default_upload_label = label_map[upload_label_choice]

    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        feature_df = raw_df.drop(columns=["label"], errors="ignore")
        prepared = prepare_features(feature_df, extracted_df=None, feature_names=model_features)

        predictions = model.predict(prepared)
        probabilities = model.predict_proba(prepared)
        confidences = probabilities.max(axis=1)

        results_df = raw_df.copy()
        results_df["prediction"] = predictions
        results_df["confidence"] = confidences

        st.write("Predictions:")
        st.dataframe(results_df)
        st.download_button("Download Results", results_df.to_csv(index=False), "predictions.csv")

        if share_data:
            if "label" in raw_df.columns:
                labels = raw_df["label"].astype(float)
            else:
                labels = pd.Series([default_upload_label] * len(results_df))
            append_user_contribution(
                prepared,
                labels=labels,
                source="upload",
                predictions=pd.Series(predictions),
                confidences=pd.Series(confidences),
            )
            st.info(
                "Uploaded rows have been stored for the next retraining cycle. Rows without labels remain for future review."
            )

        if st.button("Share on X", key="share_upload"):
            tweet_text = (
                "Analyzed exoplanet data with ExoForge AI-found potential new planets! #NASA #Exoplanets #SpaceApps"
            )
            tweet_url = f"https://twitter.com/intent/tweet?text={tweet_text}"
            st.markdown(f"[Tweet your results!]({tweet_url})")

# Live NASA data section
st.header("Live NASA Data")
nasa_limit = st.number_input(
    "Number of recent planets to fetch",
    min_value=10,
    max_value=200,
    value=NASA_DEFAULT_LIMIT,
    step=10,
    key="nasa_limit",
)

if st.button("Fetch Latest NASA Exoplanet Data", key="fetch_nasa"):
    with st.spinner("Contacting NASA Exoplanet Archive..."):
        try:
            nasa_df = fetch_nasa_exoplanet_data(limit=nasa_limit)
            if nasa_df.empty:
                st.warning("NASA API returned no results. Try increasing the limit or retry later.")
            else:
                st.session_state["nasa_data"] = nasa_df
                st.success(f"Loaded {len(nasa_df)} records from the NASA Exoplanet Archive.")
        except requests.RequestException as exc:
            st.error(f"Could not fetch data from NASA: {exc}")

nasa_df = st.session_state.get("nasa_data")
if nasa_df is not None and not nasa_df.empty:
    st.caption("Newest entries appear first. Columns match the model's required features.")

    available_years = nasa_df["disc_year"].dropna()
    display_df = nasa_df.copy()
    if not available_years.empty:
        min_year = int(available_years.min())
        max_year = int(available_years.max())
        default_start = max(min_year, max_year - 10)
        year_range = st.slider(
            "Discovery year range",
            min_value=min_year,
            max_value=max_year,
            value=(default_start, max_year),
            step=1,
            key="nasa_year_range",
        )
        display_df = nasa_df[
            ((nasa_df["disc_year"] >= year_range[0]) & (nasa_df["disc_year"] <= year_range[1]))
            | nasa_df["disc_year"].isna()
        ]
    st.dataframe(display_df)
    with st.expander("Mission briefing: Reading NASA catalogs", expanded=False):
        st.markdown("The NASA Exoplanet Archive aggregates confirmed and candidate planets from missions like Kepler, K2, and TESS. Use the year slider to zoom in on recent discoveries.")
        st.markdown("Hover any point in the charts below to see each planet's name and discovery year.")


    if not display_df.empty:
        scatter_fig = px.scatter(
            display_df,
            x="pl_orbper",
            y="pl_rade",
            color="disc_year",
            hover_name="exoplanet_name",
            labels={
                "pl_orbper": "Orbital Period (days)",
                "pl_rade": "Planet Radius (Earth radii)",
                "disc_year": "Discovery Year",
            },
            title="Orbital Period vs Radius",
        )
        scatter_fig.update_layout(legend_title="Discovery Year")
        st.plotly_chart(scatter_fig, use_container_width=True)

        scatter3d = px.scatter_3d(
            display_df,
            x="pl_orbper",
            y="pl_rade",
            z="st_teff",
            color="disc_year",
            hover_name="exoplanet_name",
            title="3D View: Period, Radius, Stellar Temperature",
        )
        scatter3d.update_layout(
            scene=dict(
                xaxis_title="Orbital Period (days)",
                yaxis_title="Planet Radius (Earth radii)",
                zaxis_title="Stellar Temperature (K)",
            )
        )
        st.plotly_chart(scatter3d, use_container_width=True)

        options = list(range(len(display_df)))

        def format_planet(idx: int) -> str:
            row = display_df.iloc[idx]
            name = row.get("exoplanet_name", f"Row {idx + 1}")
            year = row.get("disc_year", np.nan)
            if pd.notna(year):
                return f"{name} (disc. {int(year)})"
            return str(name)

        selected_index = st.selectbox(
            "Choose a planet to analyze",
            options,
            format_func=format_planet,
            key="nasa_row_selector",
        )

        nasa_label_value = np.nan
        if share_data:
            nasa_label_choice = st.selectbox(
                "Known classification for this NASA entry (optional)",
                list(label_map.keys()),
                key="nasa_label_choice",
            )
            nasa_label_value = label_map[nasa_label_choice]

        nasa_result = st.session_state.get("nasa_result")

        if st.button("Predict Selected NASA Planet", key="predict_nasa"):
            row = display_df.iloc[[selected_index]]
            feature_columns = ["pl_orbper", "pl_rade", "pl_insol", "pl_eqt", "st_teff", "st_rad", "st_logg"]
            feature_row = row[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            feature_row = feature_row.reset_index(drop=True)

            prepared = prepare_features(feature_row, extracted_df=None, feature_names=model_features)

            prediction = model.predict(prepared)[0]
            prob = model.predict_proba(prepared)[0]
            confidence = float(np.max(prob))
            planet_name = row.iloc[0].get("exoplanet_name", "Unknown planet")

            update_mission_stage(2)

            result_id = st.session_state.get("nasa_result_id", 0) + 1
            st.session_state["nasa_result_id"] = result_id

            feature_snapshot = {col: float(feature_row.iloc[0][col]) for col in feature_columns}
            share_info = None

            if share_data:
                append_user_contribution(
                    prepared,
                    labels=pd.Series([nasa_label_value], index=prepared.index),
                    source="nasa",
                    predictions=pd.Series([prediction], index=prepared.index),
                    confidences=pd.Series([confidence], index=prepared.index),
                )
                share_info = (
                    "NASA entry saved for future review. Add a label next time to strengthen retraining."
                    if np.isnan(nasa_label_value)
                    else "NASA entry stored with your label for the next model refresh."
                )

            nasa_result = {
                "result_id": result_id,
                "planet_name": planet_name,
                "features": feature_snapshot,
                "feature_columns": feature_columns,
                "feature_values": feature_row.iloc[0][feature_columns].tolist(),
                "prediction": int(prediction),
                "confidence": confidence,
                "message": f"Prediction for {planet_name}: {prediction} (Confidence: {confidence:.2f})",
                "share_info": share_info,
            }
            st.session_state["nasa_result"] = nasa_result

        nasa_result = st.session_state.get("nasa_result")
        if nasa_result:
            st.success(nasa_result["message"])

            confidence = nasa_result["confidence"]
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={"suffix": "%"},
                    title={"text": "Prediction Confidence"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#4b9cd3"},
                        "steps": [
                            {"range": [0, 50], "color": "#f8d7da"},
                            {"range": [50, 80], "color": "#fff3cd"},
                            {"range": [80, 100], "color": "#d4edda"},
                        ],
                    },
                )
            )
            gauge_fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(gauge_fig, use_container_width=True)

            feature_columns = nasa_result["feature_columns"]
            feature_values = nasa_result["feature_values"]
            baseline_lookup = {}
            if 'sample_matrix' in globals() and sample_matrix is not None:
                for col in feature_columns:
                    if col in sample_matrix.columns:
                        baseline_lookup[col] = float(sample_matrix[col].median())
            comparison_df = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "User Input": feature_values,
                    "Training Median": [baseline_lookup.get(col, 0.0) for col in feature_columns],
                }
            )
            comparison_long = comparison_df.melt(id_vars="feature", var_name="source", value_name="value")
            comparison_fig = px.bar(
                comparison_long,
                x="feature",
                y="value",
                color="source",
                barmode="group",
                title="Your Inputs vs Training Median",
            )
            comparison_fig.update_layout(
                xaxis_title="Feature",
                yaxis_title="Value",
                legend_title="Source",
                xaxis_tickangle=-40,
            )
            st.plotly_chart(comparison_fig, use_container_width=True)

            render_planet_creator(
                nasa_result["features"],
                nasa_result["prediction"],
                nasa_result["confidence"],
                source_label=f"nasa_{nasa_result['result_id']}",
            )

            sample_size = len(sample_matrix) if 'sample_matrix' in globals() and sample_matrix is not None else 1
            st.markdown(
                f"**Mission insight:** {nasa_result['planet_name']} most resembled class {nasa_result['prediction']} after comparing against {sample_size} reference systems."
            )

            if nasa_result.get("share_info"):
                st.info(nasa_result["share_info"])

            if st.button("Share on X", key="share_nasa"):
                tweet_text = (
                    f"Analyzed {nasa_result['planet_name']} with ExoForge AI - prediction {nasa_result['prediction']} at {nasa_result['confidence'] * 100:.0f}% confidence! #NASA #Exoplanets"
                )
                tweet_url = f"https://twitter.com/intent/tweet?text={tweet_text}"
                st.markdown(f"[Tweet your results!]({tweet_url})")

    else:
        st.info("No planets fall within the selected filters.")

# Model stats section
st.header("Model Insights")
if st.button("Show Feature Importance"):
    base_rf = model.estimators_[0][1]  # The fitted RandomForestClassifier
    explainer = shap.TreeExplainer(base_rf)
    shap_values = explainer(sample_data)
    st.write("SHAP Summary (feature impact from RandomForest base):")
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_data, show=False)
    st.pyplot(fig, bbox_inches="tight", use_container_width=True)
    plt.close(fig)

if st.session_state["retrain_summary"]:
    st.write(
        "Model Accuracy from Training:"
        f" {st.session_state['retrain_summary']['accuracy']:.2f} (latest refresh)."
    )
else:
    st.write("Model Accuracy from Training: ~70% (see console for details)")

st.header("Visualize a Transit")
col1, col2, col3 = st.columns(3)
with col1:
    transit_depth = st.slider("Transit depth (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="transit_depth")
with col2:
    transit_duration = st.slider("Transit duration (hours)", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="transit_duration")
with col3:
    baseline_noise = st.slider("Noise level (ppm)", min_value=0.0, max_value=300.0, value=50.0, step=10.0, key="transit_noise")
impact_parameter = st.slider("Impact parameter", min_value=0.0, max_value=0.9, value=0.2, step=0.1, key="transit_impact")

with st.expander("Learn more: The transit method", expanded=False):
    st.markdown("When a planet crosses in front of its star, the measured light dips slightly. Missions like Kepler detected thousands of worlds using this method.")
    st.markdown("Depth tells us about planet size, duration hints at orbit speed, and the impact parameter shows how centered the transit is.")

time = np.linspace(0, 10, 400)
dip = transit_depth / 100.0
width = max(transit_duration / 10.0, 0.1)
flux = 1 - dip * np.exp(-((time - 5 - impact_parameter) ** 2) / width)
if baseline_noise > 0:
    noise = np.random.default_rng(42).normal(0, baseline_noise / 1_000_000, size=time.size)
    flux = np.clip(flux + noise, 0, None)

transit_fig = go.Figure()
transit_fig.add_trace(go.Scatter(x=time, y=flux, mode="lines", name="Flux"))
transit_fig.update_layout(
    title="Interactive Transit Simulation",
    xaxis_title="Time (hours)",
    yaxis_title="Normalized Flux",
    hovermode="x unified",
)
st.plotly_chart(transit_fig, use_container_width=True)
st.caption("Adjust the sliders to explore how transit depth, duration, and impact parameter shape the observed light curve.")
update_mission_stage(1)






