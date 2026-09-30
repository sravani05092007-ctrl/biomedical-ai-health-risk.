
"""
Biomedical AI Health Risk Awareness
Run using: streamlit run app/app.py
"""

import os
import sys
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. PROJECT PATHS AND IMPORTS
# --------------------------------------------------

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SRC = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data")
RESOURCES_FILE = os.path.join(DATA_DIR, "resources.csv")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from predict import load_artifacts, predict_risk


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Health Risk Awareness",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("🩺 AI-Assisted Heart Health Risk Awareness")

st.markdown(
    """
    ### Understand your health indicators. Explore healthcare resources.

    Early awareness of potential heart-health risks and access to
    reliable healthcare information can be challenging.

    This application provides a preliminary, model-generated risk
    estimate from supported health indicators and a directory of
    listed healthcare resources.
    """
)

st.warning(
    "Educational prototype only. This application does not provide "
    "a medical diagnosis or replace professional medical advice. "
    "Do not use its results alone to make healthcare decisions."
)

with st.expander("About this project", expanded=False):
    st.markdown(
        """
        **Problem addressed**

        People may need understandable information about health
        indicators and guidance on finding healthcare resources.

        **Proposed solution**

        This application combines a machine-learning model with
        a fuzzy rule-based risk assessment. It presents the model
        outputs and provides a searchable directory of listed
        healthcare services.

        **Target users**

        People seeking preliminary heart-health risk awareness
        and information about healthcare resources.

        **Limitations**

        Results depend on the model, its training data, and the
        values entered. Predictions may be inaccurate. The
        resource directory is not a live availability service.
        """
    )


# --------------------------------------------------
# 4. LOAD MODEL AND METADATA
# --------------------------------------------------

@st.cache_resource
def get_model_artifacts():
    return load_artifacts()


try:
    model, meta = get_model_artifacts()

    if not isinstance(meta, dict) or "selected" not in meta:
        st.error(
            "The model metadata is incomplete. Please check "
            "the model artifacts and prediction module."
        )
        st.stop()

except FileNotFoundError:
    st.error(
        "Model files could not be found. Please check that all "
        "required model artifacts are included in your GitHub "
        "repository and that src/predict.py uses the correct paths."
    )
    st.stop()

except Exception as exc:
    st.error(
        "The model could not be loaded. Check the model artifacts, "
        "dependencies, and application logs."
    )
    st.caption(f"Error details: {exc}")
    st.stop()


# --------------------------------------------------
# 5. HEALTH INPUT SPECIFICATIONS
# --------------------------------------------------

SPEC = {
    "age": (
        "Age (years)", 18, 100, 50, 1
    ),
    "sex": (
        "Sex (0 = female, 1 = male)", 0, 1, 1, 1
    ),
    "cp": (
        "Chest pain type (1-4)", 1, 4, 3, 1
    ),
    "trestbps": (
        "Resting blood pressure (mmHg)", 80, 220, 130, 1
    ),
    "chol": (
        "Cholesterol (mg/dL)", 100, 600, 220, 1
    ),
    "fbs": (
        "Fasting blood sugar above 120 (0/1)", 0, 1, 0, 1
    ),
    "restecg": (
        "Resting ECG category (0-2)", 0, 2, 0, 1
    ),
    "thalach": (
        "Maximum heart rate achieved", 60, 220, 150, 1
    ),
    "exang": (
        "Exercise-induced angina (0/1)", 0, 1, 0, 1
    ),
    "oldpeak": (
        "ST depression", 0.0, 7.0, 1.0, 0.1
    ),
    "slope": (
        "ST slope (1-3)", 1, 3, 2, 1
    ),
    "ca": (
        "Major vessels coloured (0-3)", 0, 3, 0, 1
    ),
    "thal": (
        "Thal category (3-7)", 3, 7, 3, 1
    ),
}

# Features required by the model plus the existing fuzzy module.
selected_features = list(meta["selected"])

needed = list(
    dict.fromkeys(
        selected_features
        + ["age", "trestbps", "chol", "thalach"]
    )
)

# Check metadata against available input definitions.
unsupported_features = [
    feature for feature in needed if feature not in SPEC
]

if unsupported_features:
    st.error(
        "The model requires input features that are not defined "
        "in the application: "
        + ", ".join(unsupported_features)
    )
    st.stop()


# --------------------------------------------------
# 6. NAVIGATION TABS
# --------------------------------------------------

tab1, tab2 = st.tabs(
    [
        "🩺 Heart Health Risk Assessment",
        "🏥 Healthcare Resource Directory",
    ]
)


# --------------------------------------------------
# 7. HEART HEALTH RISK ASSESSMENT
# --------------------------------------------------

with tab1:

    st.header("Assess your health indicators")

    st.write(
        "Enter the requested values to obtain a model-generated "
        "estimate. Use reliable measurements where available. "
        "Do not guess medical values if you do not know them."
    )

    if selected_features:
        st.caption(
            "Model-selected features: "
            + ", ".join(selected_features)
        )

    if meta.get("dataset"):
        st.caption(
            f"Dataset recorded in model metadata: {meta['dataset']}"
        )

    st.info(
        "The values below use the input ranges configured in this "
        "prototype. They are not a substitute for clinical assessment."
    )

    vals = {}

    cols = st.columns(3)

    for i, feature in enumerate(needed):

        label, lo, hi, default, step = SPEC[feature]

        with cols[i % 3]:

            vals[feature] = st.number_input(
                label,
                min_value=lo,
                max_value=hi,
                value=default,
                step=step,
                key=f"health_input_{feature}",
                help=f"Allowed input range: {lo} to {hi}"
            )

    st.divider()

    if st.button(
        "Assess Risk",
        type="primary",
        use_container_width=True
    ):

        try:
            result = predict_risk(model, meta, vals)

            required_result_keys = [
                "category", "final", "ml", "fuzzy", "fired"
            ]

            if not all(
                key in result for key in required_result_keys
            ):
                st.error(
                    "The prediction result is incomplete. "
                    "Please check src/predict.py."
                )
                st.stop()

            category = str(result["category"]).capitalize()

            if category not in ["Low", "Medium", "High"]:
                st.error(
                    "The model returned an unsupported risk category. "
                    "Please check the prediction logic."
                )
                st.stop()

            final_score = float(result["final"])
            ml_score = float(result["ml"])
            fuzzy_score = float(result["fuzzy"])

            if not all(
                pd.notna(value)
                for value in [final_score, ml_score, fuzzy_score]
            ):
                st.error(
                    "The model returned an invalid numeric result. "
                    "Please check the model and input values."
                )
                st.stop()

            if not all(
                0 <= value <= 100
                for value in [final_score, ml_score, fuzzy_score]
            ):
                st.error(
                    "A model score is outside the expected 0-100 "
                    "range. Please check src/predict.py."
                )
                st.stop()

            st.session_state["risk_result"] = {
                "category": category,
                "final": final_score,
                "ml": ml_score,
                "fuzzy": fuzzy_score,
                "fired": result["fired"],
            }

        except Exception as exc:
            st.error(
                "Risk assessment could not be completed. "
                "Please verify the inputs and model configuration."
            )
            st.caption(f"Error details: {exc}")

    # Display the latest successful result.
    if "risk_result" in st.session_state:

        result = st.session_state["risk_result"]

        category = result["category"]
        final_score = result["final"]
        ml_score = result["ml"]
        fuzzy_score = result["fuzzy"]

        icons = {
            "Low": "🟢",
            "Medium": "🟠",
            "High": "🔴",
        }

        st.divider()
        st.header("Your assessment result")

        st.subheader(
            f"{icons[category]} {category} Risk Category"
        )

        st.metric(
            "Combined risk score",
            f"{final_score:.0f}/100"
        )

        st.progress(
            max(0.0, min(100.0, final_score)) / 100.0
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "ML model output",
            f"{ml_score:.1f}%"
        )

        c2.metric(
            "Fuzzy risk score",
            f"{fuzzy_score:.1f}/100"
        )

        st.markdown("### Understanding your result")

        st.write(
            f"The application assigned a **{category.lower()} "
            "risk category** to the entered health indicators, "
            f"with a combined score of {final_score:.0f}/100."
        )

        st.write(
            "This category is based on the application's configured "
            "model and rules. It does not establish whether a person "
            "has a medical condition, and it cannot rule out disease."
        )

        with st.expander(
            "View the fuzzy rules triggered by this assessment"
        ):
            fired_rules = result["fired"]

            if fired_rules:
                for rule_label, strength in fired_rules:
                    st.write(
                        f"- Rule category: **{rule_label}** "
                        f"(strength: {strength})"
                    )
            else:
                st.write(
                    "The model did not return any triggered fuzzy rules."
                )

        if category in ["Medium", "High"]:
            st.warning(
                "Consider discussing your health concerns and "
                "measurements with a qualified healthcare professional. "
                "Do not delay medical care based on this score."
            )
        else:
            st.info(
                "A low model-generated score does not guarantee "
                "that a person is healthy. Seek medical advice if "
                "you have symptoms or concerns."
            )

        st.caption(
            "ML model output, fuzzy score, and combined score have "
            "different meanings. They should not be interpreted as "
            "interchangeable measures of clinical risk."
        )

        st.markdown(
            "To explore listed healthcare resources, open the "
            "**Healthcare Resource Directory** tab above."
        )


# --------------------------------------------------
# 8. HEALTHCARE RESOURCE DIRECTORY
# --------------------------------------------------

with tab2:

    st.header("Healthcare Resource Directory")

    st.write(
        "Explore the healthcare resources included in this project's "
        "directory. Filter the listings by service type and city."
    )

    st.info(
        "This is a curated resource list, not a live hospital, "
        "blood bank, or organ availability system. Contact the "
        "provider directly to confirm current services and availability."
    )

    if not os.path.isfile(RESOURCES_FILE):

        st.error(
            "The healthcare resource file was not found. "
            "Please ensure data/resources.csv exists in the repository."
        )

    else:

        try:
            df = pd.read_csv(RESOURCES_FILE)

            required_columns = [
                "name", "type", "city", "website", "notes"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                st.error(
                    "The resource file is missing required columns: "
                    + ", ".join(missing_columns)
                )

            elif df.empty:

                st.info(
                    "No healthcare resources are currently listed."
                )

            else:

                # Clean text columns for display and filtering.
                for column in required_columns:
                    df[column] = (
                        df[column]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                    )

                # Remove rows without a resource name.
                df = df[df["name"] != ""]

                if df.empty:

                    st.info(
                        "No valid healthcare resource entries were found."
                    )

                else:

                    c1, c2 = st.columns(2)

                    available_types = sorted(
                        value for value in df["type"].unique()
                        if value
                    )

                    available_cities = sorted(
                        value for value in df["city"].unique()
                        if value
                    )

                    with c1:
                        selected_types = st.multiselect(
                            "Filter by service type",
                            available_types,
                            default=available_types,
                        )

                    with c2:
                        selected_cities = st.multiselect(
                            "Filter by city",
                            available_cities,
                            default=available_cities,
                        )

                    filtered_df = df[
                        df["type"].isin(selected_types)
                        & df["city"].isin(selected_cities)
                    ]

                    st.divider()

                    st.metric(
                        "Matching listed resources",
                        len(filtered_df)
                    )

                    if filtered_df.empty:

                        st.info(
                            "No resources match your selected filters. "
                            "Try selecting another service type or city."
                        )

                    else:

                        for _, row in filtered_df.iterrows():

                            with st.container(border=True):

                                st.subheader(row["name"])

                                details = []

                                if row["type"]:
                                    details.append(
                                        row["type"].replace("_", " ").title()
                                    )

                                if row["city"]:
                                    details.append(row["city"])

                                if details:
                                    st.caption(" • ".join(details))

                                if row["notes"]:
                                    st.write(row["notes"])

                                website = row["website"]

                                if (
                                    website.startswith("https://")
                                    or website.startswith("http://")
                                ):
                                    st.markdown(
                                        f"[Visit listed website]({website})"
                                    )
                                elif website:
                                    st.write(
                                        "Website entry requires verification."
                                    )

                    st.caption(
                        "Resource details are based on the supplied CSV "
                        "file. Verify the website, contact details, "
                        "services, and availability with the provider."
                    )

        except Exception as exc:

            st.error(
                "The healthcare resource file could not be read. "
                "Please verify its format and contents."
            )

            st.caption(f"Error details: {exc}")


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Biomedical AI Health Risk Awareness | Educational prototype | "
    "Model outputs are not medical diagnoses."
)
