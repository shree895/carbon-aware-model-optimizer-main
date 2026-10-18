import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(
    page_title="Carbon-Aware Model Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_json(filename):
    """Load a JSON result file without changing the source file."""
    path = RESULTS_DIR / filename

    if not path.exists():
        return None, f"File not found: {path.name}"

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file), None
    except (json.JSONDecodeError, OSError) as exc:
        return None, str(exc)


comparison_data, comparison_error = load_json("model_comparison.json")
selector_data, selector_error = load_json("model_selector_results.json")
environment_data, environment_error = load_json(
    "energy_carbon_estimates.json"
)

# Read the actual saved model comparison results.
MODEL_NAMES = {
    "baseline": "Baseline",
    "pruned": "Pruned",
    "quantized": "Quantized INT8",
}


def build_comparison_models(data):
    models = []

    if not isinstance(data, dict):
        return models

    for key, values in data.items():
        if not isinstance(values, dict):
            continue

        name = MODEL_NAMES.get(key.lower(), key)
        models.append({
            "Model": name,
            "Accuracy (%)": values.get("accuracy_percent"),
            "Checkpoint size (MB)": values.get(
                "size_MB", values.get("model_size_MB")
            ),
            "Sparsity (%)": values.get("sparsity_percent"),
        })

    return models


def get_candidates(data):
    """Use selector candidates, preserving their actual recorded values."""
    if isinstance(data, dict):
        candidates = data.get("all_candidates", [])
        if isinstance(candidates, list):
            return [item for item in candidates if isinstance(item, dict)]
    return []


def number_or_none(value):
    try:
        if value is None:
            return None
        result = float(value)
        return result if pd.notna(result) else None
    except (ValueError, TypeError):
        return None


def fmt(value, digits=4, suffix=""):
    value = number_or_none(value)
    if value is None:
        return "N/A"
    return f"{value:,.{digits}f}{suffix}"


# Sidebar
st.sidebar.title("🌱 Green AI")
st.sidebar.caption("Carbon-Aware Model Optimizer")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Model Comparison",
        "Environmental Impact",
        "Model Selector",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Experiment results are loaded from the project's results/ folder.")

st.title("🌱 Carbon-Aware Model Optimizer")
st.caption(
    "Compare machine-learning models across accuracy, storage, "
    "inference performance and estimated environmental impact."
)

if comparison_error:
    st.warning(f"Model comparison data: {comparison_error}")
if selector_error:
    st.warning(f"Model selector data: {selector_error}")
if environment_error:
    st.warning(f"Environmental data: {environment_error}")

comparison_models = build_comparison_models(comparison_data)
candidates = get_candidates(selector_data)

# Prefer selector candidates for environmental metrics because they contain
# the saved inference, energy, carbon and assumed-power fields.
if candidates:
    model_df = pd.DataFrame(candidates)
    model_df = model_df.rename(columns={
        "name": "Model",
        "accuracy_percent": "Accuracy (%)",
        "model_size_MB": "Checkpoint size (MB)",
        "inference_seconds": "Inference time (s)",
        "estimated_energy_Wh": "Estimated energy (Wh)",
        "estimated_CO2_g": "Estimated CO₂ (g)",
        "assumed_power_W": "Assumed power (W)",
    })
else:
    model_df = pd.DataFrame(comparison_models)

# Existing saved recommendation
recommended = {}
if isinstance(selector_data, dict):
    candidate = selector_data.get("recommended_model", {})
    if isinstance(candidate, dict):
        recommended = candidate

if page == "Overview":
    st.subheader("Project Overview")
    st.write(
        "This project explores how model optimization techniques can help "
        "balance predictive accuracy, model storage and environmental cost. "
        "It uses existing experiment outputs; opening this dashboard does "
        "not train or benchmark models again."
    )

    if recommended:
        st.success(
            f"Existing saved recommendation: **{recommended.get('name', 'N/A')}**"
        )
    else:
        st.info("No saved model recommendation is available.")

    if not model_df.empty:
        metrics = [
            ("Accuracy (%)", "Accuracy (%)", 2, "%"),
            ("Checkpoint size (MB)", "Checkpoint size (MB)", 3, " MB"),
            ("Estimated energy (Wh)", "Estimated energy (Wh)", 6, " Wh"),
            ("Estimated CO₂ (g)", "Estimated CO₂ (g)", 6, " g"),
        ]

        cols = st.columns(4)
        for col, (label, field, digits, suffix) in zip(cols, metrics):
            value = recommended.get({
                "Accuracy (%)": "accuracy_percent",
                "Checkpoint size (MB)": "model_size_MB",
                "Estimated energy (Wh)": "estimated_energy_Wh",
                "Estimated CO₂ (g)": "estimated_CO2_g",
            }[field])

            with col:
                st.metric(label, fmt(value, digits, suffix))

        st.subheader("Models at a glance")
        display_columns = [
            col for col in [
                "Model",
                "Accuracy (%)",
                "Checkpoint size (MB)",
                "Inference time (s)",
                "Estimated energy (Wh)",
                "Estimated CO₂ (g)",
            ]
            if col in model_df.columns
        ]
        st.dataframe(
            model_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No usable model data was found in the JSON result files.")

    st.info(
        "Environmental values are estimates based on assumed power "
        "consumption, not direct power-meter measurements. Comparisons may "
        "also be affected by differences in evaluation hardware."
    )

elif page == "Model Comparison":
    st.subheader("Model Comparison")
    st.write("Compare recorded accuracy, checkpoint size and available performance metrics.")

    if model_df.empty:
        st.warning("No model comparison data is available.")
    else:
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        chart_options = [
            "Accuracy (%)",
            "Checkpoint size (MB)",
            "Inference time (s)",
            "Estimated energy (Wh)",
            "Estimated CO₂ (g)",
        ]
        available = [
            col for col in chart_options
            if col in model_df.columns
            and model_df[col].map(number_or_none).notna().any()
        ]

        if available:
            selected_metric = st.selectbox(
                "Choose a metric to compare",
                available,
            )
            chart_df = model_df[["Model", selected_metric]].copy()
            chart_df[selected_metric] = pd.to_numeric(
                chart_df[selected_metric], errors="coerce"
            )
            chart_df = chart_df.dropna(subset=[selected_metric])
            if not chart_df.empty:
                st.bar_chart(chart_df.set_index("Model"))

        st.caption(
            "Pruning sparsity is not the same as a smaller saved checkpoint. "
            "The recorded pruned checkpoint remains 2.371 MB."
        )

elif page == "Environmental Impact":
    st.subheader("Environmental Impact")
    st.write(
        "Energy and carbon estimates can help compare candidate models, "
        "but their reliability depends on the assumptions used during evaluation."
    )

    if not model_df.empty:
        environmental_columns = [
            col for col in [
                "Model",
                "Inference time (s)",
                "Estimated energy (Wh)",
                "Estimated CO₂ (g)",
                "Assumed power (W)",
            ]
            if col in model_df.columns
        ]

        if len(environmental_columns) > 1:
            st.dataframe(
                model_df[environmental_columns],
                use_container_width=True,
                hide_index=True,
            )

        for metric in ["Estimated energy (Wh)", "Estimated CO₂ (g)"]:
            if metric in model_df.columns:
                chart_df = model_df[["Model", metric]].copy()
                chart_df[metric] = pd.to_numeric(
                    chart_df[metric], errors="coerce"
                )
                chart_df = chart_df.dropna(subset=[metric])
                if not chart_df.empty:
                    st.markdown(f"**{metric}**")
                    st.bar_chart(chart_df.set_index("Model"))

    assumptions = None
    if isinstance(environment_data, dict):
        assumptions = environment_data.get("environmental_measurements")

    st.warning(
        "These environmental metrics are estimates, not direct measurements "
        "from a power meter. Assumed power and grid carbon intensity can "
        "affect the results."
    )
    if assumptions:
        st.caption(f"Source note: {assumptions}")

elif page == "Model Selector":
    st.subheader("Constraint-Based Model Selector")
    st.write(
        "Adjust the constraints to filter the existing candidates. "
        "This is an interactive calculation over saved results; it does "
        "not retrain or re-benchmark any model."
    )

    if not candidates:
        st.warning("No candidate models were found in model_selector_results.json.")
    else:
        defaults = (
            selector_data.get("constraints", {})
            if isinstance(selector_data, dict)
            else {}
        )

        accuracy_values = [
            number_or_none(item.get("accuracy_percent"))
            for item in candidates
        ]
        accuracy_values = [v for v in accuracy_values if v is not None]
        max_accuracy = max(accuracy_values, default=100.0)

        energy_values = [
            number_or_none(item.get("estimated_energy_Wh"))
            for item in candidates
        ]
        energy_values = [v for v in energy_values if v is not None]
        max_energy = max(energy_values, default=1.0)

        carbon_values = [
            number_or_none(item.get("estimated_CO2_g"))
            for item in candidates
        ]
        carbon_values = [v for v in carbon_values if v is not None]
        max_carbon = max(carbon_values, default=1.0)

        c1, c2, c3 = st.columns(3)
        with c1:
            min_accuracy = st.number_input(
                "Minimum accuracy (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("minimum_accuracy_percent", 70.0)),
                step=0.5,
            )
        with c2:
            energy_limit = st.number_input(
                "Maximum energy (Wh)",
                min_value=0.0,
                value=float(defaults.get("maximum_energy_Wh", 1.0)),
                step=0.01,
                format="%.6f",
            )
        with c3:
            carbon_limit = st.number_input(
                "Maximum CO₂ (g)",
                min_value=0.0,
                value=float(defaults.get("maximum_CO2_g", 1.0)),
                step=0.01,
                format="%.6f",
            )

        priority = st.selectbox(
            "Among eligible models, prioritize",
            [
                "Highest accuracy",
                "Lowest estimated energy",
                "Lowest estimated CO₂",
                "Smallest checkpoint",
            ],
        )

        rows = []
        for item in candidates:
            accuracy = number_or_none(item.get("accuracy_percent"))
            energy = number_or_none(item.get("estimated_energy_Wh"))
            carbon = number_or_none(item.get("estimated_CO2_g"))
            size = number_or_none(item.get("model_size_MB"))

            checks = {
                "Accuracy passes": (
                    accuracy is not None and accuracy >= min_accuracy
                ),
                "Energy passes": (
                    energy is not None and energy <= energy_limit
                ),
                "CO₂ passes": (
                    carbon is not None and carbon <= carbon_limit
                ),
            }

            rows.append({
                "Model": item.get("name", "Unknown"),
                "Accuracy (%)": accuracy,
                "Checkpoint size (MB)": size,
                "Estimated energy (Wh)": energy,
                "Estimated CO₂ (g)": carbon,
                "Accuracy passes": checks["Accuracy passes"],
                "Energy passes": checks["Energy passes"],
                "CO₂ passes": checks["CO₂ passes"],
                "Eligible": all(checks.values()),
            })

        eligibility_df = pd.DataFrame(rows)
        st.markdown("**Candidate eligibility**")
        st.dataframe(
            eligibility_df,
            use_container_width=True,
            hide_index=True,
        )

        eligible = eligibility_df[eligibility_df["Eligible"]].copy()

        if eligible.empty:
            st.error("No candidate meets all three selected constraints.")
        else:
            if priority == "Highest accuracy":
                eligible = eligible.sort_values(
                    "Accuracy (%)", ascending=False
                )
            elif priority == "Lowest estimated energy":
                eligible = eligible.sort_values(
                    "Estimated energy (Wh)", ascending=True
                )
            elif priority == "Lowest estimated CO₂":
                eligible = eligible.sort_values(
                    "Estimated CO₂ (g)", ascending=True
                )
            else:
                eligible = eligible.sort_values(
                    "Checkpoint size (MB)", ascending=True
                )

            best = eligible.iloc[0]
            st.success(
                f"Best eligible candidate for the selected priority: "
                f"**{best['Model']}**"
            )
            st.caption(
                "This result is calculated from the saved candidate metrics "
                "and your selected constraints. It is separate from the "
                "original saved recommendation."
            )

        st.caption(
            "A missing metric cannot pass its corresponding constraint. "
            "Estimated energy and CO₂ depend on assumed power and other "
            "evaluation assumptions."
        )

st.sidebar.divider()
st.sidebar.caption("Carbon-Aware Model Optimizer · Experiment data only")
