"""Run: python -m streamlit run app.py"""
import json
import hashlib

import pandas as pd
import streamlit as st

from wastesort.common import CLASSES, ROOT, read_json
from wastesort.models import Predictor, read_upload

st.set_page_config(page_title="WasteSort", page_icon="♻️", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1120px;padding-top:2.7rem;padding-bottom:3rem}
h1 {letter-spacing:-.05em;font-size:3.5rem!important}
h2 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:#edf2ec;padding:1rem;border-radius:12px}
footer {visibility:hidden}
</style>
""", unsafe_allow_html=True)

st.caption("WASTESORT  /  MATERIAL RECOGNITION")
st.title("One photo. A clearer starting point.")
st.write("Explore the material in a single waste item with a model trained on TrashNet.")


@st.cache_resource
def load_model():
    return Predictor()


scan, research, about = st.tabs(["Classify an item", "Research results", "About the project"])
with scan:
    left, right = st.columns([1.08, 1], gap="large")
    with left:
        st.subheader("Choose a photo")
        st.write("Use a clear photo of one item against a simple background.")
        uploaded = st.file_uploader("JPEG or PNG · up to 10 MB", type=["jpg", "jpeg", "png"])
        st.caption("Photos are processed in memory. This app does not save uploaded files.")
        image = None
        upload_key = None
        if uploaded is not None:
            try:
                payload = uploaded.getvalue()
                image = read_upload(payload)
                upload_key = hashlib.sha256(payload).hexdigest()
                st.image(image, caption="Your photo", use_container_width=True)
            except ValueError as exc:
                st.error(str(exc))
        if st.session_state.get("prediction_upload") != upload_key:
            st.session_state.pop("prediction_result", None)
            st.session_state["prediction_upload"] = upload_key
    with right:
        st.subheader("Material prediction")
        if not (ROOT / "models/deployment.json").exists():
            st.info("The trained model is not installed yet. Follow the setup steps in the README.")
        elif image is None:
            st.info("Upload a photo to see the predicted material.")
            st.write("Supported materials")
            st.markdown(" · ".join(c.title() for c in CLASSES))
        else:
            if st.button("Identify material", type="primary", use_container_width=True):
                try:
                    with st.spinner("Looking at your item…"):
                        st.session_state["prediction_result"] = load_model().predict(image)
                except (OSError, ValueError, RuntimeError) as exc:
                    st.session_state.pop("prediction_result", None)
                    st.error("The model could not process this image. Check the installation or try another photo.")
                    with st.expander("Troubleshooting details"):
                        st.code(str(exc))
            result = st.session_state.get("prediction_result")
            if result:
                st.markdown(f"## {result['label'].title()}")
                scores = result["scores"]
                if scores:
                    highest = scores[result["label"]]
                    st.metric("Model score", f"{highest:.1%}")
                    st.caption("An uncalibrated model output, not the probability that this answer is correct.")
                    if highest < 0.60:
                        st.warning("The model has a low score for its top answer. Try a clearer photo and check the item yourself.")
                    st.bar_chart(pd.DataFrame({"Material": list(scores), "Score": list(scores.values())}).set_index("Material"))
                st.caption(f"Model: {result['model']} · selected using validation macro F1")
                st.download_button("Download this result", json.dumps(result, indent=2),
                                   file_name="wastesort-result.json", mime="application/json")
        st.divider()
        st.markdown("**Before choosing a bin**")
        st.write("Campus disposal instructions are awaiting verification. Check the posted campus rules.")
        st.caption("Material appearance does not establish recyclability, cleanliness, or safety. "
                   "Mixed items, hazardous waste and unrelated objects are outside this model's scope. "
                   "It can still assign a waste label to an unrelated photo.")

with research:
    st.subheader("Measured results, with context")
    comparison = ROOT / "reports/comparison.csv"
    if comparison.exists():
        report = pd.read_csv(comparison)
        metadata = read_json(ROOT / "models/deployment.json")
        audit = read_json(ROOT / "reports/data_audit.json")
        a, b, c = st.columns(3)
        a.metric("Usable dataset images", audit["usable_images"])
        b.metric("Held-out test images", audit["split_counts"]["test"])
        c.metric("Selected model", metadata["model_id"])
        st.write("The deployment model was chosen using validation results before the test set was evaluated.")
        display = report[["model", "role", "validation_macro_f1", "test_accuracy",
                          "test_macro_f1", "trash_recall"]].copy()
        names = {"svm": "HOG + SVM", "random_forest": "HOG + Random Forest",
                 "cnn": "Small CNN", "mobilenet": "MobileNetV2", "cnn_weighted": "Small CNN (weighted)"}
        display["model"] = display.model.map(names)
        display["role"] = display.role.map({"primary": "Primary comparison", "ablation": "Weighting experiment"})
        display = display.rename(columns={"model": "Model", "role": "Purpose",
                    "validation_macro_f1": "Validation macro F1", "test_accuracy": "Test accuracy",
                    "test_macro_f1": "Test macro F1", "trash_recall": "Trash recall"})
        for col in ["Validation macro F1", "Test accuracy", "Test macro F1", "Trash recall"]:
            display[col] = display[col].map(lambda value: f"{value:.2%}")
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption("Macro F1 gives each material class equal weight. Trash recall measures how many actual trash images were recognized.")
        selected = st.selectbox("Inspect a model", report.model.tolist())
        plot = ROOT / f"reports/{selected}_confusion.png"
        if plot.exists():
            st.image(str(plot), width=620)
        st.download_button("Download comparison", comparison.read_bytes(), "comparison.csv", "text/csv")
        st.info("Initial benchmark study: one split and one training seed. Human image review and separate "
                "campus-photo validation remain pending. These numbers do not measure campus performance.")
    else:
        st.info("Results will appear here after training and final evaluation finish.")
with about:
    st.subheader("A focused research prototype")
    st.write("WasteSort compares HOG + SVM, HOG + Random Forest, a small CNN, and MobileNetV2 transfer learning "
             "on the six-class TrashNet dataset. A separate CNN experiment examines class weighting.")
    st.write("KNN and Decision Tree are introductory weekly exercises. Linear Regression uses a separate "
             "numerical learning dataset and is not used for waste-material prediction.")
    st.markdown("[Dataset and attribution](https://github.com/garythung/trashnet)")
    st.markdown("[Project repository](https://github.com/haerinfan/wastesort)")
    st.caption("AI assisted implementation and experiment execution. Group verification, interpretation, "
               "and contribution records remain part of the research.")
