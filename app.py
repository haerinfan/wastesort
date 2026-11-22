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


MODEL_NAMES = {"svm": "HOG + SVM", "random_forest": "HOG + Random Forest",
               "cnn": "Small CNN", "mobilenet": "MobileNetV2", "cnn_weighted": "Small CNN (weighted)"}
DEFAULT_MODEL = "models/deployment.json"
FINETUNED_MODEL = "models/primary_finetuned.json"
MODEL_CHOICES = {"MobileNetV2 · TrashNet (selected study model)": DEFAULT_MODEL,
                 "MobileNetV2 · TrashNet + primary set (synthetic)": FINETUNED_MODEL}


@st.cache_resource
def load_model(metadata=DEFAULT_MODEL):
    return Predictor() if metadata == DEFAULT_MODEL else Predictor(metadata)


def show_image(relative, caption=None):
    path = ROOT / relative
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def percent(frame, columns):
    for col in columns:
        frame[col] = frame[col].map(lambda value: "–" if pd.isna(value) else f"{value:.1%}")
    return frame


scan, research, datasets, about = st.tabs(["Classify an item", "Research results", "Datasets", "About the project"])
with scan:
    model_file = DEFAULT_MODEL
    if (ROOT / FINETUNED_MODEL).exists():
        choice = st.radio("Model", list(MODEL_CHOICES), horizontal=True,
                          help="The second model also saw AI-generated primary-set images during training.")
        model_file = MODEL_CHOICES[choice]
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
                # A new photo or a different model both invalidate the shown result.
                upload_key = hashlib.sha256(payload).hexdigest() + "|" + model_file
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
                        st.session_state["prediction_result"] = load_model(model_file).predict(image)
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
                ranked = sorted((scores or {}).items(), key=lambda kv: -kv[1])
                if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.15:
                    st.info(f"Close call: **{ranked[1][0].title()}** scored nearly as high "
                            f"({ranked[1][1]:.1%}). Check the item yourself.")
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
        display["model"] = display.model.map(MODEL_NAMES)
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
    shift = ROOT / "reports/primary_set/domain_shift.csv"
    if shift.exists():
        st.divider()
        st.subheader("Primary set (synthetic): does TrashNet training transfer?")
        st.write("The frozen TrashNet models were tested, unchanged, on AI-generated campus scenes.")
        table = pd.read_csv(shift)
        table["model"] = table.model.map(MODEL_NAMES)
        table = table[["model", "trashnet_test_macro_f1", "primary_macro_f1", "primary_trash_recall",
                       "macro_f1_change"]].rename(columns={
            "model": "Model", "trashnet_test_macro_f1": "TrashNet test macro F1",
            "primary_macro_f1": "Primary set macro F1", "primary_trash_recall": "Primary set trash recall",
            "macro_f1_change": "Change"})
        st.dataframe(percent(table, table.columns[1:]), hide_index=True, use_container_width=True)
        show_image("reports/primary_set/mobilenet_domain_shift_confusion.png")
        finetune = ROOT / "reports/primary_set/finetune_comparison.csv"
        if finetune.exists():
            st.markdown("**Adding primary-set training images (MobileNetV2)**")
            table = pd.read_csv(finetune)
            table["model"] = table.model.map({"mobilenet": "TrashNet only",
                                              "mobilenet_primary": "TrashNet + primary set"})
            table["test_set"] = table.test_set.map({"trashnet_test": "TrashNet test",
                                                    "primary_test": "Primary set test"})
            table = table.rename(columns={"model": "Trained on", "test_set": "Tested on", "images": "Images",
                                          "accuracy": "Accuracy", "macro_f1": "Macro F1",
                                          "trash_recall": "Trash recall"})
            st.dataframe(percent(table, ["Accuracy", "Macro F1", "Trash recall"]),
                         hide_index=True, use_container_width=True)
        st.caption("Synthetic images measure transfer to generated scenes, not accuracy on real campus waste.")
with datasets:
    st.subheader("Primary dataset and secondary data (TrashNet)")
    st.write("**Secondary data:** 2,527 TrashNet photographs by Gary Thung and Mindy Yang, single items on "
             "a white background. **Primary dataset:** 11 real photographs the group took (9 single-item photos "
             "are scored), plus 2,521 images the group generated with an AI image model from a fixed, recorded "
             "prompt plan of campus-like scenes.")
    real = ROOT / "reports/primary_set/real_photos.csv"
    if real.exists():
        table = pd.read_csv(real)
        table["model"] = table.model.map({**MODEL_NAMES, "mobilenet_primary": "MobileNetV2 (TrashNet + primary)"})
        st.markdown("**The group's real photos:** correct predictions per model")
        st.dataframe(table[["model", "correct", "photos"]].rename(
            columns={"model": "Model", "correct": "Correct", "photos": "Photos"}), hide_index=True,
            use_container_width=True)
    comparison_file = ROOT / "reports/primary_set/dataset_comparison.json"
    if comparison_file.exists():
        summary = read_json(comparison_file)
        sets = summary["datasets"]
        a, b, c = st.columns(3)
        a.metric("Secondary images", sets["secondary"]["images"])
        b.metric("Primary set images", sets["primary"]["images"])
        c.metric("Domain separability", f"{summary['domain_separability']['cv_accuracy_mean']:.0%}",
                 help="Cross-validated accuracy of telling the two datasets apart. 50% = indistinguishable.")
        means = pd.DataFrame(summary["appearance_means"]).rename(
            columns={"secondary": "Secondary: TrashNet", "primary": "Primary set (synthetic)"})
        means.index = [i.replace("_", " ").capitalize() for i in means.index]
        st.dataframe(means.style.format("{:.3f}"), use_container_width=True)
        st.caption("Mean per-image values from 0 to 1. White background fraction counts near-white pixels.")
        for relative, caption in [
                ("reports/primary_set/class_distribution_comparison.png", "Class balance"),
                ("reports/primary_set/image_statistics_comparison.png", "Image appearance"),
                ("reports/primary_set/embedding_pca.png", "Visual feature overlap"),
                ("reports/primary_set/sample_grid.png", "Random examples per class")]:
            show_image(relative, caption)
    else:
        st.info("Dataset comparison appears here after running notebooks/WasteSort_Primary_Colab.ipynb.")
    audit_file = ROOT / "reports/primary_set/audit.json"
    if audit_file.exists():
        audit = read_json(audit_file)
        st.markdown(f"**Quality control:** {audit['accepted_images']} of {audit['candidates_planned']} "
                    "generated candidates kept.")
        st.dataframe(pd.Series(audit["rejected_by_reason"], name="Candidates").rename_axis("Reason"),
                     use_container_width=True)
with about:
    st.subheader("A focused research prototype")
    st.write("WasteSort compares HOG + SVM, HOG + Random Forest, a small CNN, and MobileNetV2 transfer learning "
             "on the six-class TrashNet dataset. A separate CNN experiment examines class weighting.")
    st.write("KNN and Decision Tree are introductory weekly exercises. Linear Regression uses a separate "
             "numerical learning dataset and is not used for waste-material prediction.")
    st.write("The primary set (synthetic) is a group-generated set of AI images of campus-like scenes, "
             "approved by the instructor as primary data. It tests whether TrashNet training transfers "
             "beyond white-background photos. See the Datasets tab and docs/PRIMARY_DATASET.md.")
    st.markdown("[Dataset and attribution](https://github.com/garythung/trashnet)")
    st.markdown("[Project repository](https://github.com/haerinfan/wastesort)")
    st.divider()
    st.subheader("Goal Diggers presentation materials")
    st.write("Download the slides and speaking guides for group practice. Tuesday also includes a preparation notebook with saved outputs.")
    resources = {
        "Day 1: Monday": [
            ("PowerPoint slides", "presentation/WasteSort_Monday_Concepts.pptx"),
            ("PDF slides", "presentation/WasteSort_Monday_Concepts.pdf"),
            ("Speaking guide", "presentation/WasteSort_Monday_Speaking_Guide.pdf"),
        ],
        "Day 2: Tuesday": [
            ("PowerPoint slides", "presentation/day2/WasteSort_Tuesday_Dataset_Preparation.pptx"),
            ("PDF slides", "presentation/day2/WasteSort_Tuesday_Dataset_Preparation.pdf"),
            ("Speaking guide", "presentation/day2/WasteSort_Tuesday_Speaking_Guide.pdf"),
            ("Preparation notebook", "notebooks/WasteSort_Day2_Tuesday.ipynb"),
            ("Notebook with saved outputs", "notebooks/WasteSort_Day2_Tuesday_Executed.ipynb"),
        ],
    }
    for column, (day, files) in zip(st.columns(2), resources.items()):
        with column:
            with st.expander(day):
                for label, relative in files:
                    resource = ROOT / relative
                    if resource.exists():
                        st.download_button(label, resource.read_bytes(), file_name=resource.name, key=relative)
                    else:
                        st.caption(f"{label} is missing from this copy. Download the complete repository ZIP.")
    st.caption("AI assisted implementation and experiment execution. Group verification, interpretation, "
               "and contribution records remain part of the research.")
