import streamlit as st
from pathlib import Path
from validator import run_validation

st.set_page_config(page_title="CS Crop Validation", layout="wide")

st.title("🌾 Crop Segmentation Validation")
st.markdown("Validate **Wherobots vs Legacy** CS rasters")

# -----------------------------
# User Inputs
# -----------------------------
legacy_dir = st.text_input("Legacy Folder Path", "/data/legacy")
new_dir    = st.text_input("Wherobots Folder Path", "/data/wherobots")
out_dir    = st.text_input("Output Folder", "/data/output")
season     = st.text_input("Season (e.g. K025)", "K025")

# -----------------------------
# Run Validation
# -----------------------------
if st.button("🚀 Run Validation"):
    if not Path(legacy_dir).exists():
        st.error("Legacy folder not found")
    elif not Path(new_dir).exists():
        st.error("Wherobots folder not found")
    else:
        st.info("Running validation... this may take a few minutes ⏳")

        try:
            run_validation(legacy_dir, new_dir, out_dir, season)
            st.success("Validation completed!")

            summary_csv = Path(out_dir) / "RID_Validation_Summary.csv"
            matrix_csv  = Path(out_dir) / "RID_CropSwitch_Matrix.csv"

            if summary_csv.exists():
                st.subheader("📄 RID Validation Summary")
                st.dataframe(
                    summary_csv.read_text(),
                    use_container_width=True
                )
                st.download_button(
                    "Download RID_Validation_Summary.csv",
                    summary_csv.read_bytes(),
                    file_name="RID_Validation_Summary.csv"
                )

            if matrix_csv.exists():
                st.subheader("📊 Crop Switch Matrix")
                st.dataframe(
                    matrix_csv.read_text(),
                    use_container_width=True
                )
                st.download_button(
                    "Download RID_CropSwitch_Matrix.csv",
                    matrix_csv.read_bytes(),
                    file_name="RID_CropSwitch_Matrix.csv"
                )

        except Exception as e:
            st.error(f"Validation failed: {e}")
