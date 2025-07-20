import streamlit as st
import os
import shutil
from pathlib import Path
from model_generator import run_colmap_pipeline
import pyvista as pv
from stpyvista import stpyvista

st.set_page_config(layout="wide")
st.title("3D Model Generator from Images")

# Setup directories
image_dir     = "images"
workspace_dir = "colmap_workspace"
output_dir    = "outputs"
Path(image_dir).mkdir(exist_ok=True)
Path(output_dir).mkdir(exist_ok=True)

# Image upload
uploaded = st.file_uploader(
    "Upload 3–5 images of the same object (jpg/jpeg/png)",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if uploaded:
    if not (3 <= len(uploaded) <= 5):
        st.warning("Please upload between 3 and 5 images.")
    else:
        # Clear previous
        shutil.rmtree(image_dir, ignore_errors=True)
        Path(image_dir).mkdir()
        for img in uploaded:
            with open(f"{image_dir}/{img.name}", "wb") as f:
                f.write(img.read())
        st.success(f"{len(uploaded)} images uploaded.")

        if st.button("Generate 3D Model"):
            with st.spinner("Running COLMAP pipeline… this can take a minute"):
                try:
                    model_path = run_colmap_pipeline(image_dir, workspace_dir)
                    # Copy to outputs
                    shutil.copy(model_path, f"{output_dir}/model.ply")
                    st.success("✅ 3D model generated successfully!")

                    # Viewer
                    st.subheader("Interactive 3D Preview")
                    plotter = pv.Plotter(window_size=[800,600])
                    mesh = pv.read(model_path)
                    plotter.add_mesh(mesh, show_edges=True)
                    stpyvista(plotter)

                    # Download
                    with open(f"{output_dir}/model.ply","rb") as f:
                        st.download_button("Download as .PLY", f, "model.ply")
                except Exception as e:
                    st.error(f"⚠️  {e}")

# Sidebar reset
if st.sidebar.button("Reset Everything"):
    for d in (image_dir, workspace_dir, output_dir):
        shutil.rmtree(d, ignore_errors=True)
    st.experimental_rerun()
