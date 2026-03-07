import streamlit as st
import os
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from skimage import measure

# Page Config
st.set_page_config(layout="wide", page_title="SPECT LV Explorer")

# ==========================================
# 1. Configuration & Caching
# ==========================================
DEMO_FOLDER = "demo_data"

@st.cache_data
def load_nifti(filepath):
    """Loads a NIfTI file and returns the data array."""
    if not os.path.exists(filepath):
        return None
    nii = nib.load(filepath)
    # Rotate to align typically with viewer (optional, depends on your data orientation)
    return np.rot90(nii.get_fdata(), k=1) 

@st.cache_data
def get_3d_mesh(mask_data):
    """Generates 3D mesh data (verts/faces) from the mask volume."""
    try:
        # Marching cubes algorithm to find the surface of the LV (label=1)
        verts, faces, _, _ = measure.marching_cubes(mask_data, level=0.5)
        return verts, faces
    except (ValueError, IndexError):
        return None, None

# ==========================================
# 2. Sidebar & File Selection
# ==========================================
st.sidebar.title("🫀 Control Panel")

# Check if folder exists
if not os.path.exists(DEMO_FOLDER):
    st.error(f"❌ Error: Folder '{DEMO_FOLDER}' not found in repository.")
    st.stop()

# Find all mask files (files that do NOT have _0000)
available_patients = sorted([
    f.replace(".nii.gz", "") 
    for f in os.listdir(DEMO_FOLDER) 
    if f.endswith(".nii.gz") and "_0000" not in f
])

if not available_patients:
    st.error("No prediction masks found in 'demo_data'. Make sure files are named correctly (e.g., patient001.nii.gz)")
    st.stop()

selected_id = st.sidebar.selectbox("Select Test Patient", available_patients)

# Construct paths
mask_path = os.path.join(DEMO_FOLDER, f"{selected_id}.nii.gz")
raw_path = os.path.join(DEMO_FOLDER, f"{selected_id}_0000.nii.gz")

# Load Data
mask_vol = load_nifti(mask_path)
raw_vol = load_nifti(raw_path)

if raw_vol is None:
    st.sidebar.warning(f"⚠️ Raw image not found for {selected_id}. Showing mask only.")
    raw_vol = mask_vol # Fallback

# ==========================================
# 3. Main Dashboard
# ==========================================
st.title(f"Patient ID: {selected_id}")

col1, col2 = st.columns([1, 1])

# --- LEFT: 2D SLICER ---
with col1:
    st.subheader("2D Slice Viewer")
    
    # Slider for Z-axis
    z_max = raw_vol.shape[2] - 1
    z_idx = st.slider("Select Slice (Z-Axis)", 0, z_max, z_max // 2)
    
    # Create Plotly Heatmap
    fig_2d = go.Figure()
    
    # 1. The Raw SPECT Image
    fig_2d.add_trace(go.Heatmap(
        z=raw_vol[:, :, z_idx],
        colorscale='Magma', # 'Magma' looks very medical/nuclear
        name="SPECT Signal"
    ))
    
    # 2. The Prediction Overlay
    mask_slice = mask_vol[:, :, z_idx]
    y_indices, x_indices = np.where(mask_slice > 0)
    
    if len(x_indices) > 0:
        fig_2d.add_trace(go.Scatter(
            x=x_indices, y=y_indices,
            mode='markers',
            marker=dict(color='#00FF00', size=3, opacity=0.4), # Green overlay
            name="LV Prediction"
        ))

    fig_2d.update_layout(
        width=500, height=500,
        title=f"Slice {z_idx} (Green = Predicted Wall)",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_2d, use_container_width=True)

# --- RIGHT: 3D MODEL ---
with col2:
    st.subheader("3D Reconstruction")
    
    verts, faces = get_3d_mesh(mask_vol)
    
    if verts is not None:
        fig_3d = go.Figure(data=[go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='red',
            opacity=0.8,
            flatshading=True,
            name="Left Ventricle"
        )])
        
        fig_3d.update_layout(
            width=500, height=500,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data' # Keeps the heart shape correct, not stretched
            ),
            title="Interactive 3D View (Drag to Rotate)",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("Empty mask - No Left Ventricle detected for this patient.")

st.markdown("---")
st.caption("SPECT-LV Segmenter | Built with Streamlit & nnU-Net")
