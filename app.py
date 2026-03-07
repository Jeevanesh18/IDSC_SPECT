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
    # Rotate to align typically with viewer
    return np.rot90(nii.get_fdata(), k=1) 

@st.cache_data
def get_3d_mesh_with_texture(mask_data, raw_data):
    """
    Generates 3D mesh from mask and samples the raw intensity 
    values to color the surface.
    """
    try:
        # 1. Generate the shape (vertices and faces) from the mask
        verts, faces, _, _ = measure.marching_cubes(mask_data, level=0.5)
        
        # 2. Map the raw SPECT intensity onto these vertices
        # We round the vertex coordinates to integers to find the pixel value
        x_idx = np.clip(verts[:, 0].astype(int), 0, raw_data.shape[0]-1)
        y_idx = np.clip(verts[:, 1].astype(int), 0, raw_data.shape[1]-1)
        z_idx = np.clip(verts[:, 2].astype(int), 0, raw_data.shape[2]-1)
        
        # Get the intensity values for every vertex
        intensities = raw_data[x_idx, y_idx, z_idx]
        
        return verts, faces, intensities
    except (ValueError, IndexError):
        return None, None, None

# ==========================================
# 2. Sidebar & File Selection
# ==========================================
st.sidebar.title("🫀 Control Panel")

# Check if folder exists
if not os.path.exists(DEMO_FOLDER):
    st.error(f"❌ Error: Folder '{DEMO_FOLDER}' not found.")
    st.stop()

# Find available patients
available_patients = sorted([
    f.replace(".nii.gz", "") 
    for f in os.listdir(DEMO_FOLDER) 
    if f.endswith(".nii.gz") and "_0000" not in f
])

if not available_patients:
    st.error("No prediction masks found in 'demo_data'.")
    st.stop()

selected_id = st.sidebar.selectbox("Select Test Patient", available_patients)

# Construct paths
mask_path = os.path.join(DEMO_FOLDER, f"{selected_id}.nii.gz")
raw_path = os.path.join(DEMO_FOLDER, f"{selected_id}_0000.nii.gz")

# Load Data
mask_vol = load_nifti(mask_path)
raw_vol = load_nifti(raw_path)

if raw_vol is None:
    st.sidebar.warning(f"⚠️ Raw image not found for {selected_id}.")
    raw_vol = mask_vol 

# ==========================================
# 3. Main Dashboard
# ==========================================
st.title(f"Patient ID: {selected_id}")

col1, col2 = st.columns([1, 1])

# --- LEFT: 2D SLICER ---
with col1:
    st.subheader("2D Slice Viewer")
    
    z_max = raw_vol.shape[2] - 1
    z_idx = st.slider("Select Slice (Z-Axis)", 0, z_max, z_max // 2)
    
    # Create Plotly Heatmap
    fig_2d = go.Figure()
    
    # The Raw SPECT Image
    fig_2d.add_trace(go.Heatmap(
        z=raw_vol[:, :, z_idx],
        colorscale='Magma', 
        name="SPECT Signal"
    ))
    
    # The Prediction Overlay (Green Outline)
    mask_slice = mask_vol[:, :, z_idx]
    
    # Use Contour for cleaner look than scatter
    fig_2d.add_trace(go.Contour(
        z=mask_slice,
        showscale=False,
        contours=dict(start=0.5, end=0.5, size=2, coloring='lines'),
        line=dict(color='#00FF00', width=2),
        name="LV Prediction"
    ))

    fig_2d.update_layout(
        width=500, height=500,
        title=f"Slice {z_idx}",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_2d, use_container_width=True)

# --- RIGHT: 3D MODEL WITH PERFUSION MAP ---
with col2:
    st.subheader("3D Perfusion Map")
    
    verts, faces, intensities = get_3d_mesh_with_texture(mask_vol, raw_vol)
    
    if verts is not None:
        fig_3d = go.Figure(data=[go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=intensities,    # <--- THIS IS THE MAGIC
            colorscale='Magma',       # Matches the 2D view
            showscale=True,           # Show color bar
            opacity=1.0,
            flatshading=False,
            name="Left Ventricle"
        )])
        
        fig_3d.update_layout(
            width=500, height=500,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            ),
            title="Interactive 3D Perfusion (Color = Blood Flow)",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.info("💡 **Clinical Insight:** Brighter areas indicate healthy blood perfusion. Darker areas may indicate ischemia.")
    else:
        st.warning("Empty mask - No Left Ventricle detected.")
