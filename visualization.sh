import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Paths (adjust if needed)
# ===============================
DATA_ROOT = "/data" 
output_root = os.path.join(DATA_ROOT, "nnUNet_raw/Dataset999_SPECT")
imagesTs_dir = os.path.join(output_root, "imagesTs")
predictions_dir = os.path.join(DATA_ROOT, "nnUNet_results/Dataset999_SPECT/predictions")
output_dir = os.path.join(DATA_ROOT, "nnUNet_results/Dataset999_SPECT/visualizations")
os.makedirs(output_dir, exist_ok=True)

# ===============================
# Function to combine mask & heatmap
# ===============================
def plot_combined(img_path, mask_path, npz_path, save_path):
    img = nib.load(img_path).get_fdata()
    mask = nib.load(mask_path).get_fdata() if mask_path and os.path.exists(mask_path) else None
    probs = np.load(npz_path)['probabilities'] if npz_path and os.path.exists(npz_path) else None
    lv_heatmap = probs[1] if probs is not None else None

    if lv_heatmap is not None and lv_heatmap.shape != img.shape:
        lv_heatmap = lv_heatmap.transpose(2, 1, 0)

    slices = [int(img.shape[2]*0.4), int(img.shape[2]*0.5), int(img.shape[2]*0.6)]
    slice_idx = img.shape[2] // 2  # middle slice for heatmap

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for i, s in enumerate(slices):
        # Raw SPECT
        axes[0, i].imshow(img[:, :, s], cmap='hot')
        axes[0, i].set_title(f"Raw SPECT - Slice {s}")
        axes[0, i].axis('off')

        # Mask overlay
        axes[1, i].imshow(img[:, :, s], cmap='gray')
        if mask is not None:
            axes[1, i].imshow(mask[:, :, s], cmap='jet', alpha=0.5)
        axes[1, i].set_title(f"Prediction Overlay - Slice {s}")
        axes[1, i].axis('off')

        # Heatmap overlay (middle slice only for last column)
        if i == 1:
            axes[2, i].imshow(img[:, :, slice_idx], cmap='gray')
            if lv_heatmap is not None:
                heat = axes[2, i].imshow(lv_heatmap[:, :, slice_idx], cmap='jet', alpha=0.6)
                fig.colorbar(heat, ax=axes[2, i], label="Probability of LV")
            axes[2, i].set_title("LV Probability Heatmap (Middle Slice)")
            axes[2, i].axis('off')
        else:
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# ===============================
# Loop through test images
# ===============================
for img_file in sorted(os.listdir(imagesTs_dir)):
    if not img_file.endswith(".nii.gz"):
        continue
    
    img_path = os.path.join(imagesTs_dir, img_file)
    patientId_mask = img_file.replace("_0000.nii.gz", ".nii.gz")
    mask_path = os.path.join(predictions_dir, patientId_mask)
    npz_file = img_file.replace("_0000.nii.gz", ".npz")
    npz_path = os.path.join(predictions_dir, npz_file)
    
    pdf_path = os.path.join(output_dir, img_file.replace(".nii.gz", "_visualization.pdf"))
    plot_combined(img_path, mask_path, npz_path, pdf_path)

print(f"All visualizations saved in {output_dir}")
