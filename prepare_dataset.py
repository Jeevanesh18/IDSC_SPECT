# prepare_dataset.py
import os
import shutil
import json
import zipfile
import urllib.request
import SimpleITK as sitk

# ===============================
# 1️⃣ Config paths
# ===============================
DATA_ROOT = "/data"  # mount this folder from host when running Docker
DICOM_URL = "https://physionet.org/content/myocardial-perfusion-spect/get-zip/1.0.0/"  # adjust if needed

dicom_root = os.path.join(DATA_ROOT, "DICOM")
mask_root = os.path.join(DATA_ROOT, "NIfTI")
output_root = os.path.join(DATA_ROOT, "nnUNet_raw/Dataset999_SPECT")

imagesTr_dir = os.path.join(output_root, "imagesTr")
labelsTr_dir = os.path.join(output_root, "labelsTr")
imagesTs_dir = os.path.join(output_root, "imagesTs")

os.makedirs(dicom_root, exist_ok=True)
os.makedirs(mask_root, exist_ok=True)
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)
os.makedirs(imagesTs_dir, exist_ok=True)

zip_path = os.path.join(DATA_ROOT, "MPS_dataset.zip")
if not os.path.exists(zip_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DICOM_URL, zip_path)
    print("Download complete!")

# Temporary extraction folder
temp_extract = os.path.join(DATA_ROOT, "temp_extracted")

if not os.path.exists(temp_extract):
    os.makedirs(temp_extract, exist_ok=True)
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract)
    print("Extraction done!")

    # Move DICOM files to dicom_root
    extracted_dicom = os.path.join(temp_extract, "DICOM")
    if os.path.exists(extracted_dicom):
        for f in os.listdir(extracted_dicom):
            shutil.move(os.path.join(extracted_dicom, f), dicom_root)

    # Move NIfTI masks to mask_root
    extracted_masks = os.path.join(temp_extract, "NIfTI")
    if os.path.exists(extracted_masks):
        for f in os.listdir(extracted_masks):
            shutil.move(os.path.join(extracted_masks, f), mask_root)

    # Clean up temporary extraction folder
    shutil.rmtree(temp_extract)
    print("Flattening done! DICOM and NIfTI are ready.")

# ===============================
# 3️⃣ Prepare mask lookup
# ===============================
mask_files = sorted([f for f in os.listdir(mask_root) if f.endswith("_mask.nii.gz")])
mask_ids = [f.replace("_mask.nii.gz", "") for f in mask_files]
mask_dict = {mid: f for mid, f in zip(mask_ids, mask_files)}
print(f"Found {len(mask_dict)} masks")

# ===============================
# 4️⃣ Convert DICOM → NIfTI & split train/test
# ===============================
dicom_patients = sorted(os.listdir(dicom_root))
train_count, test_count = 0, 0

for patient_file in dicom_patients:
    patient_dicom_path = os.path.join(dicom_root, patient_file)
    print(f"Processing {patient_dicom_path}")

    # Read image (assumes one file per patient; adjust if folder)
    image = sitk.ReadImage(patient_dicom_path)
    patientId = os.path.splitext(patient_file)[0]

    if patientId in mask_dict:  # has mask → train
        output_file = os.path.join(imagesTr_dir, f"patient{train_count+1:03d}_0000.nii.gz")
        mask_file = os.path.join(mask_root, mask_dict[patientId])
        label_file = os.path.join(labelsTr_dir, f"patient{train_count+1:03d}.nii.gz")

        sitk.WriteImage(image, output_file)
        shutil.copyfile(mask_file, label_file)
        train_count += 1
        print(f"[TRAIN] {patientId} -> image: {output_file}, mask: {label_file}")
    else:  # no mask → test
        output_file = os.path.join(imagesTs_dir, f"patient{test_count+1:03d}_0000.nii.gz")
        sitk.WriteImage(image, output_file)
        test_count += 1
        print(f"[TEST] {patientId} -> image: {output_file} (no mask)")

print(f"Total training patients: {train_count}")
print(f"Total test patients: {test_count}")

# ===============================
# 5️⃣ Fix mask shapes
# ===============================
print("Checking and fixing mask shapes...")

images = sorted(os.listdir(imagesTr_dir))
labels = sorted(os.listdir(labelsTr_dir))

for img_name, lbl_name in zip(images, labels):
    img_path = os.path.join(imagesTr_dir, img_name)
    lbl_path = os.path.join(labelsTr_dir, lbl_name)

    img = sitk.ReadImage(img_path)
    lbl = sitk.ReadImage(lbl_path)

    if img.GetSize() != lbl.GetSize():
        print(f"Fixing {lbl_name} | Mask {lbl.GetSize()} -> Image {img.GetSize()}")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        fixed_lbl = resampler.Execute(lbl)
        sitk.WriteImage(fixed_lbl, lbl_path)

# ===============================
# 6️⃣ Create dataset.json
# ===============================
training_list = []
for img_file, label_file in zip(sorted(os.listdir(imagesTr_dir)), sorted(os.listdir(labelsTr_dir))):
    training_list.append({
        "image": f"./imagesTr/{img_file}",
        "label": f"./labelsTr/{label_file}"
    })

test_list = [f"./imagesTs/{f}" for f in sorted(os.listdir(imagesTs_dir))]

dataset_json_v2 = {
    "name": "SPECT",
    "description": "PhysioNet Myocardial Perfusion SPECT dataset",
    "tensorImageSize": "3D",
    "channel_names": {"0": "SPECT"},
    "labels": {"background": 0, "LV": 1},
    "numTraining": len(training_list),
    "training": training_list,
    "test": test_list,
    "file_ending": ".nii.gz"
}

with open(os.path.join(output_root, "dataset.json"), "w") as f:
    json.dump(dataset_json_v2, f, indent=4)

print("Dataset preparation complete! Ready for nnU-Net preprocessing.")
