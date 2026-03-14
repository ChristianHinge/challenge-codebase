import os


def get_dataset(data_dir):

    subjects = sorted(os.listdir(data_dir))

    data = []

    for sub in subjects:

        sub_path = os.path.join(data_dir, sub)

        features = os.path.join(sub_path, "features")
        ct_label = os.path.join(sub_path, "ct-label")

        item = {
            "pet": os.path.join(features, "nacpet.nii.gz"),
            "topogram": os.path.join(features, "topogram.nii.gz"),
            "mri_in": os.path.join(features, "mri_combined_in_phase.nii.gz"),
            "mri_out": os.path.join(features, "mri_combined_out_phase.nii.gz"),
            "ct": os.path.join(ct_label, "ct.nii.gz"),
        }

        data.append(item)

    return data