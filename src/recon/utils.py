import hashlib
import json
import os


def hash_file(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


RECON_FILES = [
    "ct_face_and_bed.nii.gz",
    "face_and_bed_mask.nii.gz",
    "add_nac_rd85.hs",
    "mult_nac_rd85.hs",
    "prompts_rd85.hs",
    "offset.json",
]


def validate_recon_dir(recon_dir):
    missing = [f for f in RECON_FILES if not os.path.exists(os.path.join(recon_dir, f))]
    if missing:
        raise FileNotFoundError(
            "Missing required files in recon directory:\n" +
            "\n".join(f"  {f}" for f in missing) +
            "\nPlease specify a valid recon folder, e.g. bic-mac-dataset/train/sub-XXX/recon/"
        )


def check_input_hashes(intermediates_dir, ct_path, ct_face_and_bed_path, overwrite):
    hashes_path = os.path.join(intermediates_dir, 'input_hashes.json')
    current = {'ct': hash_file(ct_path), 'ct_face_and_bed': hash_file(ct_face_and_bed_path)}
    if os.path.exists(hashes_path) and not overwrite:
        saved = json.load(open(hashes_path))
        if saved != current:
            raise RuntimeError(
                "Input CT file or recon folder differ from the previous run in this output directory. "
                "Re-run with -w/--overwrite, delete the intermediates folder, "
                "or choose a different output directory."
            )
    else:
        json.dump(current, open(hashes_path, 'w'))
