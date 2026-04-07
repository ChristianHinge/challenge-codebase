WIP
# Tips & FAQ

---

**Do I need to understand PET reconstruction to participate?**

No. You only need to predict a pseudo-CT from the input features. The reconstruction pipeline is provided and run for you. See [pet-background.md](pet-background.md) if you want to understand what the reconstruction does and why CT quality matters for PET accuracy.

**Can I use external data for training, finetuning or validation?**

No, no external data is allowed for the challenge. Please see the [rules](rules.md) for clarification.

**Can I use pretrained models?**
Yes, you are allowed to use and finetune pretrained models, as longs as the following conditions are met:
- The model is publicly available
- The model was released prior to the start of the challenge (April 1st) 


**What subjects have sinogram data for local reconstruction?**

Sinogram data is provided for the following subjects:
- `train/`: `sub-000`, `sub-001`, `sub-002`, `sub-005`, `sub-006`, `sub-008`, `sub-013`, `sub-014`
`val/`: `sub-004`, `sub-009` ,`sub-010` ,`sub-018`
The remaining 67 training subjects have `features/` and `ct-label/` only — you can train and evaluate CT metrics on them, but cannot run closed-loop PET reconstruction locally. We chose to provide sinogram data for only 8 of the 67 training subjects to keep the dataset size managable. 

**What does the face mask do?**

The face region is replaced with ground-truth CT values before reconstruction (step 2 of the pipeline). This means errors in face prediction do not affect PET or CT metrics — your model is only evaluated on the body region. `features/mri_face_mask.nii.gz` marks the corresponding region in MRI space if you want to apply the same masking during training.

**Why is `prediction_mask.nii.gz` in `ct-label/`?**

It marks the voxels your model is responsible for predicting (body minus face and scanner bed). During training you may want to restrict your loss to this mask so the model is not penalised for face/bed regions that are overwritten anyway during reconstruction.

**The MRI comes in chunks — do I need to stitch them?**

Pre-stitched versions (`mri_combined_in_phase.nii.gz`, `mri_combined_out_phase.nii.gz`) are provided if you want a single whole-body volume. The individual chunks (`mri_chunk_{0-3}_{in/out}_phase.nii.gz`) are available if you prefer to work per bed position. 


**What format does my pseudo-CT need to be in?**

A NIfTI file (`.nii.gz`) in Hounsfield units, with the same shape and affine as `features/nacpet.nii.gz`. Copying the header directly from the NAC-PET when saving is the safest approach:

```python
ref = nib.load("features/nacpet.nii.gz")
nib.save(nib.Nifti1Image(pred_hu, ref.affine, ref.header), "ct.nii.gz")
```

**Do I need to install STIR locally?**

No, you do not even need to run reconstruction locally - unless you want to validate using the PET-based challenge metrics. If we you do wish to do reconstruction, we recommend using the Docker image (`ghcr.io/bic-mac-challenge/recon:latest`), which includes STIR and all dependencies. The image wraps the python code in (`src/recon`). You can run call this code directly if you have a local STIR build. Please see [STIR User Guide](https://stir.sourceforge.net/documentation/STIR-UsersGuide.pdf) for installation instructions. IMPORTANT: Make sure to install STIR from source and not a prepackaged version, since the critical reconstruction bugs related to Quadra Sinograms remain present in version 6.3. 

**Reconstruction is slow — how long should I expect it to take?**

Roughly 20–120 minutes per subject on a modern CPU, dominated by the OSEM reconstruction step (step 9). Intermediate outputs are cached, so re-runs resume from where they left off unless `OVERWRITE=1` is set.

**How do I debug a failed reconstruction?**

Check `output_dir/intermediates/recon.log` for the full STIR log. Rerun with `VERBOSE=1` (Docker) or `-v` (Python) to stream STIR output to the terminal in real time.

**Which metrics are reported on the validation leaderboard?**

Four metrics in total: Whole-body SUV MAE, Brain Outlier Score, Organ Bias, and CT μ-MAE. See the evaluation section of the main README for descriptions. The Brain Outlier Score is a dataset-level metric — it cannot be computed for a single subject. The fifth and final metric "TAC Bias" is only computed for the final test set. The metric calculation requires reconstruction using dynamic sinograms, which are unfortunately too large to share. 

**Can I evaluate without running reconstruction?**

Yes — CT μ-MAE only requires your pseudo-CT and the ground-truth CT. Pass `--pred_ct` without `--pred_pet` to `eval_subject.py`:

```bash
python src/evaluation/eval_subject.py \
  --subject_dir data/sub-000 \
  --pred_ct outputs/sub-000/ct.nii.gz
```


**Do I need to submit both a pseudo-CT and a PET?**
For the validation phase, you can submit CT-only, PET-only, or both to CodaBench in a zip file. Please see [Submission Guide](submission-guide.md) for instructions. Submitting both PET and CT unlocks all four metrics. Note that to submit a PET image, you have to run reconstruction locally. For the Final Test phase, the organizers will run reconstruction so you only submit the the Docker image with your pseudo-CT model. 

**How can I make sure that my submitted Docker image will work?**
Once the validation phase starts, you can submit your pseudo-CT container for "Dry-Run". The organizers will the run your container on the hardware used for final evaluation and report back the CT-based metrics for the validation set. This way you can check that the container runs successfully and within the 5-minute time limit. 

**Does my final container need to be the same as the dry-run container?**
No. But we recommend doing a dry-run for the container you intend to submit for the final validation to ensure that it will not crash or run out of memory. 

