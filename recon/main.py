import stir
import argparse
from reconstruction import *
from ct_to_acf import *
import json 

def reconstruction_pipeline(intemediates_dir,
         out_pet_nifti_path,
         ct_path,
         add_sino_path,
         mult_sino_path,
         prompts_sino_path,
         recon_template,
         filter_template,
         overwrite=False
         ):

    mumap_nifti_path = os.path.join(intemediates_dir, 'mumap.nii.gz')
    mumuap_smoothed_nifti_path = os.path.join(intemediates_dir, 'mumap_smoothed.nii.gz')
    mumap_hv_path = os.path.join(intemediates_dir, 'mumap_stir.hv')
    acf_sino_path = os.path.join(intemediates_dir, "acf.hs")
    add_acf_sino_path = os.path.join(intemediates_dir, "add.hs")
    mult_acf_sino_path = os.path.join(intemediates_dir, "mult.hs")
    pet_hv_path = os.path.join(intemediates_dir, "pet.hv")    

    os.makedirs(intemediates_dir, exist_ok=True)

    # Offset data needed to position the reconstructed PET image correctly in space
    with open("offset.json") as f:
        offsets = json.load(f)

    vertical_bed_start = offsets['vertical_bed_start']
    horizontal_bed_start = offsets['horizontal_bed_start']
    gantry_offset = offsets['gantry_offset']

    ## CT to ACF sinogram
    if not os.path.exists(mumap_nifti_path) or overwrite:
        print("Converting CT HU to mu-map...")
        mu = hu_to_mu(ct_path, kvp=120)
        mu.to_filename(mumap_nifti_path)

    if not os.path.exists(mumuap_smoothed_nifti_path) or overwrite:
        print("Smoothing mu-map...")
        mu_smoothed = smooth_image(mu, fwhm_mm=4)
        mu_smoothed.to_filename(mumuap_smoothed_nifti_path)

    if not os.path.exists(mumap_hv_path) or overwrite:
        print("Converting mu-map to STIR format...")
        mumap_to_stir(mumuap_smoothed_nifti_path, mumap_hv_path, ring_spacing_mm=3.29114)
    
    if not os.path.exists(acf_sino_path) or overwrite:
        print("Calculating ACF sinogram...")
        calculate_acf(mumap_hv_path, add_sino_path, acf_sino_path)

    ## Apply ACF to sinograms and reconstruct

    if not os.path.exists(add_acf_sino_path) or overwrite:
        print("Multiplying ACF on additive sinogram...")
        apply_acf_to_sinogram(add_sino_path, acf_sino_path, add_acf_sino_path)
    
    if not os.path.exists(mult_acf_sino_path) or overwrite:
        print("Multiplying ACF on multiplicative sinogram...")
        apply_acf_to_sinogram(mult_sino_path, acf_sino_path, mult_acf_sino_path)
    
    if not os.path.exists(pet_hv_path) or overwrite:
        print("Reconstructing PET image...")
        run_reconstruction(recon_template, filter_template, mult_acf_sino_path, prompts_sino_path, pet_hv_path)
    
    ## Convert back to nifti with correct origin
    if not os.path.exists(out_pet_nifti_path) or overwrite:
       print("Converting PET image to NIfTI format with correct origin...")
       stir_pet_to_nifti(vertical_bed_start,horizontal_bed_start,gantry_offset,pet_hv_path,out_pet_nifti_path)

    print(f"Done. Output PET image saved to {out_pet_nifti_path}")

if __name__ == "__main__":
    _recon_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="PET reconstruction pipeline")
    parser.add_argument("recon_folder", help="Folder containing input sinograms for a particular subject, e.g. '/path/to/sub-000/recon'")
    parser.add_argument("ct", help="CT input nifti file")
    parser.add_argument("pet_out", help="PET output NIfTI file")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Whether to overwrite existing intermediate and output files")
    parser.add_argument(
        "--intermediates_dir",
        default=None,
        help="Directory for intermediate files (default: <pet_out dir>/intermediates)",
    )
    args = parser.parse_args()

    intermediates_dir = args.intermediates_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.pet_out)), "intermediates"
    )

    reconstruction_pipeline(
        intemediates_dir=intermediates_dir,
        out_pet_nifti_path=args.pet_out,
        ct_path=args.ct,
        add_sino_path=os.path.join(args.recon_folder, "additive_term_SSRB.hs"),
        mult_sino_path=os.path.join(args.recon_folder, "mult_factors_forSTIR_SSRB.hs"),
        prompts_sino_path=os.path.join(args.recon_folder, "prompts_SSRB.hs"),
        recon_template=os.path.join(_recon_dir, "recon_OSEM_template.par"),
        filter_template=os.path.join(_recon_dir, "postfilter_4mm.par"),
        overwrite=args.overwrite,
    )
