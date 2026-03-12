import stir
import os, subprocess
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

def hu_to_mu(ct_path, kvp=120):
    """Carney et al. 2006 (Med Phys 33:976-983) bilinear HU to mu at 511 keV."""
    bone_slope = {80: 3.84e-5, 100: 4.56e-5, 120: 5.10e-5, 140: 5.64e-5}

    ct = nib.load(ct_path)
    hu = ct.get_fdata(dtype=np.float32)

    mu = np.where(hu <= 0,
                  9.6e-5 * (hu + 1000),
                  9.6e-5 * 1000 + bone_slope[kvp] * hu)
    mu = np.clip(mu, 0, None)

    return nib.Nifti1Image(mu, ct.affine, ct.header)


def smooth_image(img, fwhm_mm=4.0):
    """Gaussian smoothing. FWHM in mm, converted to sigma per axis."""
    voxel_sizes = img.header.get_zooms()[:3]
    sigma = [fwhm_mm / (2.355 * v) for v in voxel_sizes]
    smoothed = gaussian_filter(img.get_fdata(dtype=np.float32), sigma=sigma)
    return nib.Nifti1Image(smoothed, img.affine, img.header)

def save_stir_to_nifti(stir_img, output_path):
    stir.ITKOutputFileFormat().write_to_file(output_path, stir_img)

def calculate_acf(mumap_hv, reference_sinogram, output_hs):
    subprocess.run(['calculate_attenuation_coefficients', '--ACF', output_hs, mumap_hv, reference_sinogram], check=True)
    

def mumap_to_stir(input_path, output_path, ring_spacing_mm):
    """Zero origins, resample z to ring_spacing/2, snap z-origin to grid."""
    plane_sep = ring_spacing_mm / 2

    img = stir.FloatVoxelsOnCartesianGrid.read_from_file(input_path)
    img.set_origin(stir.FloatCartesianCoordinate3D(0.0, 0.0, 0.0))

    v = img.get_voxel_size()
    zoom_z = v.z() / plane_sep
    max_idx, min_idx = img.get_max_indices(), img.get_min_indices()
    z_size = int(max_idx[1] - min_idx[1]) + 1
    xy_size = int(max_idx[2] - min_idx[2]) + 1

    new_sizes = stir.Int3BasicCoordinate()
    new_sizes[1] = round(z_size * zoom_z)
    new_sizes[2] = xy_size
    new_sizes[3] = xy_size

    img_z = stir.zoom_image(img,
        stir.FloatCartesianCoordinate3D(zoom_z, 1.0, 1.0),
        stir.FloatCartesianCoordinate3D(0.0, 0.0, 0.0),
        new_sizes, stir.ZoomOptions(stir.ZoomOptions.preserve_values))

    o2 = img_z.get_origin()
    snapped_z = round(o2.z() / plane_sep) * plane_sep
    img_z.set_origin(stir.FloatCartesianCoordinate3D(snapped_z, 0.0, 0.0))

    stir.InterfileOutputFileFormat().write_to_file(output_path, img_z)
    print(f"z-origin snapped: {o2.z():.4f} -> {snapped_z:.4f} mm, plane_sep={plane_sep:.5f} mm")


# def convert_ct_to_acf(ct_path, reference_sinogram, output_hs, ring_spacing_mm=3.29114,fwhm_mm=4.0,kvp=120):
#     root = os.path.dirname(output_hs)
#     mumap_hv_path = os.path.join(root, 'mumap_stir.hv')
#     mumap_nifti_path = os.path.join(root, 'mumap.nii.gz')
#     mumuap_smoothed_nifti_path = os.path.join(root, 'mumap_smoothed.nii.gz')
#     os.makedirs(root, exist_ok=True)

#     if not os.path.exists(mumap_nifti_path):
#         print("Converting CT HU to mu-map...")
#         mu = hu_to_mu(ct_path, kvp=kvp)
#         mu.to_filename(mumap_nifti_path)

#     if not os.path.exists(mumuap_smoothed_nifti_path):
#         print("Smoothing mu-map...")
#         mu_smoothed = smooth_image(mu, fwhm_mm=fwhm_mm)
#         mu_smoothed.to_filename(mumuap_smoothed_nifti_path)

#     if not os.path.exists(mumap_hv_path):
#         print("Converting mu-map to STIR format...")
#         mumap_to_stir(mumuap_smoothed_nifti_path, mumap_hv_path, ring_spacing_mm)
    
#     if not os.path.exists(output_hs):
#         print("Calculating ACF sinogram...")
#         calculate_acf(mumap_hv_path, reference_sinogram, output_hs)


# if __name__ == "__main__":
#     convert_ct_to_acf('ct.nii.gz', 'additive_term_SSRB.hs', 'outs/acf.hs', ring_spacing_mm=3.29114, debug=True)