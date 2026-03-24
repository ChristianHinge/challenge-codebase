import logging
import re
import numpy as np
import shutil
import subprocess
import stir
import os
from tqdm import tqdm

log = logging.getLogger('recon')

def apply_acf_to_sinogram(sino_path, sino_acf_path, out_sino_path):
    sino_path_s = sino_path.replace(".hs", ".s")
    sino_3d_path_s = sino_acf_path.replace(".hs", ".s")
    out_sino_path_s = out_sino_path.replace(".hs", ".s")

    a4d = np.fromfile(sino_path_s, dtype=np.float32).reshape((33, 50, 5189, 520))
    a3d = np.fromfile(sino_3d_path_s, dtype=np.float32).reshape((50, 5189, 520))
    a4d *= a3d
    a4d.tofile(out_sino_path_s)
    shutil.copy(sino_path, out_sino_path)
    # Update the data file reference in the copied header
    new_data_filename = os.path.basename(out_sino_path_s)
    with open(out_sino_path, "r") as f:
        header = f.read()
    header = re.sub(r"(?m)^(name of data file\s*:=\s*).*$", r"\g<1>" + new_data_filename, header)
    with open(out_sino_path, "w") as f:
        f.write(header)



def stir_pet_to_nifti(vertical_bed_start,horizontal_bed_start,gantry_offset,pet_hv_path,output_path):
    """Fix STIR .hv origin from E7 headers. Usage: python fix.py e7_sino.s.hdr listmode.hdr pet.hv output.hv"""
    #import re, sys, stir

    image = stir.FloatVoxelsOnCartesianGrid.read_from_file(pet_hv_path)

    stir_z = -(horizontal_bed_start + gantry_offset) 
    stir_y = -vertical_bed_start

    image.set_origin(stir.FloatCartesianCoordinate3D(float(stir_z), float(stir_y), 0.0))
    stir.ITKOutputFileFormat().write_to_file(output_path, image)
    

def run_reconstruction(recon_template, add_sino_path, mult_sino_path, prompts_sino_path, out_image_path, verbose=False):
    out_image_base = out_image_path.replace("_20.hv", "")

    with open(recon_template,"r") as f:
        recon_cmd = f.read().strip()

    recon_cmd = recon_cmd.replace("PROMPTS_SINO", prompts_sino_path)
    recon_cmd = recon_cmd.replace("ADD_SINO", add_sino_path)
    recon_cmd = recon_cmd.replace("MULT_SINO", mult_sino_path)
    recon_cmd = recon_cmd.replace("OUT_FILE_PREFIX", out_image_base)

    recon_file = os.path.join(os.path.dirname(out_image_path), 'recon.par')
    with open(recon_file, "w") as f:
        f.write(recon_cmd)

    subiteration_re = re.compile(r'OSEM subiteration #(\d+) completed')

    with subprocess.Popen(['stdbuf', '-oL', 'OSMAPOSL', recon_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        with tqdm(total=20, desc='OSEM subiteration', unit='subit', leave=False) as pbar:
            for line in proc.stdout:
                line = line.rstrip()
                log.debug(line)
                m = subiteration_re.search(line)
                if m:
                    pbar.update(int(m.group(1)) - pbar.n)
        proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)


