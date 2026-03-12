
import sys, json, re

if __name__ == "__main__":
    sino_hdr, lm_hdr, output = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(sino_hdr) as f: sino = f.read()
    with open(lm_hdr) as f: lm = f.read()

    vert = float(re.search(r"start vertical bed position \(mm\)\s*:=\s*([-\d.]+)", sino).group(1))
    horiz = float(re.search(r"start horizontal bed position \(mm\)\s*:=\s*([-\d.]+)", sino).group(1))
    gantry_z = float(re.search(r"%gantry offset \(mm\) \[3\]\s*:=\s*([-\d.]+)", lm).group(1))

    with open(output, 'w') as f:
        json.dump({'vertical_bed_start': vert, 'horizontal_bed_start': horiz, 'gantry_offset': gantry_z}, f)