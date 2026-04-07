## Suggested public datasets


### PET/CT
> Note that all PET images in the following datasets are attenuation corrected (usually by the accompanying CT), which means that the PET may encode some of the CT information.
- [Vienna QUADRA_HC](https://zenodo.org/records/16588733) 96 whole-body 18F-FDG PET/CT studies from 48 participants. Like BIC-MAC, the PET/CT is acquired on a Siemens Biograph Vision Quadra and the participants are healthy controls. [citation](https://www.nature.com/articles/s41597-025-05997-4)

- [AutoPET V](https://fdat.uni-tuebingen.de/records/0zs4c-89f12) 1014 whole-body 18F-FDG PET/CT studies, 597 PSMA PET/CT studies [citation](https://www.nature.com/articles/s41597-022-01718-3)

- [PETWB-REP](https://zenodo.org/records/18670487) 565 whole-body 18F-FDG PET/CT studies [citation](https://arxiv.org/pdf/2508.04062)

- [ENHANCE.PET](https://pubmed.ncbi.nlm.nih.gov/40799763/) 1,597 whole-body 18F-FDG PET/CT studies. Downloaded by running `moosez -dtd -dd path/to/download/` (install [moosez](https://github.com/ENHANCE-PET/MOOSE)) [citation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12340901/#S1).

- [ViMED-PET](https://huggingface.co/datasets/dacthai2807/ViMed-PET)  2,757 whole-body 18F-FDG PET/CT studies. [citation](https://arxiv.org/abs/2509.24739v1)

- [Lung-PET-CT-Dx](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/) 436 whole-body (no head) 18F-FDG PET/CT studies [citation](https://doi.org/10.7937/TCIA.2020.NNC2-0461)

- [Deep-PSMA](https://zenodo.org/records/15281784) 100 whole-body PSMA and 18F-FDG PET/CT studies [citation](https://doi.org/10.5281/zenodo.15281783)

### MRI/CT

- [SynthRAD2025](https://zenodo.org/records/14918089) 890 paired MRI–CT and 1,472 CBCT–CT sets covering head-and-neck, thorax, and abdomen from 5 European university medical centers. [citation](https://arxiv.org/abs/2502.17609)

- [CHAOS](https://zenodo.org/records/3431873) 40 abdominal CT and 40 abdominal MRI studies (T1-DUAL, T2-SPIR) from healthy subjects. CT and MRI are from **different** patients (unpaired). [citation](https://doi.org/10.1016/j.media.2020.101950)

- [Paired CT–MRI (T1+T2)](https://doi.org/10.1016/j.dib.2025.111768) Small co-registered CT and MRI (T1- and T2-weighted) dataset from the same patients. [citation](https://doi.org/10.1016/j.dib.2025.111768)

- [Learn2Reg Abdomen MR-CT](https://learn2reg.grand-challenge.org/Datasets/) 16 paired and 90 unpaired abdominal CT and MRI scans [citation](https://doi.org/10.1109/TMI.2022.3213983)

- [RIRE](https://rire.insight-journal.org/) ~20 brain patients with paired CT, T1, T2, and PD MRI with gold-standard marker-based registration transforms. [citation](https://rire.insight-journal.org/)


### CT

- [NLST](https://www.cancerimagingarchive.net/collection/nlst/) ~26,000 low-dose chest CT studies. [citation](https://doi.org/10.7937/TCIA.HMQ8-J677)

- [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) 47,149 chest CT volumes with paired radiology reports. [citation](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)

- [TotalSegmentator CT](https://zenodo.org/records/10047292) 1,228 whole-body CT studies with segmentations of 117 anatomical structures. [citation](https://doi.org/10.1148/ryai.230024)

- [AbdomenAtlas 1.0 Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini) 5,195 abdominal CT studies with 9-organ segmentations. [citation](https://arxiv.org/abs/2305.09666)

### MRI
- [TotalSegmentator MRI](https://zenodo.org/records/14710732) 616 whole-body MRI studies. [citation](https://doi.org/10.1148/ryai.230024)

- [FOMO-300K](https://huggingface.co/datasets/FOMO-MRI/FOMO300K) 81,282  brain MRI studies with a total of 306,303 scans. [citation](https://arxiv.org/abs/2506.14432)

### Chest X-ray (Topogram-like)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) 224,316 chest radiographs of 65,240 patients with 14 pathology labels. [citation](https://arxiv.org/abs/1901.07031)
