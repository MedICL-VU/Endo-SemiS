# Endo-SemiS
Endo-SemiS: Towards Robust Semi-supervised Image Segmentation for Endoscopic Video

This work has been submitted to MIDL2026

##
Our work can be summarized as follows: \
(a) We use a cross-supervision framework to avoid biased learning from a single network. \
(b) We use uncertainty to improve the quality of each network’s pseudo-labels. \
(c) When one network’s prediction has a large defect with high confidence values, we fuse a joint pseudo-label by selecting the most confident regions from both network and use this pseudo-label to supervise them. \
(d) We use multi-level mutual learning to further mitigate confirmation bias and improve consistency between networks, producing more reliable pseudo-labels.

<img src='figs/framework_new.png' width='800'>

The details of our methods and results can be viewed in the paper.

## Usage

**Installation**
```
conda create -n endosemis python=3.9
conda activate endosemis
pip install -r requirements.txt # or conda env create -f environment.yml
```

**Train kidney dataset**

```
python train.py --name <your running name> --json_path <your json file path> --height 256 --width 256
```
The code will automatically create a folder to store logs under /src/checkpoints/your running name/

**Test kidney dataset**

```
python test.py --name your running name --input <your test image folder> --label <your test label folder> --height 256 --width 256
```
use --save to save the predictions


**Train polyp dataset**

```
python train_semi_polygen.py --name <your running name> --json_path <your json file path>
```

**Test polyp dataset**

```
python test_semi_polygen.py --name <your running name> --json_path <your json file path>
```

same augments will be used for **train_semi_polygen.py** and **test_sup_polygen.py**

## Datasets

Kidney stone dataset: this is a in-house dataset and we are not able to share it. If you are an internal collaborator, please reach out to me.

Polyp Screening dataset: the dataset used in our experiments can be viewed [here](https://drive.google.com/drive/u/1/folders/1BOXBV-FuvldKuylV82Yqf2F9HOb1ru7S), more information can be viewed via this [link](https://github.com/DebeshJha/PolypGen).

Our training/test data split is [here](https://github.com/MedICL-VU/Endo-SemiS/tree/main/src/data_split).

## Pretrained models and logs
| kidney Endo-SemiS | polyp Endo-SemiS | polyp full sup | polyp semi10 sup |
|------------------------------|------------------------------|------------------------------|------------------------------|
| Internal, currently (contact me) |[Download](https://drive.google.com/drive/u/1/folders/1yW8wm1IKgKUbqxoCRjtfC4uKsuVGJdif)| [Download](https://drive.google.com/drive/u/1/folders/1HyaMSjIDLcjZXCS8qz6FMO7TFTdd1N7e) |[Download](https://drive.google.com/drive/u/1/folders/1Qs0m1GiR_KwFvjOce8R48dmxAD5kbcgY)|

Test and training logs are attached to these links.

## Quantitative results of kidney dataset (10% labeled data)


**Kidney results (mean ± stdev., in %) with 10% labeled data.** Sections: supervised, semi-supervised (single network), cross-supervised, and supervised with 100% labeled data (upper bound). Our method achieves the highest Dice, Sensitivity, F1, and Accuracy.

🏆 = best

| Group | Methods | Dice | Sensitivity | Specificity | Pre. | Rec. | F1 | Acc. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Supervised (10%) | U-Net | 80.5±32.1 | 88.6±22.0 | 95.4±8.4 | 88.7 | 95.3 | 92.8 | 90.1 |
| Supervised (10%) | nnU-Net | 79.5±33.8 | 85.9±27.4 | 95.5±9.1 | 90.1 | 91.1 | 90.6 | 87.6 |
| Semi-supervised (single network) | Generic | 78.5±31.7 | 86.1±25.7 | 92.3±13.9 | 90.7 | 95.3 | 92.9 | 90.5 |
| Semi-supervised (single network) | AllSpark | 77.0±31.2 | 88.0±24.8 | 89.3±18.0 | 94.7 | 92.8 | 93.8 | 91.7 |
| Semi-supervised (single network) | UPRC | 80.7±31.4 | 84.0±27.3 | 96.4±7.8 | 92.9 | 94.6 | 93.7 | 91.6 |
| Semi-supervised (single network) | FixMatch | 81.9±31.7 | 89.8±22.4 | 94.3±10.9 | 89.7 | `96.5` 🏆 | 93.0 | 90.5 |
| Semi-supervised (single network) | UniMatch | 85.5±27.6 | 89.4±23.2 | 95.5±8.9 | 94.3 | 96.4 | 95.4 | 91.7 |
| Semi-supervised (single network) | Mean Teacher | 82.2±31.2 | 84.1±28.6 | 96.6±8.5 | 95.6 | 90.5 | 93.0 | 91.1 |
| Cross-supervised | CPS | 85.2±28.0 | 88.8±22.8 | 95.8±8.8 | 94.0 | 96.1 | 95.0 | 93.4 |
| Cross-supervised | Cross Teaching | 85.6±28.7 | 87.6±26.5 | `96.7±7.4` 🏆 | `96.5` 🏆 | 92.6 | 94.8 | 92.9 |
| Cross-supervised | Endo-SemiS (Ours) | `87.6±26.4` 🏆 | `91.1±21.5` 🏆 | 96.0±8.4 | 95.0 | 96.1 | `95.6` 🏆 | `94.1` 🏆 |
| Upper bound (100%) | Upper bound U-Net | 85.3±29.2 | 89.0±24.5 | 96.5±8.2 | 94.4 | 94.2 | 94.3 | 92.5 |
| Upper bound (100%) | Upper bound nnU-Net | 85.5±28.5 | 89.3±24.5 | 96.0±8.6 | 92.4 | 93.3 | 92.9 | 90.5 |





## Qualitative results of kidney dataset (10% labeled data)
The kidney stone laser lithotomy (surgery) exhibits large variation in image quality due to the complex in vivo environment during surgery. Here we show qualitative kidney stone results (10\% labeled data). Yellow circles highlight poor visibility areas. (a) fiberoptic frames, (b) digital frames, (c) fluid distortions,  (d) motion blur, (e) debris during stone ablation, and (f) illumination changes.
<img src='figs/qualitative_zoom.png' width='800'>

## Citing Endo-SemiS
If you find our Endo-SemiS helpful, please use the following BibTeX entry.

```
@inproceedings{li2024interactive,
  title={Endo-SemiS: Towards Robust Semi-Supervised Image Segmentation for Endoscopic Video,
  author={Li, Hao and Lu, Daiwei, Wang, Jiacheng and Kavoussi, Nicholas and Oguz, Ipek},
  booktitle={Medical Imaging in Deep Learning},
  year={2026 (submitted)}
}
```
## Contact
Email: hao.li.1@vanderbilt.edu

## Acknowledgements
[SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

