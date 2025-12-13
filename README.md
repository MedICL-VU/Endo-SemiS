# Endo-SemiS
Endo-SemiS: Towards Robust Semi-supervised Image Segmentation for Endoscopic Video

This work has been submitted to MIDL2026


<img src='figs/qualitative_zoom.pdf' width='600'>

PRISM is a robust model/method for interactive segmentation in medical imaging. We strive for human-level performance, as a human-in-loop interactive segmentation model with prompts should gradually refine its outcomes until they closely match inter-rater variability.

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



## Pretrained models
| kidney Endo-SemiS | polyp Endo-SemiS | polyp full sup | polyp semi10 sup |
|------------------------------|------------------------------|------------------------------|------------------------------|
| Internal, currently (contact me) |[Download](https://drive.google.com/drive/u/1/folders/1yW8wm1IKgKUbqxoCRjtfC4uKsuVGJdif)| [Download](https://drive.google.com/drive/u/1/folders/1HyaMSjIDLcjZXCS8qz6FMO7TFTdd1N7e) |[Download](https://drive.google.com/drive/u/1/folders/1Qs0m1GiR_KwFvjOce8R48dmxAD5kbcgY)|

Test and training logs are attached to these links.

## Acknowledgements
[SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

