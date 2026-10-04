<div align="center">

<h1>
  EchoScene: A Bio-Inspired Sonar Dataset for Depth Prediction
</h1>

### ACCV 2026 🇯🇵🏯

<a href="https://www.linkedin.com/in/nz-ismail">Nazrul Ismail</a><sup>1,2</sup>,
<a href="https://www.linkedin.com/in/muhd-amirul-raziq-hj-rosman-511b09299/">Muhammad Amirul Raziq Rosman</a><sup>1,2</sup>,
Owais Ahmed Malik<sup>3</sup>,
<a href="https://ailab.space/">Ong Wee Hong</a><sup>1,2</sup>

<sup>1</sup>School of Digital Science, Universiti Brunei Darussalam<br>
<sup>2</sup>Robotics and Intelligent Systems Laboratory (RoboLab)<br>
<sup>3</sup>Atlantic Technological University<br>

[![CVF - Coming Soon](https://img.shields.io/badge/CVF-Coming_Soon-yellow?logo=ieee&logoColor=white)](#)
[![Dataset Status](https://img.shields.io/badge/Dataset-Upon_Request_(Filtering_Scenes)-yellow?logo=database&logoColor=white)](#dataset) </div>
---

## Abstract
>Echolocation is a foundational exteroceptive sense for bats, enabling rich spatial perception in the complete absence of light. While
prior art has shown that real-world echoes can support audio-based depth prediction, existing datasets remain limited to audible frequencies with fixed chirp waveforms that hinder usability, leaving spatial understanding from biosonar underexplored. We present EchoScene, the
first biosonar dataset comprising 15,187 synchronised audio-depth samples captured across 25 indoor locations spanning three floors, including scenes recorded under extremely low-light conditions for an audioonly ablation. We conduct benchmark evaluation binaural vs monaural on EchoScene and also show that cross-modal distillation from vision foundation models enables echo-based perception without visual
input at test time. By releasing aligned acoustic and visual modalities, EchoScene aims to advance bio-inspired robotic sensing beyond what existing real-world and simulated datasets allow
> 

---

## Environment Setup

**Tested on:** Python 3.12, PyTorch 2.2.1, NVIDIA RTX 4090

The training code is the official [V2E-CCL](https://github.com/ailabspace/Visual2Echo-Compositional-Contrastive-Learning-for-Depth-Prediction) release, pulled in as a git submodule under `third_party/Visual2Echo`.

```bash
git clone --recursive https://github.com/ailabspace/SonarScene-A-Bio-Inspired-Sonar-Dataset-for-Depth-prediction.git
cd SonarScene-A-Bio-Inspired-Sonar-Dataset-for-Depth-prediction
# already cloned without --recursive:
git submodule update --init

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively with conda:
```bash
conda create -n v2e python=3.10
conda activate v2e
pip install -r requirements.txt
```

## Dataset

> **Note:** The dataset is currently being processed to filter scenes and refine annotations. During this period, access is available **upon request**. 

If you need access to the dataset for research or reproducibility purposes, please email to [Nazrul Ismail](mailto:23h1701@ubd.edu.bn).

Each recording is one HDF5 file, `<rec>.h5`, with the binaural echoes (320 kHz) and the RealSense D435i depth (mm). The matching RGB is a separate `<rec>_rgb.h5`. The RGB is a resize of the colour stream and does **not** share the depth camera's field of view. `benchmark/registration.py` warps colour-frame maps onto the depth grid using the D435i calibration.

Once you have the release, build the flat views the loaders expect. The script symlinks the files and copies the split files, and it lists any recordings that are missing:

```bash
python3 scripts/prepare_data.py --release /path/to/EchoScene
```

The test floor is held out of training and checkpoint selection. The dark floors have no lit counterpart and are used for evaluation only. Window lists are in `splits/{board,dark}/`.

---


## Training

All flags live in `scripts/common.sh`. Every script takes `[SEED] [extra train_ccl.py flags]` where noted. Outputs go to `work/` (override with `WORK=`), and the data view defaults to `data/board` (override with `DATA=`).

```bash
# echo-only U-Net baseline
bash scripts/train_unet.sh 1

# V2E-CCL: fine-tune the MoGe-2 teacher, cache its features, then distil into the echo network
bash scripts/finetune_teacher.sh        # -> work/teacher/moge2_vits_ft/model.pt, selected on val
bash scripts/precompute_teacher.sh      # -> work/latents/teacher_{train,valtest}.h5
bash scripts/train_v2e_ccl.sh 1
```

Each run is linked as `work/runs/<exp>`. Checkpoint selection uses the val split only (`audiodepth_biosonar_bestval.pth`).

---

## Benchmark

```bash
bash scripts/benchmark.sh                  # every run in work/runs/ + RGB baselines
bash scripts/benchmark.sh work/runs/unet_s1  # selected runs
RGB=0 bash scripts/benchmark.sh            # echo models only
```

The benchmark scores every model on the board test floor and on the dark floors, and writes `work/results/*.json` plus a summary table in `work/results/benchmark.md`.

- **Echo models**: per-window `ABS_REL`, `SQ_REL`, `RMSE`, `LOG10`, `MAE`, δ<1.25/1.25²/1.25³ over GT-valid pixels (≤ 8 m) on the 128×128 depth grid, with no scaling. They are reported on the full grid, and also on the colour-camera FOV so the comparison with RGB is like-for-like.
- **RGB baselines**: MoGe-2 ViT-L, Depth Anything 3, and the fine-tuned teacher. Each prediction is registered to the depth grid and scored inside the colour FOV with three protocols: `raw` (metric), `median` (median-scaled) and `affine` (least-squares scale + shift).

---

## Architecture

- **Audio stream** (`SimpleAudioDepthNet`): BinauralResNet18/34 or MALFNet on STFT magnitudes (+ sin/cos IPD with `--use_ipd`, ILD with `--use_ild`) → U-Net decoder → `[B, 1, 128, 128]`
- **Visual teacher**: frozen RGB U-Net (`RGBDepthNet`) or MoGe-2 (`--rgb_teacher moge_v2`, live or cached)
- **Material stream** (`MaterialPropertyNet`): frozen ResNet18 → 23 MINC classes
- **CCL**: `CompositionalEmbedding` heads align audio features with teacher depth and material features; `ProjectionHead`s for the contrastive term

---

## Project Structure

```
scripts/prepare_data.py       build data/board and data/dark from the release
scripts/common.sh             shared data / training / teacher flags
scripts/train_unet.sh         echo-only U-Net
scripts/finetune_teacher.sh   fine-tune MoGe-2 ViT-S on the train split
scripts/precompute_teacher.sh cache teacher features (HDF5)
scripts/train_v2e_ccl.sh      V2E-CCL with the cached teacher
scripts/benchmark.sh          full benchmark + summary table
benchmark/eval_echo.py        score an echo checkpoint
benchmark/eval_rgb.py         MoGe-2 / Depth Anything 3 baselines
benchmark/registration.py     D435i colour -> depth warp
teacher/ft_moge_teacher.py    teacher fine-tuning
splits/                       board and dark-floor window lists
third_party/Visual2Echo/      V2E-CCL (submodule)
```

## Acknowledgement
Some codes in this repo are adapted from [VisualEchoes](https://github.com/facebookresearch/VisualEchoes.git) and [Beyond Image to Depth](https://github.com/krantiparida/beyond-image-to-depth).  We thank the authors for making their code and ideas publicly available.

## Citation
If you find this work or the code useful in your research, please consider citing our paper:
```bibtex
@InProceedings{Ismail_2026_ACCV,
    author    = {Ismail, Nazrul and Rosman, Muhammad Amirul Raziq and Malik, Owais Ahmed and Hong, Ong Wee},
    title     = {EchoScene: A Bio-Inspired Sonar Dataset for Depth prediction},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2026},
    note      = {To appear}
}

@InProceedings{Ismail_2026_CVPR,
    author    = {Ismail, Nazrul and Malik, Owais Ahmed and Hong, Ong Wee},
    title     = {Visual2Echo Compositional Contrastive Learning (V2E-CCL): Binaural Knowledge Distilled Network for Depth Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {6019-6028}
}
```
