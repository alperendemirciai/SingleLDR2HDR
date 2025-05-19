Sure! Here's a polished `README.md` file based on your outline:

---

# SingleLDR2HDR

This repository contains the source code for training and evaluating an **image-to-image translation model** that reconstructs HDR (High Dynamic Range) images from a single LDR (Low Dynamic Range) image. The project supports various model architectures and upsampling methods and provides tools for training, validation, and testing with quantitative evaluation metrics.

---

## 📂 Project Structure

```
.
├── dataset/                # Custom dataset class for HDRRealDataset
├── data/                # Directory for LDR-HDR pair dataset.
    ├── LDR_in/                # LDR inputs as .jpeg files.
    ├── HDR_gt/                # Corresponding HDR images as .hdr files.
├── models/                 # Contains model implementations (UNet, model_a, model_b)
├── outputs/                # Saved checkpoints, results, and plots from training/testing
├── utils/                  # Helper functions and argument parsers
├── train_val.py           # Training and validation pipeline
├── test.py                # Testing script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 Features

* Custom models for HDR image prediction from LDR inputs:

  * UNet
  * model\_a
  * model\_b (e.g., MobileNetV3\_UNet)
* Flexible training-validation-test split ratios
* Pixel-wise loss using a combination of **L1**, **LPIPS**.
* Metrics for evaluation:

  * PSNR (Peak Signal-to-Noise Ratio)
  * SSIM (Structural Similarity Index Measure)
* Image denormalization and saving utilities
* Checkpointing and result plotting

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Train a Model

To train a model on the HDR-Real dataset:

```bash
python train_val.py --data ./data/HDR-Real --save_dir ./outputs/3rd_run_model_b --lr 0.001 --batch_size 32 --epochs 30 --print_every 2 --upsampling_method pixelshuffle --random_state 20 --save_freq 1 --train_ratio 0.05 --val_ratio 0.01 --test_ratio 0.01 --model model_b
```

---

## 🧪 Test a Model

To test a pretrained model and evaluate its performance:

```bash
python test.py --data ./data/HDR-Real --save_dir ./outputs/3rd_run_model_b --batch_size 32 --upsampling_method pixelshuffle --random_state 31 --train_ratio 0.05 --val_ratio 0.01 --test_ratio 0.01 --model model_b --save_n 3
```

Make sure to use the same `--random_state` and split ratios as in training to ensure consistency.

---

## 📊 Output

The following will be saved to the specified `--save_dir`:

* Best model checkpoint: `checkpoints/gen_best.pth`
* Example output images: `results/`
* Metric plots: `plots/`
* Console logs showing training/validation/test progress with pixel loss, PSNR, and SSIM

---

## 📎 Notes

* Input and output images are normalized to `[-1, 1]` during training.
* LPIPS loss requires a pretrained VGG-based feature extractor (automatically handled).

---

## 📬 Contact

If you have any questions or issues, feel free to open an issue or reach out to the project maintainer.

---
