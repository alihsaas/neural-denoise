# NeuralDenoise

> I measured the reference results using the original benchmark environment.

**Real phone-camera noise: how much does a small learned denoiser beat the
classical filters, and what does each one cost in compute time?**

I compare classical and learned denoisers on the same real noisy photos from
five smartphones, then time each one on the CPU the way it would run on a small
device.

```text
noisy sRGB photo -> classical: non-local means (OpenCV), BM3D
                 -> learned:   DnCNN (predicts the noise and subtracts it),
                               full size and small, 3 seeds each, GPU training
                 -> deploy:    best DnCNN -> ONNX FP32 -> ONNX INT8
                 -> PSNR and SSIM against the clean photo, and CPU milliseconds
                    per 256 x 256 block
```

## Setup

Python 3.12, [uv](https://docs.astral.sh/uv/), PyTorch for CUDA 12.8.

```powershell
uv sync --extra neural
uv run pytest -q --basetemp=.pytest-tmp
```

`run_study.bat` runs the tests, the data check, the split, the three studies and
the gain comparison.

## Data and split

SIDD, the Smartphone Image Denoising Dataset (Abdelhamed, Lin and Brown, 2018).
I train on SIDD-Small (160 noisy and clean sRGB pairs), with 20 scene instances
held out for validation. I test on the official SIDD sRGB validation blocks:
1280 blocks of 256 x 256 pixels from 40 other photos. Every parameter and
checkpoint is chosen on my validation crops; each method sees the test blocks
once, except the NLM and BM3D rerun I explain under Results.

## Metrics

- **PSNR**: peak signal-to-noise ratio in dB, from the mean squared error on
  8-bit pixels. Higher is better; +1 dB is about 21% less squared error.
- **SSIM**: structural similarity, which compares local contrast and structure
  rather than raw pixel error. 1.0 means identical.
- **95% CI**: bootstrap over the 40 test photos, since blocks from one photo
  share its noise level.
- **Gain over BM3D**: the PSNR difference to BM3D on the same blocks, with the
  same photo bootstrap.
- **Speed**: median milliseconds per 256 x 256 block on the CPU, at 1 thread and
  all 12 threads, and on the GPU for DnCNN.

## Results

All numbers are on the 1280 test blocks. Times are median milliseconds per
256 x 256 block from `scripts/run_deploy_study.py`, on an Intel i5-12450H (12
logical CPUs) and an RTX 3050 Laptop GPU. A dash means not measured.

| Method | Test PSNR, dB (95% CI) | SSIM | Gain over BM3D, dB (95% CI) | CPU ms, 1 thread | CPU ms, 12 threads | GPU ms |
|---|---|---|---|---|---|---|
| Noisy input | 23.66 (22.24-24.94) | 0.3335 | -9.88 (-10.52 to -9.22) | - | - | - |
| NLM, h 20 | 30.97 (29.14-32.69) | 0.7680 | -2.57 (-3.00 to -2.12) | 195.3406 | 194.0830 | - |
| BM3D, sigma 40 | 33.54 (31.60-35.35) | 0.8054 | - | 4452.8099 * | 4039.4872 * | - |
| DnCNN 17 x 64, 558403 parameters, seeds 0 / 1 / 2 | 37.55 / 37.59 / 37.53 | 0.8948 / 0.8952 / 0.8949 | 4.01 / 4.05 / 3.98 | - | - | 14.7865 |
| DnCNN 10 x 32, 76003 parameters, seeds 0 / 1 / 2 | 37.07 / 37.10 / 37.13 | 0.8941 / 0.8944 / 0.8956 | 3.53 / 3.56 / 3.58 | - | - | 4.5033 |
| DnCNN seed 1, ONNX FP32, 2.2844 MB | 37.59 (36.32-38.84) | 0.8952 | 4.05 (3.35-4.85) | 800.3821 | 218.5283 | - |
| DnCNN seed 1, ONNX INT8, 0.6257 MB | 37.39 (36.16-38.60) | 0.8911 | 3.85 (3.12-4.67) | 241.4929 | 58.1600 | - |

\* The `bm3d` package ignores my thread setting, so both BM3D times ran on
several cores. GPU times are PyTorch with float16, one block at a time,
including the copies to and from the GPU, for the seed with the best validation
PSNR (seed 1 for the full model, seed 2 for the small one).

- Every DnCNN run beats my tuned BM3D by 3.53 to 4.05 dB, and no 95% interval
  for the gain goes below 2.83 dB.
- The small DnCNN, with 76003 instead of 558403 parameters, still reaches
  37.099 dB against 37.555 dB (mean over the 3 seeds).
- INT8 shrinks the ONNX file from 2.2844 MB to 0.6257 MB and lowers test PSNR
  from 37.59 to 37.39 dB.
- On one thread NLM (195.3406 ms) is a little faster than INT8 DnCNN
  (241.4929 ms). On 12 threads INT8 DnCNN is the fastest CPU method
  (58.1600 ms), because NLM did not get faster with more threads.
- My first grids stopped at h 14 and sigma 30, and both picks sat at the top.
  I widened the grids, chose again on validation, and scored NLM and BM3D on
  the test blocks a second time. They scored 29.62 and 32.51 dB before; the
  first run is kept in `results/classical_narrow_grid`.
- While the times were measured, other programs used 39.2 to 47.3 percent of
  one CPU core (`results/logs/deploy_load.csv`).

## References

- A. Abdelhamed, S. Lin, M. S. Brown. "A High-Quality Denoising Dataset for
  Smartphone Cameras." CVPR 2018.
- A. Buades, B. Coll, J.-M. Morel. "A Non-Local Algorithm for Image Denoising."
  CVPR 2005.
- K. Dabov, A. Foi, V. Katkovnik, K. Egiazarian. "Image Denoising by Sparse 3-D
  Transform-Domain Collaborative Filtering." IEEE TIP 2007.
- K. Zhang, W. Zuo, Y. Chen, D. Meng, L. Zhang. "Beyond a Gaussian Denoiser:
  Residual Learning of Deep CNN for Image Denoising." IEEE TIP 2017.
