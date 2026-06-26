# CTBENet

CTBENet is a PyTorch project for RGB-thermal semantic segmentation experiments.

## Project Structure

- `net/`: CTBENet model code and DINOv3-related modules.
- `toolbox/`: datasets, losses, metrics, optimizer utilities, and training helpers.
- `configs/`: dataset and experiment configuration files.
- `evaluate.py`: evaluation entry point.
- `CTGloss.py`: thermal causal effect loss implementation.

## Notes

- Large model weights such as `*.safetensors`, `*.pt`, `*.pth`, and `*.ckpt` are tracked with Git LFS.
- Runtime caches, TensorBoard logs, local datasets, and experiment outputs are excluded from Git.

## Requirements

Install the Python dependencies listed in `net/requirements.txt`, together with the CUDA/PyTorch versions required by your environment.
