# TOTNet: Temporal and Spatial Network for Ball Tracking

TOTNet is specifically designed to utilize temporal and spatial information for ball tracking, especially in challenging occlusion scenarios.

---

## Environment Setup

### Recommended Environment

- **Python Version:** 3.10

### Installation Steps

1. Clone this repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd TOTNet
   ```

2. Install dependencies from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the installation:
   ```bash
   python --version  # Ensure it is Python 3.10
   ```

---

## Data

### Table Tennis Data

The TTNet dataset can be downloaded from [TTNet GitHub Repository](https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch).

### Tennis and Badminton Data

Download the tennis and badminton datasets from [WASB-SBDT Repository](https://github.com/nttcom/WASB-SBDT?tab=readme-ov-file).

### Dataset Organization

After downloading, organize your datasets as follows:
```
data/
├── tta_dataset/
├── tennis_data/
├── badminton_data/
```

---

## How to Run

### Training Command

Use the following command to train the model:
```bash
torchrun --nproc_per_node=3 main.py \
    --num_epochs 30 \
    --saved_fn 'tracking_288_512_motion_light_TTA(5)_new_data' \
    --num_frames 5 \
    --optimizer_type adamw \
    --lr 5e-4 \
    --loss_function WBCE \
    --weight_decay 5e-5 \
    --img_size 288 512 \
    --batch_size 24 \
    --print_freq 100 \
    --dist_url 'env://' \
    --dist_backend 'nccl' \
    --multiprocessing_distributed \
    --distributed \
    --dataset_choice 'tta' \
    --weighting_list 1 2 2 3 \
    --model_choice 'motion_light' \
    --occluded_prob 0.1 \
    --ball_size 4 \
    --val-size 0.2 \
    --no_test
```

### Explanation of Arguments

- `--nproc_per_node=3`: Specifies the number of GPUs to use.
- `--num_epochs`: Number of training epochs.
- `--saved_fn`: Name of the folder to save results.
- `--num_frames`: Number of consecutive frames to process.
- `--optimizer_type`: Optimizer choice (e.g., `adamw`).
- `--lr`: Learning rate.
- `--loss_function`: Loss function (e.g., `WBCE` for weighted binary cross-entropy).
- `--img_size`: Resolution of input images (height x width).
- `--batch_size`: Batch size for training.
- `--dist_url` and `--dist_backend`: Used for distributed training setup with `torchrun`.
- `--dataset_choice`: Dataset to use (`tta`, `tennis`, or `badminton`).
- `--model_choice`: Model to use (e.g., `motion_light`).
- `--occluded_prob`: Probability of occlusion during training.
- `--val-size`: Validation dataset size as a fraction of the training dataset.
- `--no_test`: Disable testing after training.

---

## Notes for Debugging

1. **Distributed Training:**
   - Ensure `--nproc_per_node` matches the number of GPUs available.
   - If debugging or testing on a single GPU, set `--nproc_per_node=1` and remove `--multiprocessing_distributed` and `--distributed`.

2. **Dataset Preparation:**
   - Verify the dataset paths are correct.
   - Organize the datasets as described in the `Dataset Organization` section.

3. **Logs and Checkpoints:**
   - Checkpoint files and logs will be saved in the directory specified by `--saved_fn`.
   
---

## Example Dataset Structure

```
data/
├── tta_dataset/
│   ├── training/
│   │   ├── images/
│   │   ├── labels.csv
│   ├── test/
│   │   ├── images/
│   │   ├── labels.csv
├── tennis_data/
├── badminton_data/
```

---

## Example for Single GPU Debugging

To debug or run on a single GPU, use the following modified command:
```bash
python main.py \
    --num_epochs 10 \
    --saved_fn 'debug_run' \
    --num_frames 5 \
    --optimizer_type adamw \
    --lr 5e-4 \
    --loss_function WBCE \
    --weight_decay 5e-5 \
    --img_size 288 512 \
    --batch_size 16 \
    --dataset_choice 'tta' \
    --model_choice 'motion_light' \
    --val-size 0.2
```
This will run on a single GPU without distributed training.

---

For further details or troubleshooting, refer to the documentation or open an issue in the repository. Happy Training!

