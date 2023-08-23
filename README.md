# Environment
```
Ubuntu            16.04
Python            3.8.10
Tensorflow-gpu    2.5.0
CUDA              11.2
```

# Train and Test ContextNet
1. Download UrbanLF-Syn dataset 
2. Run `python train_urban.py` to train model
  - Checkpoint files will be saved in **'LF_checkpoints/XXX_ckp/iterXXXX_valmseXXXX_bpXXX.hdf5'**.
  - Training process will be saved in 
    - **'LF_output/XXX_ckp/train_iterXXXXX.jpg'**
    - **'LF_output/XXX_ckp/val_iterXXXXX.jpg'**.
3. Run `python evaltion_urban.py`
  - `path_weight='pretrained_contextnet.hdf5'`
# Submit ContextNet 
- Run `python submission_urban.py`
  - `path_weight='pretrained_contextnet.hdf5'`



The code and data are available in the [https://drive.google.com/file/d/1tRKyA74IzwETa4RLGxrSHFmYBzrpVU6q/view?usp=drive_link](https://drive.google.com/file/d/1tRKyA74IzwETa4RLGxrSHFmYBzrpVU6q/view?usp=drive_link).