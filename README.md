<h1 align="center">IRPNet [TGRS 2026]</h1>

<p align="left">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</p>

<p align="center">
  <strong>This is the official repository of the paper "IRPNet: Infrared Small Target Detection via RGB Prior Guidance and Physics Feature Fusion".</strong>
</p>

<p align="center">
  <em>Rui Yao, Nana Guo, Hancheng Zhu, Kunyang Sun, Fuyuan Hu, Xixi Li, and Jiaqi Zhao</em>
</p>


#### 1. Requirement
```bash
clip=1.0
cuda-nvcc=11.8.89
cudatoolkit=11.6.2
cudnn=8.9.7.29
numpy=1.22.3
opencv=4.6.0
python=3.8.18
pytorch=1.12.1
```

#### 2. Train.

```bash
python train.py
```


#### 3. Test.

```bash
python test.py --st_model IRPNet --model_dir NUAA-SIRST/resnet34/IRPNet/mIoU_IRPNet_NUAA-SIRST.pth.tar
```

#### 4. Visulize predicts with the best weight.
```bash
python visual_seg.py --st_model IRPNet --model_dir NUAA-SIRST/resnet34/IRPNet/pos/04/mIoU_IRPNet_NUAA-SIRST.pth.tar
```

#### 5. Test and visulization.
```bash
python test_and_visulization.py --st_model NUAA-SIRST_IRPNet --model_dir NUAA-SIRST_IRPNet/mIoU_IRPNet_NUAA-SIRST.pth.tar
```

#### Citation:
```bibtex
@article{yao2026irpnet,
  title={IRPNet: Infrared Small Target Detection via RGB Prior Guidance and Physics Feature Fusion},
  author={Yao, Rui and Guo, Nana and Zhu, Hancheng and Sun, Kunyang and Hu, Fuyuan and Li, Xixi and Zhao, Jiaqi},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
```