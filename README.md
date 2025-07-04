## [RecSys'25] Rethinking Overconfidence in VAEs: Can Label Smoothing Help?

This is the Pytorch implementation for RecSys Paper

### Data  

The datasets is available at `data/`.

### Requirements

```bash
conda create -n rethink python=3.8.20
pip install cornac==1.18.0
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118/
pip install pandas==2.0.3
```

### Run

Run `train.sh` to train DualVAE with LS : 

    bash train.sh


