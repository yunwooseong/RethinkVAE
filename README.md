## [RecSys'25] Rethinking Overconfidence in VAEs: Can Label Smoothing Help?
[![View Paper](https://img.shields.io/badge/View%20Paper-PDF-E24D35)](https://dl.acm.org/doi/10.1145/3705328.3748039) [![DOI](https://img.shields.io/badge/DOI-10.1145/3705328.3748039-blue)](https://doi.org/10.1145/3705328.3748039)

## 📄 Paper
This is the Pytorch implementation for RecSys Paper

## 📊 Data  

The datasets is available at `data/`.

## 🛠️ Requirements

```bash
conda create -n rethink python=3.8.20
pip install cornac==1.18.0
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118/
pip install pandas==2.0.3
```

## 🚀 Run

Run `train.sh` to train DualVAE with LS : 

    bash train.sh

## 🙏 Acknowledgements
This work was partly supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-00553785) and the Institute of Information \& Communications Technology Planning \& Evaluation (IITP) grant funded by the Korean government (MSIT) (RS-2021-II211341, Artificial Intelligence Graduate School Program of Chung-Ang University)
