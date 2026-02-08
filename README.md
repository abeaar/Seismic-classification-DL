```markdown
# Multi-Domain Loss Convolutional Autoencoder  
## for Volcanic Seismic Signal Classification

This repository contains the implementation of a **Convolutional Autoencoder (CAE)** with a **multi-domain loss function** for volcanic seismic signal classification.  
The proposed approach integrates **time-domain reconstruction** and **frequency-domain constraints** to improve latent feature quality for downstream classification.

---

## Abstract

Volcanic seismic signal classification is challenged by non-stationary characteristics and high noise levels. This work proposes a Convolutional Autoencoder (CAE) trained using a multi-domain loss that jointly optimizes time-domain reconstruction accuracy and frequency-domain consistency. The learned latent representations are then utilized for classification using XGBoost. Experimental results and ablation studies demonstrate that the proposed approach outperforms conventional CAE-based feature extraction methods.

---

## Repository Structure

```

data/        Raw and preprocessed seismic signals
prepo/       Signal preprocessing and time–frequency transformation
model/       CAE architecture, multi-domain loss, and classifiers
experiments/ Experimental results and ablation studies

```

**Directory details:**
- **data/**
  - `raw/` : original seismic signals  
  - `processed/` : preprocessed data used for training and evaluation  
- **prepo/**
  - preprocessing, normalization, padding/trimming, and STFT computation  
- **model/**
  - CAE architecture, multi-domain loss implementation, and XGBoost classifier  
- **experiments/**
  - ablation studies and evaluation results  

---

## Method Overview

The overall processing pipeline is defined as follows:

```

Raw Seismic Signal
→ Preprocessing (normalization, trimming, padding)
→ Time–Frequency Transformation (STFT)
→ Convolutional Autoencoder (Multi-Domain Loss)
→ Latent Feature Extraction
→ XGBoost Classification

````

### Multi-Domain Loss Function

The CAE is optimized using a weighted combination of time-domain and frequency-domain losses:

\[
\mathcal{L} = \lambda_1 \mathcal{L}_{time} + \lambda_2 \mathcal{L}_{frequency}
\]

where:
- \(\mathcal{L}_{time}\) enforces reconstruction fidelity in the time domain  
- \(\mathcal{L}_{frequency}\) preserves spectral characteristics in the frequency domain  

---

## Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
````

or using Conda:

```bash
conda env create -f environment.yml
```

---

## Usage

### Preprocessing

```bash
python prepo/preprocessing.py
```

### Model Training

```bash
python model/train.py
```

### Feature Extraction and Classification

```bash
python model/xgboost_classifier.py
```

---

## Experiments and Evaluation

This repository includes ablation studies to evaluate:

* The impact of CAE-based feature extraction
* The contribution of the multi-domain loss formulation
* Performance comparison against baseline models

Experimental results are provided in the `experiments/` directory and correspond directly to the analysis reported in the associated thesis or publication.

---

## Dataset

The dataset is intended **for academic research use only**.

* Public datasets: access details are provided in `data/README.md`
* Restricted datasets: raw seismic data are not publicly available due to institutional or licensing constraints

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{yourname2026multidomain,
  title   = {Multi-Domain Loss for Convolutional Autoencoder in Volcanic Seismic Signal Classification},
  author  = {Your Name},
  journal = {Journal Name},
  year    = {2026}
}
```

---

## License

This project is released under the **MIT License** (or **CC BY-NC 4.0** for non-commercial research use).

```
```
