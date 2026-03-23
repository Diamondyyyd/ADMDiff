# ADMDiff

This repository provides the official implementation of **ADMDiff**:  
**Adaptive Dynamic Decomposition-Driven Masked Diffusion for Multivariate Time Series Anomaly Detection**.

ADMDiff is an unsupervised anomaly detection framework for multivariate time series.  
The proposed method integrates adaptive dynamic decomposition with masked diffusion modeling to enhance robustness against complex temporal patterns and missing or corrupted observations.  
We evaluate ADMDiff on three widely used public benchmarks and demonstrate its effectiveness compared with existing approaches.

---

## Results

The main experimental results are summarized in the following table.  
Our method achieves superior performance on the majority of evaluation metrics across multiple benchmark datasets.

![Image Description](result.png)

---

## Data

We evaluate ADMDiff on the following datasets:

- **PSM**
- **SMD**
- **SWaT**

### Dataset Preparation

Two benchmark datasets can be downloaded from [Google Cloud](https://drive.google.com/drive/folders/1UJ6SGfb6h-9R0L18FLDXpISKh1nhaqWA), where the data have been well pre-processed.  

---

## Environment Setup

We recommend using Conda to create an isolated environment.

```bash
conda create -n admdiff python=3.10.17
conda activate admdiff
pip install -r requirements.txt
```
Before training the diffusion model, the decomposition module should be trained first. For example, using the SMD dataset:

```shell
python d3r_decomposer.py --dataset SMD --device 0
```

After the decomposition model is trained, run the following command to train ADMDiff:

```shell
python train_model.py --device cuda:0 --dataset SMD
```
After completing training, perform inference using:
```shell
python evaluate_model.py --device cuda:0 --dataset SMD
```
After inference, anomaly scores can be computed using:
```shell
python evaluation.py --dataset_name SMD
```
The final results will be saved in the corresponding result directory.
