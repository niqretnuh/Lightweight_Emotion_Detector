# Lightweight Real-Time Emotion Detector  

We present a highly efficient facial emotion recognition system—under 1 MB in size and capable of sub-200 ms per-frame inference on CPU—built using knowledge distillation and dynamic quantization. We distill large “teacher” networks into tiny student models, prune & quantize them, then deploy in an OpenCV-powered real-time inference pipeline.

---

## Table of Contents  
1. [Introduction](#introduction)  
2. [Design Overview](#design-overview)  
3. [Data Selection & Augmentation](#data-selection--augmentation)  
4. [Model Ablation & Selection](#model-ablation--selection)  
5. [Student Architectures](#student-architectures)  
6. [Compression Techniques](#compression-techniques)  
   - Knowledge Distillation  
   - Structured Pruning  
   - Dynamic Quantization  
7. [Real-Time Inference Pipeline](#real-time-inference-pipeline)  
8. [Performance & Benchmarks](#performance--benchmarks)  
9. [Installation & Usage](#installation--usage)  
10. [Models Directory](#models-directory)  

---

## Introduction

The rapid advancement of deep learning—through overparameterized non-linear systems like Convolutional Neural Networks (CNNs), Residual Networks (ResNets), and Vision Transformers—has dramatically improved accuracy on computer vision tasks, including facial emotion recognition (FER). However, these gains often come with prohibitive computational and memory requirements, limiting deployment on resource-constrained or real-time platforms. In this work, we tackle this bottleneck by combining knowledge distillation and dynamic quantization to compress state-of-the-art “teacher” models into ultra-compact “student” networks (≈0.2 M parameters). After pruning and quantizing the distilled students, we integrate them into an OpenCV-powered real-time inference pipeline that achieves sub-200 ms latency per frame on a CPU. Our contributions are threefold:
1. **Model Ablation on RAF-DB**: A systematic evaluation of modern FER architectures under clean and perturbed conditions.  
2. **Compression Techniques**: An assessment of knowledge distillation, structured pruning, and dynamic quantization in reducing model size and inference time while preserving accuracy.  
3. **Real-Time Deployment**: The design and implementation of a lightweight, modular pipeline for live emotion detection with minimal resource usage.  

For full experimental details, architecture diagrams, and quantitative benchmarks, please consult the [Lightweight_Emotion_Detection.pdf](Lightweight_Emotion_Detection.pdf).

---

## Design Overview  
Our end-to-end pipeline consists of three major stages (see Figures A–C in the report):

1. **Data Selection & Preprocessing**  
2. **Model Ablation, Compression & Selection**  
3. **Real-Time Inference Pipeline**  

Each stage is modular, reproducible, and optimized for speed and robustness.

---

## Data Selection & Augmentation  
- **Dataset**: Real-world Affective Faces Database (RAF-DB), ~15 K images across 7 basic emotions  
- **Perturbation Strategies**:  
  - *Geometric*: random horizontal flips, small-angle rotations  
  - *Noise*: additive Gaussian noise (σ configurable), random distortions  
- **Rationale**: simulates real-world camera and sensor artifacts, ensures model robustness under low-quality or adversarial conditions  

---

## Model Ablation & Selection  
1. **Baseline CNN**: five convolutional layers + FC, with max-pooling, batch-normalization, and dropout  
2. **Teacher Models Evaluated**:  
   - ResNet-18 (~11 M params)  
   - ResEmoteNet (~80 M params)  
   - DeiT-Tiny (~5.7 M params)  
3. **Selection Criteria**:  
   - Validation accuracy on clean & perturbed test sets  
   - Inference speed (CPU & GPU)  
   - Parameter count & storage size  
4. **Outcome**: DeiT-Tiny chosen as primary teacher for its best trade-off between accuracy and size  

---

## Student Architectures  
Two compact student networks trained via knowledge distillation:

1. **ResNet-20** (~0.27 M parameters)  
   - 3 groups of BasicBlocks: 16→32→64 channels  

2. **Tiny DeiT** (~0.20 M parameters)  
   - Single-head attention removed, FFN hidden size 192  
   - Patch embedding + positional encoding + classification head  

**Distillation Loss**:  
\[
\mathcal{L}_{\text{total}} = (1-\alpha)\,\mathcal{L}_{\text{CE}} + \alpha\,\tau^2\,\mathcal{L}_{\text{KL}}
\]

---

## Compression Techniques  

### Knowledge Distillation  
- Transfer soft‐label knowledge from teacher logits to student  
- Temperature (τ) softens distributions for stable learning  
- Balance (α) trades off CE and KL terms  

### Structured Pruning  
- Remove low-magnitude filters/channels via global thresholding  
- Reduces FLOPs and memory while preserving structure  

### Dynamic Quantization  
- `torch.quantization.quantize_dynamic` on `nn.Linear` & `nn.Conv2d`  
- `qint8` weights, QNNPACK backend for CPU  
- Maintains accuracy with minimal latency impact  

---

## Real-Time Inference Pipeline  
1. **Face Detection**  
   - OpenCV Haar Cascade (frontal face) for efficiency  
2. **Preprocessing**  
   - ImageNet normalization
   - Geometric + Noise Perturbation (See section III.A of the pdf)
3. **Inference**  
   - Quantized student model on CPU  
   - Softmax over 7 emotion classes → (label, confidence)  
4. **Visualization**  
   - Bounding boxes + emotion text overlay on live frames  
   - Average latency < 80 ms  

---

## Performance & Benchmarks  
- **Accuracy**: 85 % + accuracy
- **Model Size**: ~0.8 MB after pruning & quantization  
- **Inference Latency**: ~50 ms/frame (laptop CPU)  
- **Comparison**:  
  - Teacher DeiT-Tiny: ~400 ms/frame, 5.7 M params  
  - Student Tiny DeiT: ~50 ms/frame, 0.2 M params  

See detailed tables & plots in the PDF report.

---

## Installation & Usage  
```bash
git clone https://github.com/<username>/lightweight-emotion-detector.git
cd lightweight-emotion-detector
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python pipeline.py
```
- Press **ESC** to exit the webcam window
- Modify `pipeline.py` if using a different camera index or model path

### Models Directory
```bash
models/
├── model.pth               # Final compressed student (CPU-compatible)
├── quantized_deit_full.pth # Dynamic quantized DeiT for GPU backend
└── quantized_deit_0.2M.pth # Tiny quantized DeiT (~200 K parameters)
```
Loadable via EmotionModel() in pipeline.py.