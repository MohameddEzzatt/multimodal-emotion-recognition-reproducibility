# Multimodal Emotion Recognition — Reproducibility Package

## Overview

This repository provides a minimal reproducibility package accompanying our study on speaker-independent multimodal emotion recognition using the RAVDESS dataset.

The proposed framework combines audio and visual modalities through multiple fusion strategies and evaluates performance under a strict Leave-One-Speaker-Out (LOSO) protocol to assess speaker-independent generalization.

The repository includes:

- Extracted audio features
- Extracted visual features
- Predefined LOSO train/test splits
- Trained classification models
- Reproducibility resources for the experimental pipeline

The proposed framework follows a modular pipeline consisting of:

1. Audio extraction from video recordings
2. Video frame extraction
3. Audio feature extraction
4. Visual feature extraction
5. Feature preprocessing and normalization
6. Multi-view fusion
7. Speaker-independent LOSO evaluation
8. Classification using a stacking ensemble framework

---

# Dataset

The experiments were conducted using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

## Dataset Characteristics

- 24 professional actors
- Audio-visual emotional speech recordings
- Multiple emotion categories
- Speaker-independent evaluation protocol

The dataset is publicly available from its official source and is not redistributed within this repository.

---

# Audio Extraction

Audio signals were extracted directly from the original RAVDESS video recordings prior to feature extraction. Each video file was processed individually to isolate the speech signal while preserving the original emotional content.

## Audio Processing Procedure

The audio extraction pipeline includes:

- Loading video recordings from the RAVDESS dataset
- Extracting the corresponding speech waveform from each video
- Converting audio streams into a standardized processing format
- Organizing extracted audio files using their associated video identifiers

The extracted audio files were subsequently used for acoustic feature computation and multimodal fusion.

---

# Video Frame Extraction

Visual information was extracted from the video recordings through frame-level sampling. Frames were uniformly sampled from each video sequence to capture representative facial expressions while reducing computational overhead.

## Frame Extraction Procedure

The visual preprocessing pipeline includes:

- Loading video recordings frame-by-frame
- Uniform temporal frame sampling
- Face-centered visual processing
- Frame resizing and preprocessing for deep visual feature extraction

Frames were sampled at approximately 3 frames per second (3 FPS), providing a balance between temporal coverage and computational efficiency.

The extracted frames were subsequently processed using a deep visual feature extractor based on a pre-trained VGGFace-ResNet50 architecture.

---

# Audio Feature Extraction

Audio features were extracted using a hybrid acoustic representation designed to capture complementary spectral, temporal, and harmonic characteristics relevant to emotion recognition.

## Extracted Audio Descriptors

The extracted feature set includes:

- Mel-Frequency Cepstral Coefficients (MFCCs)
- Zero-crossing rate (ZCR)
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Root Mean Square Energy (RMS)

These descriptors were computed from each speech signal and subsequently aggregated into a unified audio feature representation.

## Audio Processing Pipeline

The audio preprocessing pipeline includes:

- Audio loading and preprocessing
- Feature extraction from multiple acoustic descriptors
- Temporal aggregation of frame-level features
- Feature normalization and scaling

Each audio sample contains:

- `video_id`
- `file_name`
- `label`
- Extracted audio feature vector

The extracted audio features are provided in:

```text
features/Audio_Features_RAVDESS.csv
```

---

# Visual Feature Extraction

Visual features were extracted using a deep convolutional neural network based on the VGGFace-ResNet50 architecture. The model was employed as a high-level facial representation extractor to capture discriminative facial expression patterns associated with emotional states.

## Visual Processing Pipeline

The visual feature extraction pipeline includes:

- Frame loading and preprocessing
- Deep facial feature extraction using a pre-trained VGGFace-ResNet50 backbone
- High-level embedding generation from the penultimate network representation
- Feature aggregation across sampled frames

Each sampled frame was passed through the pre-trained network to obtain a deep visual embedding. The extracted frame-level representations were subsequently aggregated to form a compact video-level representation for each sample.

The resulting visual feature vectors originally consisted of 2048-dimensional embeddings.

## Visual Feature File

Each visual sample contains:

- `video_id`
- Extracted visual feature vector

The visual feature file is provided in compressed format due to file size considerations. The compressed feature archive is available as:

```text
Video_Features_RAVDESS.zip
```

---

# Feature Preprocessing and Normalization

Prior to fusion and classification, the extracted multimodal features underwent several preprocessing operations to improve numerical stability and reduce redundancy.

## Audio Feature Scaling

Audio features were standardized using z-score normalization through the `StandardScaler` implementation from Scikit-learn.

For each LOSO iteration:

- Scaling parameters were estimated exclusively from the training speakers
- The fitted scaler was subsequently applied to the held-out test speaker

This protocol prevents information leakage between training and evaluation data.

## Visual Feature Normalization

Visual features were processed using PCA-based dimensionality reduction before fusion.

The PCA transformation was similarly fitted only on the training partition during each LOSO iteration and then applied to the corresponding test partition.

## Label Encoding

Emotion labels were converted into numerical representations using `LabelEncoder` prior to classifier training.

This preprocessing stage ensured compatibility with machine learning classifiers and maintained consistent label mappings across all LOSO folds.

---

# Multi-View Fusion

The proposed framework employs a multi-view fusion strategy designed to capture complementary interactions between audio and visual modalities through multiple independent fusion mechanisms.

The fusion architecture combines four distinct fusion approaches:

1. Simple Concatenation Fusion
2. Cross-Modal Attention Fusion
3. Gated Fusion
4. Multiplicative Fusion

The outputs from all fusion branches are subsequently concatenated into a unified multimodal representation used for classification.

---

## 1. Simple Concatenation Fusion

Concatenation fusion represents the most direct early-fusion strategy.

The normalized audio feature vector and the PCA-reduced visual feature vector are concatenated into a single multimodal representation:

```math
F_concat = [V_PCA ; A_scaled]
```

This method preserves the complete information from both modalities without explicitly modeling cross-modal interactions.

---

## 2. Cross-Modal Attention Fusion

Cross-modal attention fusion was designed to model bidirectional dependencies between audio and visual representations.

### Attention Pipeline

The attention fusion process includes:

- Projection of audio and visual features into a shared latent space
- Bidirectional cross-attention computation
- Concatenation of attended representations

Both modalities were projected into a shared latent representation of:

```text
128 dimensions
```

using Dense layers with ReLU activation.

Subsequently:

- Audio features were used as query vectors
- Visual features were used as key and value vectors

The process was then repeated symmetrically with reversed modality roles to capture bidirectional cross-modal interactions.

The resulting attention-enhanced representations were concatenated to form the final attention-based fusion representation.

The projection and attention parameters remained fixed during feature transformation.

---

## 3. Gated Fusion

Gated fusion introduces a modality-weighting mechanism to regulate the contribution of audio and visual representations.

### Gating Procedure

The gated fusion pipeline includes:

- Sigmoid-based gating value generation
- Softmax normalization of modality weights
- Weighted feature modulation
- Feature concatenation

For each modality:

- A Dense layer with sigmoid activation generated a gating value
- The resulting gating values were normalized through a Softmax operation

This process produced two modality weights:

```math
\alpha + \beta = 1
```

where:

- `α` corresponds to the audio modality weight
- `β` corresponds to the visual modality weight

The original feature vectors were then scaled by their corresponding modality weights before concatenation:

```math
F_gated = [\alpha \cdot A_{scaled} ; \beta \cdot V_{PCA}]
```

The gating parameters remained fixed during feature processing.

---

## 4. Multiplicative Fusion

Multiplicative fusion was employed to capture direct element-wise interactions between audio and visual representations.

### Multiplicative Fusion Pipeline

The multiplicative fusion process includes:

- Projection of audio and visual features into a common latent space
- Element-wise multiplication between projected representations
- Dropout regularization
- Nonlinear activation

Both modalities were projected into a shared latent space of:

```text
256 dimensions
```

using Dense layers.

An element-wise multiplication operation was then applied between the projected feature vectors to model direct cross-modal interactions.

Unlike classical tensor fusion approaches based on outer-product operations, this formulation employs dimension-wise multiplicative interactions with substantially lower computational complexity.

To improve numerical stability and reduce overfitting:

- Dropout regularization with rate `0.3` was applied
- ReLU activation was subsequently used

The projection parameters remained fixed during feature transformation.

---

# Final Feature Aggregation

The outputs from all fusion branches were concatenated to form the final multimodal representation:

```math
F_{final} =
[F_{concat} ;
F_{attention} ;
F_{gated} ;
F_{multiplicative}]
```

The resulting fused feature representation was subsequently used for speaker-independent classification under the LOSO evaluation protocol and provided in:

```text
features/Fused_features.csv
```
---

# Speaker-Independent LOSO Evaluation

The proposed framework was evaluated using a strict Leave-One-Speaker-Out (LOSO) cross-validation protocol to assess speaker-independent generalization.

## LOSO Protocol

For each LOSO iteration:

- One speaker was selected as the held-out test subject
- All remaining speakers were used for training

This process was repeated for all speakers in the dataset.

The LOSO protocol ensures that:

- No speaker overlap exists between training and testing sets
- Evaluation reflects realistic speaker-independent performance
- Models are required to generalize across unseen identities

The predefined LOSO train/test splits used in the experiments are provided in:

```text
splits/
```

Each split file contains the row indices corresponding to the training or testing samples for a given LOSO fold.

---

# Training and Classification

The final fused multimodal representations were evaluated using a stacking-based ensemble classification framework.

## Base Classifiers

Three machine learning classifiers were employed as base learners:

### Support Vector Machine (SVM)

```text
Kernel: RBF
C = 10
Gamma = auto
Probability estimation enabled
```

### Random Forest

```text
Number of trees = 300
Random state = 42
```

### XGBoost

```text
Number of estimators = 150
Maximum depth = 4
Learning rate = 0.1
Subsample ratio = 0.7
Column sample ratio = 0.7
Tree method = hist
Random state = 42
```

---

## Stacking Ensemble

The outputs from the base classifiers were combined using a stacking ensemble framework.

### Stacking Configuration

- Base learners:
  - SVM
  - Random Forest
  - XGBoost

- Meta-classifier:
  - Linear SVM

- Stacking method:
  - Probability-based stacking (`predict_proba`)

The stacking classifier was implemented using the `StackingClassifier` module from Scikit-learn.

---

# Feature Scaling During LOSO

Within each LOSO iteration:

1. The training partition was isolated
2. `StandardScaler` was fitted exclusively on training samples
3. The fitted scaler was subsequently applied to the held-out test speaker

This protocol prevents information leakage between train and test partitions.

---

# Trained Models

The repository includes the trained machine learning models used in the final experiments.

Available models include:

```text
trained_models/
├── svm_model.pkl
├── random_forest_model.pkl
├── xgboost_model.pkl
└── stacking_model.pkl
```

These models correspond to the trained classifiers used during the speaker-independent LOSO evaluation experiments.

---

# Reproducibility Notes

This repository was designed to facilitate reproducibility of the reported experiments.

To reproduce the evaluation procedure:

1. Load the provided feature files
2. Use the predefined LOSO splits
3. Apply the corresponding preprocessing pipeline
4. Train or load the provided classifiers
5. Evaluate performance using the LOSO protocol

All preprocessing, fusion, and evaluation procedures were implemented under fixed random seeds (42) to ensure deterministic and reproducible behavior across runs.

---

# Citation

If you use this repository or reproduce the experiments presented in this work, please cite the associated manuscript.

---
