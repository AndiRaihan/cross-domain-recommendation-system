# Cross-Domain Recommendation System

## Objective

This project implements a deep learning-based cross-domain recommendation system, specifically designed to transfer user preferences from a source domain (e.g., Movies) to a target domain (e.g., Music). It aims to solve the cold-start problem where users have little to no interaction history in the target domain.

The system implements and compares three models:

1. **CMF (Cross-Domain Matrix Factorization)**: A linear baseline model.
2. **NCF (Neural Collaborative Filtering)**: A deep learning baseline model (Target-only).
3. **PTUPCDR (Personalized Transfer of User Preferences)**: A state-of-the-art deep learning model that learns a mapping function to transfer user embeddings from source to target.

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Results](#results)

## Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd cross-domain-recommendation-system
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Data

The project uses the [**Amazon Review Data** (2023 version)](https://amazon-reviews-2023.github.io/main.html).

- **Source Domain**: Movies and TV
- **Target Domain**: CDs and Vinyl

The data pipeline handles:

- **Preprocessing**: Filtering users/items, k-core filtering, and splitting into train/validation/test sets.
- **Feature Engineering**: Generating text embeddings for items using `SentenceTransformer`.
- **User Profiling**: Creating user content profiles based on their interaction history.

Data is expected to be in `data/raw/` and processed data is saved to `data/processed/`.

## Usage

The project uses a central CLI entry point `main.py`.

### 1. Data Processing

To run the full data processing pipeline (Preprocessing -> Splitting -> Feature Engineering -> Slicing -> User Profiling):

```bash
python main.py process-data \
    --source "data/raw/Movies_and_TV.jsonl" \
    --target "data/raw/CDs_and_Vinyl.jsonl" \
    --meta-source "data/raw/meta_Movies_and_TV.jsonl" \
    --meta-target "data/raw/meta_CDs_and_Vinyl.jsonl" \
    --output "data/processed"
```

### 2. Train Models

Train the CMF Baseline:

```bash
python main.py train-cmf
```

Train the NCF Baseline:

```bash
python main.py train-ncf
```

Train the PTUPCDR Model (Full Pipeline):

```bash
python main.py train-ptupcdr
```

### 3. Visualization & Analysis

Generate visualizations (Latent Space, Case Studies, Performance Comparison):

```bash
python main.py visualize --type all
```

Available types: `latent`, `case-study`, `ncf-failure`, `comparison`, `all`.

## Model Architecture

### 1. CMF (Cross-Domain Matrix Factorization)

A matrix factorization approach that jointly factorizes the rating matrices of both domains, sharing user embeddings to enable knowledge transfer.

### 2. NCF (Neural Collaborative Filtering)

A neural network-based collaborative filtering model that uses a multi-layer perceptron (MLP) to learn non-linear interactions between users and items. This baseline is trained only on the target domain to demonstrate the cold-start problem.

### 3. PTUPCDR (Personalized Transfer of User Preferences)

A meta-learning based approach. It consists of:

- **Source Encoder**: A LightGCN model trained on the source domain.
- **Target Encoder**: A LightGCN model trained on the target domain.
- **Meta-Network**: A mapping network that learns to transform source user embeddings (and content profiles) into target user embeddings, effectively "generating" a warm-start embedding for cold-start users.

## Evaluation

Models are evaluated using **Leave-One-Out** protocol on the test set.
Metrics include:

- **Recall@10** (Hit Ratio)
- **NDCG@10** (Normalized Discounted Cumulative Gain)
- **MAP** (Mean Average Precision)

## Results

Evaluation results and training history are saved in the `reports/` directory.

- `reports/baseline/`: NCF results.
- `reports/cmf_baseline/`: CMF results.
- `reports/ptupcdr_model/`: PTUPCDR results.
- `reports/figures/`: Generated plots and case studies.
