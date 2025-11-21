# DA5401 End-Semester Data Challenge: Metric Learning for AI Evaluation System

🎯 **Objective**: Build a metric learning model to predict the "fitness" or similarity score between AI evaluation metrics and prompt-response pairs.

## 📋 Project Overview

This project tackles a challenging machine learning problem involving metric learning for AI evaluation systems. The goal is to predict similarity scores between AI evaluation metric definitions (provided as embeddings) and corresponding prompt-response text pairs across multiple languages.

### Student Information
- **Student Name**: Shivam Tiwari
- **Roll No**: DA25C019

## 🎯 Problem Statement

### Inputs
- **Metric Definition**: Text embeddings from Gemma model (768-dimensional)
- **Prompt-Response Pair**: (system_prompt, user_prompt, response) in text format

### Output
- **Fitness Score**: Continuous value in range [0, 10]

### Evaluation Metric
- **Root Mean Squared Error (RMSE)** – Lower is better

### Key Challenges
1. **Multilingual data** – Tamil, Hindi, Assamese, Bengali, Bodo, Sindhi, English
2. **Highly imbalanced** – 92%+ scores are ≥ 8
3. **Distribution shift** – Test may differ from training
4. **Limited information** – Metric definitions only as embeddings

## 📊 Dataset Statistics

- **Training samples**: 4,999
- **Test samples**: 30,001
- **Unique metrics**: 10
- **Metric embedding shape**: (10, 768)

### Score Distribution
- **Score 9.0**: 3,123 samples (62.5%)
- **Score 10.0**: 1,442 samples (28.8%)
- **Score 8.0**: 259 samples (5.2%)
- **Scores ≤ 7.0**: 175 samples (3.5%)

This extreme imbalance poses significant modeling challenges, requiring specialized techniques to handle minority classes.

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Comprehensive analysis of score distributions
- Text length analysis across different languages
- Metric frequency analysis
- Correlation studies between features and target

### 2. Feature Engineering

#### Text Processing
- **Gemma Embeddings**: Generated 768-dimensional embeddings for concatenated text (system_prompt + user_prompt + response)
- **TF-IDF Features**: Traditional text vectorization as baseline comparison

#### Feature Combination
- **Metric Embeddings**: 768 dimensions
- **Context Embeddings**: 768 dimensions  
- **Cosine Similarity**: 1 dimension (key engineered feature)
- **Total Features**: 1,537 dimensions

### 3. Model Architecture

#### Final Model: Multi-Layer Perceptron (MLP) with Gemma Embeddings

**Pipeline Flow:**
1. **Preprocessing**:
   - **StandardScaler**: Normalizes features to zero mean and unit variance
   - **PCA**: Reduces dimensionality from 1,537 to 256 components

2. **MLP Regressor**:
   - **Hidden Layers**: 256 → 128 (Tanh activation)
   - **Output**: 1 neuron (Linear)
   - **Optimizer**: Adam (learning rate = 0.003)

3. **Post-Processing**:
   - Convert log-transformed prediction back using `expm1()` to obtain final score

### 4. Model Comparison

| Model Architecture | Features | CV RMSE | Status |
|-------------------|----------|---------|---------|
| **MLP + PCA** | **Gemma Embeddings** | **1.844** | **Selected** |
| Stacking (HGB+RF) | Gemma Embeddings | 3.200 | Discarded |
| HistGradientBoosting | Gemma Embeddings | 3.500 | Discarded |
| Stacking (HGB+RF) | TF-IDF | 3.630 | Discarded |
| HistGradientBoosting | TF-IDF | 3.704 | Baseline |

## 📈 Results

### Performance Metrics
- **Cross-Validation RMSE**: 1.844 ± 0.362
- **Leaderboard Score**: 2.637
- **Improvement over Baseline**: 28.8%

### Improvement Journey
1. **TF-IDF Baseline**: 3.704 RMSE
2. **Stacking**: 3.630 RMSE (2% improvement)
3. **Gemma Embeddings**: ~3.2 RMSE (12% improvement)
4. **MLP + Cosine Similarity**: 1.844 CV RMSE (50% improvement)

## 🔍 Key Insights

### What Worked
1. **Semantic Embeddings > TF-IDF**: Multilingual, context-heavy data demands embeddings, not frequency counts
2. **Feature Engineering**: Adding Cosine Similarity boosted performance more than complex architectures
3. **Simplicity Wins**: A single MLP outperformed stacking ensembles once high-quality embeddings were used

### Challenges
1. **Generalization Gap**: Notable gap between CV (1.844) and LB (2.637) suggests distribution shift
2. **Class Imbalance**: 92%+ of samples have scores ≥ 8, making it difficult to predict lower scores
3. **Multilingual Complexity**: Handling 7 different languages requires robust embedding strategies

## 📁 Project Structure

```
KaggleCompetitionFinal/
├── Data_Kaggle_challenge.ipynb    # Main analysis notebook
├── train_data.json                # Training dataset
├── test_data.json                 # Test dataset
├── metric_names.json              # Metric name mappings
├── metric_name_embeddings.npy     # Pre-computed metric embeddings
├── submission_new_output_v2.csv   # Final submission file
├── eda_visualizations.png         # EDA plots
└── README.md                      # Project documentation
```

## 🚀 Usage

### Requirements
```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
```

### Running the Analysis
1. Open `Data_Kaggle_challenge.ipynb`
2. Run all cells sequentially
3. The notebook includes:
   - Data loading and preprocessing
   - Comprehensive EDA with visualizations
   - Feature engineering pipeline
   - Model training and validation
   - Performance analysis and insights

## 🔮 Future Directions

1. **Augment Low-Score Data**: Use LLM-based data generation to synthesize examples with scores 0-5
2. **Fine-tune Gemma**: Adapt the embedding model via LoRA (Low-Rank Adaptation) for task-specific tuning
3. **Advanced Architectures**: Experiment with Transformer-based models for better multilingual understanding
4. **Distribution-Aware Validation**: Implement stratified sampling strategies to better handle class imbalance

## 📝 Lessons Learned

1. **Quality over Quantity**: High-quality embeddings (Gemma) significantly outperformed traditional features (TF-IDF)
2. **Feature Engineering Matters**: Simple features like cosine similarity can provide substantial performance gains
3. **Regularization is Key**: PCA and proper regularization prevented overfitting in high-dimensional space
4. **Cross-Validation Strategy**: The high variance in CV scores (±0.362) indicates sensitivity to data splits

## 🏆 Achievements

- Successfully handled multilingual text data across 7 languages
- Achieved 50%+ improvement over baseline with elegant feature engineering
- Demonstrated the power of semantic embeddings for complex NLP tasks
- Created comprehensive analysis with actionable insights for future work

---

*This project was completed as part of the DA5401 End-Semester Data Challenge, showcasing advanced machine learning techniques for multilingual metric learning in AI evaluation systems.*