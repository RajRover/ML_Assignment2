# ML_Assignment2
# P23 – IMDB Sentiment Analysis: BiLSTM vs TF-IDF

## Overview

This project implements and compares two approaches to binary sentiment classification using the IMDB Movie Reviews Dataset.

The objective is to classify movie reviews as **Positive** or **Negative** and evaluate the performance of traditional Machine Learning against Deep Learning techniques.

Two models are implemented:

1. **TF-IDF + Logistic Regression** – A traditional Machine Learning approach using statistical text representation.
2. **BiLSTM (Bidirectional Long Short-Term Memory)** – A Deep Learning approach that learns sequential and contextual relationships in text.

## Dataset

**IMDB Movie Reviews Dataset**

- Total reviews: 50,000
- Training samples: 25,000
- Testing samples: 25,000
- Positive reviews: 25,000
- Negative reviews: 25,000

Dataset: [Hugging Face – IMDB](https://huggingface.co/datasets/stanfordnlp/imdb)

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TensorFlow / Keras
- Hugging Face Datasets
- Google Colab

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Dataset inspection and statistical analysis.
- Sentiment distribution visualization.
- Review length analysis.
- Examination of positive and negative reviews.

### 2. Text Preprocessing

- HTML tag removal.
- Lowercase conversion.
- Special character handling.
- Text cleaning.
- Preservation of important sentiment words and negations.

### 3. TF-IDF + Logistic Regression

- TF-IDF feature extraction.
- Unigram and bigram representation.
- Vocabulary size of up to 20,000 features.
- Logistic Regression classification.
- Model evaluation using standard classification metrics.

### 4. BiLSTM Deep Learning Model

The BiLSTM architecture consists of:

| Layer | Configuration |
|---|---|
| Embedding | 128 dimensions |
| Bidirectional LSTM | 64 units per direction |
| Dropout | 0.5 |
| Dense | 1 neuron |
| Activation | Sigmoid |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Batch Size | 64 |
| Maximum Epochs | 5 |

Additional techniques:
- Tokenization and integer encoding.
- Sequence padding and truncation.
- Early stopping.
- Training and validation monitoring.

### 5. Model Evaluation and Comparison

Both models are evaluated using the same independent test dataset.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training Time

Performance comparison is visualized using comparative graphs.

## Model Comparison

| Aspect | TF-IDF + Logistic Regression | BiLSTM |
|---|---|---|
| Approach | Traditional ML | Deep Learning |
| Text Representation | TF-IDF | Word Embeddings |
| Sequence Learning | N-gram features | Bidirectional LSTM |
| Feature Learning | Statistical | Learned representations |
| Computational Cost | Generally lower | Generally higher |

Actual experimental results are presented in the notebook.

## Project Structure

```text
P23-IMDB-Sentiment-Analysis/
│
├── IMDB_Sentiment_Analysis.ipynb
├── README.md
└── requirements.txt
```

The trained BiLSTM model can optionally be saved as `bilstm_sentiment_model.keras`.

## Installation and Execution

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd P23-IMDB-Sentiment-Analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `IMDB_Sentiment_Analysis.ipynb` in Google Colab or Jupyter Notebook.

Execute the notebook cells sequentially to reproduce the experiments.

## Requirements

```text
datasets
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
beautifulsoup4
joblib
```

## Key Learning Outcomes

- Understanding text preprocessing for NLP tasks.
- Learning statistical text representation using TF-IDF.
- Implementing traditional Machine Learning classifiers.
- Understanding tokenization, embeddings, RNNs and LSTMs.
- Building a Bidirectional LSTM architecture.
- Evaluating and comparing traditional ML and Deep Learning models.

## Conclusion

This project explores two different approaches to sentiment analysis and highlights the differences between statistical feature-based learning and sequential deep learning.

The comparative evaluation provides insights into model performance, computational efficiency, and the effectiveness of different text representation techniques for sentiment classification.

---

**Project:** P23 – IMDB Sentiment Analysis: BiLSTM vs TF-IDF  
**Domain:** Natural Language Processing (NLP)  
**Task:** Binary Sentiment Classification
