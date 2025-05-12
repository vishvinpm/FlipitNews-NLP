# 📰 FlipIt News – NLP Business Case Study

This project explores natural language processing techniques to classify news articles from FlipIt News into categories such as Politics, Business, Sports, Entertainment, and Technology. The work demonstrates how NLP and machine learning can enhance content categorization and improve user engagement for digital media platforms.

---

## 🎯 Objective

- Automatically categorize news articles using machine learning and deep learning models.
- Compare performance across traditional ML models (Naive Bayes, Random Forest, etc.) and neural networks (RNN, LSTM, BERT).
- Improve discoverability and user personalization for FlipIt News.

---

## 📦 Dataset Overview

- **Source**: FlipIt News internal news archive
- **Size**: 2,225 articles (after deduplication: 2,126)
- **Categories**:
  - Politics
  - Technology
  - Sports
  - Business
  - Entertainment
- **Features**:
  - `Article`: The news content
  - `Category`: Labeled category

---

## 🧪 Technologies Used

- **Languages**: Python
- **Libraries**:
  - Data: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - NLP: NLTK, TensorFlow, HuggingFace Transformers
  - ML: Scikit-learn
- **Models**:
  - Traditional: Naive Bayes, Decision Tree, Random Forest, KNN
  - Neural Nets: RNN, LSTM
  - Pretrained: BERT + LSTM

---

## 🔍 Workflow

1. **Data Cleaning & Preprocessing**:
   - Removed duplicates
   - Tokenization, stopword removal, lemmatization

2. **Feature Engineering**:
   - CountVectorizer (BoW)
   - TF-IDF Vectorization
   - BERT Embeddings

3. **Modeling**:
   - Classical ML: Naive Bayes (98% accuracy), Random Forest (97%)
   - Neural Networks: RNN, LSTM (moderate performance)
   - BERT + LSTM: Achieved ~96.7% accuracy with strong generalization

4. **Evaluation**:
   - Accuracy, F1 Score, Confusion Matrix
   - Visualizations for each model

---

## 🧠 Key Insights

- **Naive Bayes and Random Forest** with TF-IDF features gave high accuracy with low training time.
- **Deep learning models (RNN, LSTM)** underperformed without pre-trained embeddings.
- **BERT + LSTM** yielded the best results with ~96.7% accuracy, showcasing the power of transfer learning in text classification.

---

