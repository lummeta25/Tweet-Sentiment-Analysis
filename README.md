# Tweet Sentiment Analysis

Binary tweet sentiment classifier (positive/negative) using the NLTK Twitter Samples corpus. Benchmarks 7 ML models across CountVectorizer and TF-IDF features. Best result: **73.95% accuracy** with Multinomial Naïve Bayes.

## Preprocessing

Each tweet is cleaned through username removal, lowercasing, tokenization, stop word filtering, and Porter stemming. Tweets under 10 words are dropped to reduce noise. The dataset is shuffled after labeling to ensure even distribution across train/test splits.

## Models & Results

| Model | Accuracy |
|---|---|
| Multinomial Naïve Bayes | **73.95%** |
| Neural Network (dropout) | 73.60% |
| Logistic Regression (`liblinear`) | 72.55% |
| Logistic Regression (`saga`, L1) | 72.50% |
| SVM (linear kernel) | 71.45% |
| Random Forest | 70.45% |

TF-IDF was also tested across all models with similar results.

The neural network (Dense → Dropout → Dense → Dropout → Sigmoid) showed clear overfitting — training accuracy hit ~98% while validation plateaued around 73%. Naïve Bayes generalised best given the small dataset and short text nature of tweets.

## Dependencies

```bash
pip install nltk scikit-learn tensorflow matplotlib seaborn numpy
```

```python
nltk.download('twitter_samples')
nltk.download('stopwords')
```
