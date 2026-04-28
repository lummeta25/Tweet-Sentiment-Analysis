# Tweet Sentiment Analysis

Binary tweet sentiment classifier (positive/negative) using the NLTK Twitter Samples corpus. Tests 7 ML models across two vectorization strategies. Best result: **73.95% accuracy** with Multinomial Naïve Bayes.

---

## Dataset

- Source: NLTK `twitter_samples` corpus
- 5,000 positive tweets and 5,000 negative tweets
- After filtering short tweets (under 10 words): ~3,770 positive / ~3,526 negative samples

---

## Preprocessing Pipeline

1. **Username removal** — strips the leading `@mention` from each tweet
2. **Lowercasing** — normalizes text casing
3. **Tokenization** — splits into word tokens using `RegexpTokenizer`
4. **Stop word removal** — filters common English stop words via NLTK
5. **Stemming** — reduces words to root form using `PorterStemmer`
6. **Short tweet filtering** — removes tweets under 10 words to reduce noise and ambiguity

After cleaning, labels are assigned (1 = positive, 0 = negative) and the full dataset is **shuffled** before splitting. This is important because NLTK provides the data in blocks — training on all positives then all negatives leads to poor generalization.

**Example:**
```
Before: @TomParker oh wow!! That is beautiful tom :)
After:  oh wow beauti tom
```

---

## Feature Extraction

Two strategies were tested:

- **CountVectorizer** with `ngram_range=(1,2)` — raw word/bigram counts
- **TF-IDF** with `ngram_range=(1,2)` — weights terms by frequency vs. uniqueness across the corpus

---

## Models & Results

### CountVectorizer

| Model | Accuracy |
|---|---|
| Multinomial Naïve Bayes | **73.95%** |
| Neural Network (with dropout) | 73.60% |
| Logistic Regression (`liblinear`) | 72.55% |
| Logistic Regression (`saga`, L1) | 72.50% |
| SVM (linear kernel) | 71.45% |
| Random Forest | 70.45% |

### TF-IDF

| Model | Accuracy |
|---|---|
| Multinomial Naïve Bayes | 73.70% |
| SVM (linear kernel) | 73.30% |
| Logistic Regression (`liblinear`) | 73.05% |
| Logistic Regression (`saga`, L1) | 71.15% |

Switching to TF-IDF produced marginal improvements for some models but no significant change overall.

---

## Neural Network

Architecture:

```
Dense(8, relu) → Dropout(0.5) → Dense(4, relu) → Dropout(0.5) → Dense(1, sigmoid)
```

- Optimizer: Adam (`lr=0.001`)
- Loss: Binary Crossentropy
- Epochs: 8, Batch size: 16

Training showed clear **overfitting** from the first epoch:

```
Epoch 1:  train_acc=0.61  val_acc=0.72
Epoch 3:  train_acc=0.96  val_acc=0.74
Epoch 10: train_acc=0.98  val_acc=0.73
```

Training accuracy climbed to ~98% while validation accuracy stagnated around 73%, with validation loss steadily increasing. Dropout layers helped slightly but didn't resolve the issue. The likely causes are the small dataset size and the high proportion of short, low-context tweets that are easy to memorize but hard to generalize from.

---

## Key Takeaway

Multinomial Naïve Bayes outperformed all other models including SVM and the neural network. This is a well-known pattern with short-text classification — bag-of-words assumptions hold well for tweets, and simpler models generalize better when the dataset is small. More complex models either overfit or don't gain enough signal from the sparse feature space to justify their complexity.

---

## Dependencies

```bash
pip install nltk scikit-learn tensorflow matplotlib seaborn numpy
```

```python
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
```
