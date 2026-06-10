# BBC News Classifier — Classical vs Modern NLP

A text classification project that reads a news headline and automatically 
predicts its category — sport, politics, tech, business, or entertainment.

## Approaches

### Classical (TF-IDF + Naive Bayes)
- Converted raw text into numerical vectors using TF-IDF
- Trained a Naive Bayes classifier on 1780 BBC news articles
- Achieved 95.3% accuracy on unseen test data

### Modern (HuggingFace Zero-Shot Classification)
- Used a pretrained transformer model via HuggingFace pipelines
- No training data needed — pass any categories on the fly
- Flexible but trades some accuracy for flexibility

## Dataset
BBC News dataset — 2225 articles across 5 categories

## Tech Stack
- Python
- scikit-learn
- HuggingFace Transformers
- Pandas

## Key Learning
Classical ML needs labeled data but gives high accuracy.
Modern zero-shot needs no training but works with any categories.
Fine-tuned transformers (coming next) give the best of both worlds.
