# Emoji Prediction from Tweet Text
Datathon project, predicting which emoji a tweet was originally posted with, using classical NLP and machine learning classifiers.

# Overview
Given a tweet with its emoji removed, can a model predict which emoji the author used? This project explores that question by training and comparing three classical machine learning classifiers on a labeled dataset of over 225,000 tweets across 10 emoji classes.

Each model is implemented as a self-contained Google Colab notebook that anyone can open and run without any local setup.

# The Problem
The dataset consists of two files:

1. tweets.txt — one tweet per line, with the emoji stripped out  
2. emoji.txt — the name of the emoji that appeared in the corresponding tweet

The 10 emoji classes are: `blush`, `flushed`, `grin`, `heart_eyes`, `relaxed`, `smirk`, `sob`, `weary`, `wink`, `yum`.The task is a multi-class text classification problem: given the text of a tweet, predict which of the 10 emoji it originally contained.

# Our Approach: Shared preprocessing
All three models share the same preprocessing pipeline and train/test split, ensuring fair comparison. The shared steps are:

1. Clean tweets — lowercase, strip URLs, mentions, punctuation
2. TF-IDF vectorization — convert tweet text into numerical feature vectors using term frequency–inverse document frequency
3. Train/test split — 80% training (180,264 tweets), 20% test (45,067 tweets)
 
The preprocessed split is saved as data_split.pkl and loaded by each notebook, so every model trains and evaluates on identical data.

# The Models

**Model 1 - Logistic Regression**  
Logistic Regression is the most interpretable of the three models. It learns a weight for every word in the TF-IDF vocabulary and uses those weights to assign a probability to each emoji class, predicting the class with the highest probability.

**Model 2 - Naive Bayes**  
Naive Bayes is a probabilistic classifier rooted in Bayes' theorem. Given that this tweet contains these words, the theorem asks what the probability it belongs to each emoji class is. The "naive" assumption is that each word contributes to the prediction independently of the others. While this simplification does not hold in reality, it worked surprisingly well in practice for text.

**Model 3 - Linear SVC**  
Support Vector Machines find the decision boundary that maximises the margin between classes. The Linear SVC variant is optimised for high-dimensional sparse data, exactly the kind that TF-IDF produces, and is typically the strongest performer on text classification tasks.

# How to Run
Each notebook is fully self-contained and runs on Google Colab with no local installation required.
1. Download each model's Notebook file
2. Open Google CoLab
3. Click File -> Upload Notebook -> Upload
4. Upload the Notebook
5. Upload data_split.pkl to your Colab session (or mount Google Drive if you have it stored there)
6. Run all cells from top to bottom. Each notebook will train the model, evaluate it, and produce the classification report, bar chart, and confusion matrix.

# Results Summary

| **Model**            | **Accuracy** |
| :------------------: | :----------: |
| Logistic Regression  | 0.52         | 
| Naive Bayes          | 0.51         |
| Linear SVC           | 0.44         |

# Classification Reports

Logistic Inference:  
<img width="575" height="356" alt="image" src="https://github.com/user-attachments/assets/b2787b7b-7d66-4c2e-b023-a9bb0bea4a13" />

Naive Bayes:  
<img width="668" height="442" alt="image" src="https://github.com/user-attachments/assets/0abd21c1-9e00-446b-8dbf-813760b564d1" />

Linear SVC:  
<img width="956" height="596" alt="image" src="https://github.com/user-attachments/assets/1917b8ac-0f63-46b0-a940-8225baff1e71" />


# Key Findings 
The models excel at predicting emotionally distinct emojis, like `heart_eyes` and `sob`. Tweets containing these emoji tend to use strong, unambiguous language, such as "I'm obsessed" or "I'm literally crying", that TF-IDF can reliably detect. Our model struggled with emojis that have similar emotional tones to one another. The `sob` and `weary` tends to be the largest source of confusion since both appear in distress-related tweets and their surrounding text is nearly identical. Similarly, `wink` and `smirk` are hard to distinguish because both appear in playful, ironic contexts.

Our project also struggled with class imbalance, as sob has 9,989 test examples whereas flushed only has 2,067. Even with `class_weight='balanced`, the model is pulled toward predicting majority classes.
