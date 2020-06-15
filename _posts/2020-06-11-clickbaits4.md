# Detecting Clickbaits (4) - A Practitioner Strategy!

![image info](images/p4-header.jpg "by Alexandru Acea")


**Problem**.
Given a set of headlines and labels, whether that headline is a clickbait or 
not, you're asked to build a model to detect clickbait headlines.

**Approach**.

Based on my experience in different industries, as an AI/ML Practitioner/Applied Scientist,
 generally, I prioritize the techniques as below:
 
1. Conventional (Random Forst, Logistic Regreesion, XGBoost, SVM, etc.) 
- At this phase: find potential issues e.g. lack of enough 
training data, bias, skewness, noise, etc. Also apply simple enhancements 
such as text normalizations, text fixing, using different types of classifiers, 
manual boosting, TTA, etc.
- [Detecting Clickbaits (2) - Logistic Regression](https://hminooei.github.io/2020/04/21/clickbaits2.html) and
- [Detecting Clickbaits (3) - Manual Boosting](https://hminooei.github.io/2020/05/10/clickbaits3.html)
- and if needed, move to Transfer Learning (TL).
- This has super fast train and dev cycles (`1.3s` for training a model for this 
problem).
- Better model explainability and hence easier to detect issues.
2. Transfer Learning 
- Again, try find potential issues with the data, fixing text encodings, etc.
- [Detecting Clickbaits (1) - Universal-Sentence-Encoder Transfer Learning](https://hminooei.github.io/2020/04/14/clickbaits1.html)
- and if needed, move to deep learning.
- This has much longer cycles compared to conventional methods. In this case, it took 45mins to 
train the network for 2 epochs.
- It can benefit from the pre-trained language models such as BERT, USE, ELMo, etc.
- Overall it has much more cost of training/tuning and maintenance.
3. Deep Learning (CNN, LSTM, Transformers, etc.)
- The most expensive approach.
- Usually takes many more cycles to find the right structure and parameters and tune them 
compared to TL.


**NOTE**.
- This is recommended for text classification, for other types of 
unstructured data you might want to skip the conventional algorithms and 
start from transfer learning. 
- There is always exceptional situations.
