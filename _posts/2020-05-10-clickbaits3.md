# Detecting Clickbaits (3) - Manual Boosting


![image info](./images/p3-header.jpeg "by Foodiesfeed")

This is the continuation of the previous post [Detecting Clickbaits (2) - Logistic Regression](https://hminooei.github.io/2020/04/21/clickbaits2.html).

**Problem**.
Given a set of headlines and labels, whether that headline is a clickbait or 
not, you're asked to build a model to detect clickbait headlines.

**Solution**.

Read data: ...

Split into train/validation/test sets: ...

And finally, we build and train/tune a pipeline model - `cfr_pipeline` - 
that uses `LogisticRegression` classifier. The model's macro precision on 
test set is `0.9650`.

**Manual Boosting**

First find the mislabeled samples in the training set:
```
def get_incorrect_predictions(trained_model, all_data, text_df, label_series):
    col_name = text_df.columns.values.tolist()[0]
    preds = trained_model.predict(text_df[col_name])
    incorrectly_predicted = text_df.loc[label_series != preds]
    incorrectly_predicted.shape
    res = incorrectly_predicted.merge(all_data, on=col_name, suffixes=("_left", "_right"))
    return res
    
to_be_added = get_incorrect_predictions(cfr_pipeline, df, text_train.to_frame(name="headline"), label_train)
```
There are 432 such samples. Let's prepare and add them to the training set. In other words
let's manually boost our training set:
```
extra_text_train = to_be_added["headline"]
extra_label_train = to_be_added["clickbait"]
extra_label_train = np.array(extra_label_train)

boosted_text_train = pd.concat([text_train, extra_text_train])
boosted_label_train = np.concatenate([label_train, extra_label_train], axis=0)

cfr_pipeline_1x = train_measure_model(boosted_text_train, boosted_label_train, 
                                      text_val, label_val,
                                      cv_binary=True, cv_analyzer="word", cv_ngram=("w", 1, 3), 
                                      cv_max_features=5000, cv_have_tfidf=True, cv_use_idf=True, 
                                      cfr_penalty="l2", cfr_C=1.0, stop_words=None)

measure_model_on_test(cfr_pipeline_1x, text_test, label_test)
```
Remember, the `train_measure_model` was defined as:
```
def train_measure_model(text_train, label_train, text_val, label_val,
                        cv_binary, cv_analyzer, cv_ngram, cv_max_features,
                        cv_have_tfidf, cv_use_idf, cfr_penalty, cfr_C, stop_words=None, 
                        text_column_name="headline"):
    cv = CountVectorizer(binary=cv_binary, stop_words=stop_words,
                               analyzer=cv_analyzer,
                               ngram_range=cv_ngram[1:3],
                               max_features=cv_max_features)
    if cv_have_tfidf:
        pipeline = Pipeline(steps=[("vectorizer", cv), 
                                   ("tfidf", TfidfTransformer(use_idf=cv_use_idf)),
                                   ("classifier", LogisticRegression(penalty=cfr_penalty,
                                                                     C=cfr_C,
                                                                     random_state=9,
                                                                     max_iter=100,
                                                                     n_jobs=None))])
    else:
        pipeline = Pipeline(steps=[("vectorizer", cv), 
                                   ("classifier", LogisticRegression(penalty=cfr_penalty,
                                                                     C=cfr_C,
                                                                     random_state=9,
                                                                     max_iter=100,
                                                                     n_jobs=None))])

    pipeline.fit(text_train, label_train)
    
    print_metrics(pipeline, text_train, label_train, text_val, label_val)

    return pipeline
```

Now the macro precision on test set is `0.9661`.
Let's boost it one more time:
```
boosted_text_train_2x = pd.concat([text_train]+[extra_text_train]*2)
boosted_label_train_2x = np.concatenate([label_train]+[extra_label_train]*2, axis=0)

cfr_pipeline_2x = train_measure_model(boosted_text_train_2x, boosted_label_train_2x, 
                                      text_val, label_val,
                                      cv_binary=True, cv_analyzer="word", cv_ngram=("w", 1, 3), 
                                      cv_max_features=5000, cv_have_tfidf=True, cv_use_idf=True, 
                                      cfr_penalty="l2", cfr_C=1.0, stop_words=None)

measure_model_on_test(cfr_pipeline_2x, text_test, label_test)
```

Now the macro precision on test set is `0.9664`. We stop here since continuing 
more will not add improve the test metrics and at the same time will 
start to overfit.


**Important Points**.
- In this example we see a slight (`<1%`) however, similar to other techniques, 
depending on the problem, it might improve more or worsen the metrics.
- This technique can be applied to other models, conventional or TL or DL.
- It can be applied to other classification types as well, i.e. multi-label or multi-class 
classifications.
- This technique is not restricted to NLP.
- Overall, it's a good idea to check the quality of the mislabeled samples 
in terms of labels as bad-labeling/inconsistent-labeling is higher in this set. 
(will explain this in another post)


**Note**.
- The complete code for this post can be found at https://github.com/hminooei/DSbyHadi/blob/master/blog/clickbait_conventional.ipynb 