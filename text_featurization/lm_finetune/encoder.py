"""Encoder module for featurizing and fine-tuning language models on new data.
Utilizes pre-trained modules available on tensorflow_hub and Keras for defining model graph and computing forward-pass or fine-tuning.

Notes:
    - fine-tuning on GPU doesn't work for USE, see [https://github.com/tensorflow/hub/issues/70](https://github.com/tensorflow/hub/issues/70)
    - fine-tuning for ELMO uses a lot of GPU when working with long sequences, and will likely lead to OOM, so you may want to split/truncate.
"""

import tensorflow as tf
import tensorflow_hub as hub
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Keras Imports
import keras
import keras.backend as K
from keras import layers
from keras import Model

class encoder():


    def __init__(self, model="elmo", trainable=False, num_labels=2):
        """Defines an encoder using a pre-trained TensorflowHub module. 
        Can be used for featurization or fine-tuned for classification.
        
        Parameters
        ----------
        model : str, optional
            Trained model from tfhub to use (the default is "elmo")
        trainable : bool, optional
            Whether to fix weights or make them trainable (the default is False)
        num_labels : int, optional
            Number of labels in target class, applicable for classification (the default is 2)

        Raises
        ------
        NotImplementedError
            If self.model not in ["elmo", "nnlm", "use"]
        
        """


        super(encoder, self).__init__()
        # Create a TensorFlow Session and run initializers
        self.session = tf.Session()
        self.model = model
        self.trainable = trainable
        self.num_labels = num_labels
    
    def embed(self, x):
        """Embed string to lower dimensional vector
        
        Parameters
        ----------
        x : list
            list of strings to tokenize and featurize
        
        Returns
        -------
        Trainable featurizer
        """

        
        ELMO = "https://tfhub.dev/google/elmo/2"
        NNLM = "https://tfhub.dev/google/nnlm-en-dim128/1"
        USE = "https://tfhub.dev/google/universal-sentence-encoder-large/2"
        
        model_name = self.model
        if model_name == "elmo":
            elmo = hub.Module(ELMO, trainable=self.trainable)
            executable = elmo(
                tf.squeeze(tf.cast(x, tf.string)),
                signature="default",
                as_dict=True)["default"]

        elif model_name == "nnlm":
            nnlm = hub.Module(NNLM)
            executable = nnlm(tf.squeeze(tf.cast(x, tf.string)))

        elif model_name == "use":
            encoder = hub.Module(USE)
            executable = encoder(tf.squeeze(tf.cast(x, tf.string)),
                                 signature="default", as_dict=True)["default"]
        else:
            raise NotImplementedError

        return executable
    
    
    def transform_model(self):
        """Used for encoding a list of text sentence to a fixed vector.
        
        Returns
        -------
        Keras model for forward pass only
        """

        if self.model == "use":
            embed_size = 512
        elif self.model == "elmo":
            embed_size = 1024
        elif self.model == "nnlm":
            embed_size = 128
            
        input_text = layers.Input(shape=(1,), dtype=tf.string)
        embedding = layers.Lambda(self.embed,
                                  output_shape=(embed_size,))(input_text)
        model = Model(inputs=[input_text], outputs=embedding)
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', metrics=['accuracy'])
    
        return model
    
    def trainable_model(self):
        """A trainable keras module for classification
        """

        
        if self.model == "use":
            embed_size = 512
        elif self.model == "elmo":
            embed_size = 1024
        elif self.model == "nnlm":
            embed_size = 128
            
        n_labels = self.num_labels
            
        input_text = layers.Input(shape=(1,), dtype=tf.string)
        embedding = layers.Lambda(self.embed,
                                  output_shape=(embed_size,))(input_text)
        dense = layers.Dense(256, activation='relu')(embedding)
        pred = layers.Dense(n_labels, activation='softmax')(dense)
        model = Model(inputs=[input_text], outputs=pred)
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', metrics=['accuracy'])
        
        return model

    

if __name__ == "__main__":

    
    ### Featurizing with Pre-Trained Encoders ###
    
    import pathlib
    import pandas as pd
    import os
    import csv
    from load_data import load_imdb_data
    
    data_dir = pathlib.Path.home() / "active-learning-workshop" / "text_featurization" / "data"

    imdb_df = load_imdb_data()
    use_encoder = encoder(model="use")
    featurizer = use_encoder.transform_model()
    featurizer.summary()

    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        transformed_review = featurizer.predict(imdb_df.review.values, batch_size=64)

    pd.DataFrame(transformed_review).to_csv(str(data_dir / "use_reviews.csv"), index=False)

    elmo_encoder = encoder(model="elmo", trainable=True)
    trainable_model = elmo_encoder.trainable_model()
    trainable_model.summary()

    from keras.utils import to_categorical
    y_cat = to_categorical(imdb_df.sentiment)
    y_train = y_cat[:25000]
    y_test = y_cat[25000:]

    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        history = trainable_model.fit(imdb_df.review.values[:25000],
                                      y_train, batch_size=8)

    
    ### Evaluating GloVe Embeddings vs Encoder Representations for Toxic Comments ###
    
    from load_data import load_wiki_attacks, load_attack_encoded
    from load_data import tokenize, create_glove_lookup
    
    glove_src =  str(data_dir / "glove.6B.300d.txt")
    glove_lookup = create_glove_lookup(glove_src)
    glove_df = pd.read_csv(glove_src, sep=" ", header=None, quoting=csv.QUOTE_NONE, encoding='utf-8', index_col=0)
    
    
    toxic_df = load_wiki_attacks()
    toxic_df = tokenize(toxic_df, "comment_text")
    toxic_df['glove_aggregate'] = toxic_df.tokens.apply(lambda x: np.mean([glove_lookup[v] for v in x], axis=0)) 
    
    encoded_attacks = load_attack_encoded()
    toxic_df['encoded_comment'] = encoded_attacks.values.tolist()
    
    from sklearn.model_selection import learning_curve
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import roc_auc_score, make_scorer
    
    lb = LabelBinarizer()
    train_sizes = np.arange(0.1, 1.1, 0.1)
    
#     estimator = GaussianNB()
#     estimator = LogisticRegression()
    estimator = RandomForestClassifier()
    
    def featurize(df=toxic_df):
    
        labels = np.concatenate(lb.fit_transform(df.is_attack.values))
        glove_features = np.vstack(df.glove_aggregate.values)
        use_features = np.vstack(df.encoded_comment.values)
        
        return labels, glove_features, use_features
    
    labels, glove_features, use_features = featurize()
        
    g_train_sizes, g_train_scores, g_test_scores = learning_curve(estimator=estimator, 
                                                                  X=glove_features,
                                                                  y=labels, 
                                                                  scoring=make_scorer(roc_auc_score),
                                                                  n_jobs=8, train_sizes=train_sizes)
    
    e_train_sizes, e_train_scores, e_test_scores = learning_curve(estimator=estimator, 
                                                                  X=use_features,
                                                                  y=labels, 
                                                                  scoring=make_scorer(roc_auc_score),
                                                                  n_jobs=8, train_sizes=train_sizes)
    
    results_df = pd.DataFrame({"train_perc": train_sizes,
                               "bow_auc": np.mean(g_test_scores, axis=1),
                               "encoder_auc": np.mean(e_test_scores, axis=1)})

    sns.lineplot(x="train_perc", y="AUC", hue="features", 
                 data=results_df.melt("train_perc", var_name="features", value_name="AUC"), 
                )

    
    model_df = toxic_df.loc[:,['is_attack', 'comment_text', 'glove_aggregate', 'encoded_comment']]
    
    from sklearn.model_selection import train_test_split, GridSearchCV

    train_df, test_df = train_test_split(model_df, train_size=0.75)
    
    def cv_predict_eval():
        
        cv = GridSearchCV(RandomForestClassifier(),
                         param_grid={
                             'n_estimators': [10, 100],
                             'max_features': ['sqrt', 'log2'],
                             'max_depth': [3, 5, None]}, 
                          refit=True, 
                          n_jobs=20)
        
        
        labels, glove_features, use_features = featurize(train_df)
        labels_test, glove_test, use_test = featurize(test_df)
        
        glove_cv = cv.fit(glove_features, labels)
        glove_hat = cv.predict(glove_test)
        use_cv = cv.fit(use_features, labels)
        use_hat = cv.predict(use_test)
        
        results_df = test_df
        results_df['use_pred'] = use_hat
        results_df['glove_pred'] = glove_hat
        
        return results_df
    
    
    ## where glove fails and use succeeds
    
    results_df.loc[(results_df["is_attack"] == results_df["use_pred"]) & (results_df["is_attack"] != results_df["glove_pred"]) & (results_df["is_attack"] == False), ["comment_text", "is_attack"]]
