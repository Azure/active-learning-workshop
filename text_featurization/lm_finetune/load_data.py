"""Data loaders
Data belongs to original source. Just here for convenience.
"""

import pathlib
import os
import urllib
import pandas as pd
import numpy as np
import csv
import sys
import time
import tarfile
import spacy
from collections import Counter, defaultdict


data_dir = pathlib.Path.home() / "active-learning-workshop" / "text_featurization" / "data"


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.**2 * duration)
    percent = count * block_size * 100. / total_size
    sys.stdout.write("\r%d%% | %d MB | %.2f MB/s | %d sec elapsed" %
                    (percent, progress_size / (1024.**2), speed, duration))
    sys.stdout.flush()

def downloader(source, target, reporthook=reporthook):
    
    if (sys.version_info < (3, 0)):
        import urllib
        urllib.urlretrieve(source, target, reporthook)
    else:
        import urllib.request
        urllib.request.urlretrieve(source, target, reporthook)



def load_imdb_data(data_dir:str=data_dir, out_numpy:bool=False):
    
    """Load IMDB Dataset into numpy or pandas dataframe

    Returns movies DataFrame of reviews. For more information about the dataset, please see the original source:
    [Andrew Maas, Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/). 
    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). 
        Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

    data_dir : str, optional
        Path to save data (the default is data_dir, which is `tatk.utils.load_data.data_dir`)
    out_numpy : bool, optional
        Whether to return numpy or Pandas (the default is True, which returns a tuple of length four of numpy arrays)
    
    Returns
    -------
    Numpy or Pandas data of reviews
    """
    
        
    import sys
    import time
    import tarfile
    import pandas as pd
    import pathlib

    tar_file = pathlib.Path(data_dir) / "imdb" / "aclImdb_v1.tar.gz"  
    tar_file = str(tar_file)
    target_dir = pathlib.Path(data_dir) / "imdb"
    target_dir = str(target_dir)

    print(50 * '-')
    print("Loading from disk if available, otherwise extract and load downloaded tar")

    csv_path = pathlib.Path(data_dir) / "imdb" / "reviews.csv"  
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True)

    if os.path.isfile(str(csv_path)):
        print(50 * "-")
        print("Loading saved CSV from disk")
        df = pd.read_csv(str(csv_path))
    else:
        if not os.path.isdir(os.path.join(target_dir, "aclImdb")) and not os.path.isfile(tar_file):
            print(50 * '=')
            print('Downloading the IMDB movie review dataset from source')

        if not os.path.isdir(os.path.join(target_dir, "aclImdb")):
            source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            target = tar_file
            downloader(source, target)

        if not os.path.isdir(os.path.join(target_dir, "aclImdb")):
            print(50 * '=')
            print("Extracting tarfile")
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(path = target_dir)
                
        print(50 * "-")
        print("CSV not found, loading from source, this might take a few minutes")
        labels = {'pos': 1, 'neg': 0}
        df = pd.DataFrame()
        for s in ('train', 'test'):
            for l in ('pos', 'neg'):
                path = os.path.join(target_dir, "aclImdb", s, l)
                for file in os.listdir(path):
                    with open(os.path.join(path, file), 'r',
                              encoding='utf-8') as infile:
                        txt = infile.read()
                    df = df.append([[txt, labels[l]]], ignore_index=True)
        df.columns = ['review', 'sentiment']
        df.to_csv(str(csv_path), index=False)

    X_train = df.loc[0:24999, "review"]
    y_train = df.loc[0:24999, 'sentiment']
    X_test = df.loc[25000:, "review"]
    y_test = df.loc[25000:, 'sentiment']


    if out_numpy:
        print("Returning numpy arrays, X_train, y_train, X_test, y_test")
        return X_train, y_train, X_test, y_test
    else:
        print("Returning Pandas DataFrame")
        df["sample"] = "train"
        df.loc[25000:, "sample"] = "test"
        return df

def load_wiki_attacks():

    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    if not data_dir.joinpath("text_data").exists():
        source = "https://activelearning.blob.core.windows.net/activelearningdemo/text_data.zip"
        target = str(data_dir / "text_data.zip")

        downloader(source, target)
        import zipfile
        with open(str(data_dir / "text_data.zip")) as f:
            zf = zipfile.ZipFile(f)
            zf.extractall(str(data_dir / "attacks"))

    toxic_df = pd.read_csv(str(data_dir / "text_data" / "attack_data.csv"), encoding="ISO-8859-1") 
    toxic_df["comment_text"] = toxic_df.comment.replace(r'NEWLINE_TOKEN|[^.,A-Za-z0-9]+',' ', regex=True) 

    return toxic_df



def load_attack_encoded():
    
    source = "https://activelearnwestus.blob.core.windows.net/activelearningdemo/attacks_use_encoded.csv"
    
    if not data_dir.joinpath("attacks_use_encoded.csv").exists():
        target = str(data_dir / "attacks_use_encoded.csv")
        downloader(source, target)
    
    attacks_features = pd.read_csv(str(data_dir / "attacks_use_encoded.csv"))
    
    return attacks_features

def download_glove():
    
    source = "http://nlp.stanford.edu/data/glove.6B.zip"
    target = data_dir.joinpath("glove6B.zip")
    if not target.exists():
        downloader(source, target)
    import zipfile
    with open(str(target)) as f:
        zf = zipfile.ZipFile(f)
        zipfile.extractall(f)
        
def load_glove_k2v(glove_src, w2v_tgt):
    

    from gensim.scripts.glove2word2vec import glove2word2vec
    from gensim.models import KeyedVectors
    
    glove2word2vec(glove_src, w2v_tgt)    
    
    glove_model = KeyedVectors.load_word2vec_format(w2v_tgt, binary=False)
    
    return glove_model

    

def get_random_rep(embedding_dim=50, scale=0.62):
    """The `scale=0.62` is derived from study of the external GloVE
    vectors. We're hoping to create vectors with similar general
    statistics to those.
    """
    return np.random.normal(size=embedding_dim, scale=0.62)


def dataframe2glove(df, vocab):
    
    EMBED_DIM = df.shape[1]
    return np.array([df.loc[w].values if w in df.index else get_random_rep(EMBED_DIM)
                     for w in vocab])


def create_random_lookup(vocab):
    """Create random representations for all the words in `vocab`,
    and return new random representstions for new words tha we
    try to look-up, adding them to the lookup when this happens.
    """
    data =  {w: get_random_rep() for w in vocab}
    return defaultdict(lambda : get_random_rep(), data)


def glove2dict(glove_filename):
    
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        data = {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}
    return data


def create_glove_lookup(glove_filename):
    """Turns an external GloVe file into a defaultdict that returns
    the learned representation for words in the vocabulary and
    random representations for all others.
    """
    glove_lookup = glove2dict(glove_filename)
    EMBED_DIM = len(next(iter(glove_lookup.values())))
    glove_lookup = defaultdict(lambda : get_random_rep(EMBED_DIM), glove_lookup)
    return glove_lookup


def create_lookup(X):
    """Map a dataframe to a lookup that returns random vector reps
    for new words, adding them to the lookup when this happens.
    """
    embedding_dim = X.shape[1]
    data = defaultdict(lambda : get_random_rep())
    for w, vals in X.iterrows():
        data[w] = vals.values
    return data

def tokenize(data, text_col, max_words=500):
    
    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
    data['docs'] = data[text_col].apply(lambda x: nlp(x))
    data['tokens'] = data['docs'].apply(lambda x: [str(word) for word in x])
    data['tokens'] = data['tokens'].apply(lambda x: x if len(x) < max_words else x[:max_words])
    
    return data


    
if __name__ == "__main__":
    
    glove_src =  str(data_dir / "glove.6B.300d.txt")
    glove_lookup = create_glove_lookup(glove_src)
    glove_df = pd.read_csv(glove_src, sep=" ", header=None, quoting=csv.QUOTE_NONE, encoding='utf-8', index_col=0)
    
    
    toxic_df = load_wiki_attacks()
    toxic_df = tokenize(toxic_df, "comment_text")
    toxic_df['glove_aggregate'] = toxic_df.tokens.apply(lambda x: np.mean([glove_lookup[v] for v in x], axis=0))