import os
import re
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from conversion import convert_examples_to_features, convert_text_to_examples

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True,
    )

    train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))

    return train_df, test_df





def get_imdb_data(data_dir, tokenizer, max_seq_length):
    fn = os.path.join(data_dir, "data_"+str(max_seq_length)+".npz")
    if Path(fn).is_file():
        f= np.load(fn)
        train_input_ids = f['train_input_ids']
        train_input_masks = f['train_input_masks']
        train_segment_ids = f['train_segment_ids']
        train_labels = f['train_labels']
        test_input_ids = f['test_input_ids']
        test_input_masks = f['test_input_masks']
        test_segment_ids = f['test_segment_ids']
        test_labels = f['test_labels']
        f.close()
    else:
        train_df, test_df = download_and_load_datasets()

        # Create datasets (Only take up to max_seq_length words for memory)
        train_text = train_df["sentence"].tolist()
        train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
        train_text = np.array(train_text, dtype=object)[:, np.newaxis]
        train_label = train_df["polarity"].tolist()
    
        test_text = test_df["sentence"].tolist()
        test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
        test_text = np.array(test_text, dtype=object)[:, np.newaxis]
        test_label = test_df["polarity"].tolist()
    
        # Convert data to InputExample format
        train_examples = convert_text_to_examples(train_text, train_label)
        test_examples = convert_text_to_examples(test_text, test_label)
    
        # Convert to features
        (
            train_input_ids,
            train_input_masks,
            train_segment_ids,
            train_labels,
        ) = convert_examples_to_features(
            tokenizer, train_examples, max_seq_length=max_seq_length
        )
        (
            test_input_ids,
            test_input_masks,
            test_segment_ids,
            test_labels,
        ) = convert_examples_to_features(
            tokenizer, test_examples, max_seq_length=max_seq_length
        )
        
        np.savez(fn,
            train_input_ids=train_input_ids,
            train_input_masks=train_input_masks,
            train_segment_ids=train_segment_ids,
            train_labels=train_labels,
            test_input_ids=test_input_ids,
            test_input_masks=test_input_masks,
            test_segment_ids=test_segment_ids,
            test_labels=test_labels
        )
    
    
    return (
        train_input_ids,
        train_input_masks,
        train_segment_ids,
        train_labels,
        test_input_ids,
        test_input_masks,
        test_segment_ids,
        test_labels
        )
    