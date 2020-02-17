# see https://www.spinningbytes.com/resources/germansentiment/ and https://github.com/aritter/twitter_download for obtaining the data.

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from conversion import convert_examples_to_features, convert_text_to_examples

def load_datasets(data_dir, num_categories, test_size):
    data = pd.read_csv(os.path.join(data_dir, "downloaded.tsv"), sep="\t", na_values="Not Available",
                   names=["id", "sentiment", "tweet_id", "?", "text"], index_col='id')
    data = data.dropna(how='any')[['sentiment', 'text']]
    data['sentiment'][data['sentiment'] == 'neutral'] = 2
    data['sentiment'][data['sentiment'] == 'negative'] = 0
    data['sentiment'][data['sentiment'] == 'positive'] = 1
    if num_categories == 2:
        data = data[np.logical_not(data.sentiment==2)]
    X = data['text']
    y = data['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0)
    return (X_train, y_train, X_test, y_test)

def get_tweets_data(data_dir, subtask, num_categories, tokenizer, max_seq_length, test_size):
    fn = os.path.join(data_dir, "data_"+subtask+"_"+str(num_categories)+"cat_"+str(max_seq_length)+".npz")
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
        X_train, y_train, X_test, y_test = load_datasets(data_dir, num_categories, test_size)

        # Create datasets (Only take up to max_seq_length words for memory)
        train_text = X_train.to_list()
        train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
        train_text = np.array(train_text, dtype=object)[:, np.newaxis]
        train_label = y_train.tolist()
    
        test_text = X_test.tolist()
        test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
        test_text = np.array(test_text, dtype=object)[:, np.newaxis]
        test_label = y_test.tolist()
    
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
    