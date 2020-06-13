import os
import tensorflow as tf
import pandas as pd
from langdetect import detect
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from conversion import convert_examples_to_features, convert_text_to_examples

def my_detect(s):    
    try: 
        lang = detect(s)
    except:
        lang = "na"
    return lang

def sentiment(stars, polarized):
    if polarized:
        if stars == 1:
            sent = 0
        elif stars == 5:
            sent = 1
        else:
            sent = 2
    else:
        if stars < 3:
            sent = 0
        elif stars == 3:
            sent = 2
        else:
            sent = 1
    return sent
    
    
def download_dataset(data_dir):
    fn = os.path.join(data_dir, "reviews.csv")
    if os.path.isfile(fn):
        data = pd.read_csv(fn, index_col=0)
    else:
        dataset = tf.keras.utils.get_file(
            fname="amazon_reviews_multilingual_DE_v1_00.tsv.gz",
            origin="https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_DE_v1_00.tsv.gz",
            extract=True
        )
    
        data = pd.read_csv(dataset, sep="\t")[["review_body", "star_rating"]] # only need these two columns
        data["language"] = data.apply(lambda row: my_detect(row.review_body), axis=1) # detect language; some reviews are in English
        data = data[data.language=='de'] # only keep German reviews
        data = data.drop_duplicates(subset='review_body') # drop duplicate reviews
        data = data[["review_body", "star_rating"]] # drop language column
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data.to_csv(fn)    
    return data

def load_datasets(data_dir, test_size, num_categories, polarized):
    data = download_dataset(data_dir)
    data['sentiment'] = data.apply(lambda row: sentiment(row.star_rating, polarized), axis=1)
    if num_categories == 2:
        data = data[np.logical_or(data.star_rating==1, data.star_rating==5)]
    grouped = data.groupby('sentiment')
    sample = grouped.apply(lambda group: group.sample(min(grouped.size()), random_state=0))
    sample.reset_index(drop=True)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    sample.to_csv(os.path.join(data_dir, "sample.csv"))
    X = sample['review_body']
    y = sample['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0)
    return (X_train, y_train, X_test, y_test)


def get_reviews_data(data_dir, subtask, num_categories, tokenizer, max_seq_length, test_size, polarized):
    
    fn = os.path.join(data_dir, "data_"+subtask+"_"+str(num_categories)+"cat_"+str(max_seq_length)+("_pol" if polarized else "")+".npz")
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
        X_train, y_train, X_test, y_test = load_datasets(data_dir, test_size, num_categories, polarized)

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
    
    