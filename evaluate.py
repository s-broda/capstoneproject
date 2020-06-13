import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score
import bert # https://github.com/kpe/bert-for-tf2/
from langdetect import detect
from conversion import convert_examples_to_features, convert_text_to_examples
from pathlib import Path
import nltk
from nltk import tokenize
nltk.download('punkt')
parser = argparse.ArgumentParser()

parser.add_argument("--experiment_name", type=str, required=True, help="Experiment to get trained weights from")
parser.add_argument("--data_dir", type=str, default="data", help="Data directory.")
parser.add_argument("--log_dir", type=str, default="D:\\logs", help="Log directory.")
parser.add_argument("--num_examples", type=int, default=150, help="How many examples to classify.")


# read variables
ARGS = parser.parse_args()
experiment_name = ARGS.experiment_name
log_dir = ARGS.log_dir

        
with open(os.path.join(log_dir, experiment_name, 'config.json'), "r") as read_file:
    config = json.load(read_file)

polarized = False # doesn't exist for all experiments. may be overwritten by next line
locals().update(config)
experiment_name = ARGS.experiment_name
num_examples = ARGS.num_examples
log_dir = ARGS.log_dir
data_dir = ARGS.data_dir
task = 'democrasci'


log_dir = os.path.join(log_dir, experiment_name)
data_dir = os.path.join(data_dir, task)

def my_detect(s):    
    try: 
        lang = detect(s)
    except:
        lang = "na"
    return lang

def get_speeches(data_dir):
    fn = os.path.join(data_dir, "speeches.csv")    
    if os.path.isfile(fn):
        speeches = pd.read_csv(fn, index_col=0)
        print("Reloaded speeches dataframe.")
    else:
        print("Creating speeches dataframe.")
        speeches = pd.DataFrame(columns=['year', 'session_id', 'speech_id', 'speech'])  
        for year in np.arange(1900, 1981):
            print('Processing year: ', year)
            all_speeches_year = pickle.load(open(os.path.join(data_dir, "AB", year, "06_collectedinformation.pickle"), "rb") )
            for session_id in all_speeches_year.keys():
                for speech_id in all_speeches_year[session_id]['dict_speeches'].keys():
                    speech = all_speeches_year[session_id]['dict_speeches'][speech_id][1]
                    if detect(speech) == 'de':
                        speeches = speeches.append({'year': year, 'session_id': session_id, 'speech_id': speech_id, 'speech': speech}, ignore_index=True)
            speeches.to_csv(os.path.join(data_dir, 'speeches.csv'))        
    return speeches

def get_sentences(data_dir):
    fn = os.path.join(data_dir, "sentences.csv")
    if os.path.isfile(fn):
        sentences = pd.read_csv(fn, index_col=0)
        print("Reloaded sentences dataframe.")
    else:
        print("Creating sentences dataframe.")
        speeches = get_speeches(data_dir)
        for index, row in speeches.iterrows():
            sents = tokenize.sent_tokenize(row['speech'])
            speeches.at[index, 'speech'] = sents    
        sentences = speeches.explode('speech')
        sentences = sentences.sample(frac=1, random_state=0)
        sentences.reset_index(drop=True, inplace=True)
        sentences.rename(columns={'speech':'sentence'}, inplace=True)
        sentences.drop(sentences.columns[sentences.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        sentences['label'] = -1
        sentences.to_csv(os.path.join(data_dir, 'sentences.csv'))
    return sentences


def get_data(data_dir):       
    fn = os.path.join(data_dir, "test.csv")
    if os.path.isfile(fn):
        test = pd.read_csv(fn, index_col=0)
        print("Reloaded test dataframe.")
    else:
        print('Creating test dataframe.') 
        sentences = get_sentences(data_dir)
        num_pos = sum(sentences['label'] == 1)
        num_neg = sum(sentences['label'] == 0)
        num_neu = sum(sentences['label'] == 2)
        if min(num_pos, num_neg, num_neu) < num_examples // 3: 
            print("Test set incomplete. Let's label some more.")
            for index, row in sentences.iterrows():
                if min(num_pos, num_neg, num_neu) >= num_examples // 3: 
                    print('Test set complete.')
                    break            
                if row['label'] == -1:
                    print(row['sentence'])
                    l = input('Enter +/-/0/stop:')
                    if l == '+':
                        num_pos += 1
                        sentences.at[index, 'label'] = 1
                    elif l == '-':
                        num_neg += 1
                        sentences.at[index, 'label'] = 0
                    elif l == '0':
                        num_neu += 1
                        sentences.at[index, 'label'] = 2
                    elif l == 'stop':
                        break
                    else:
                        sentences.at[index, 'label'] = -2 # label so it won't be shown again
            sentences.to_csv(os.path.join(data_dir, 'sentences.csv')) # store the labels in sentences, too, so we don't need to label things again
        test = sentences[sentences.label >= 0]
        test = test.groupby('label')
        test = test.apply(lambda group: group.sample(num_examples // 3, random_state=0))
        test.reset_index(drop=True, inplace=True)
        test = test.drop(['year', 'session_id', 'speech_id'], axis=1)
        if min(num_pos, num_neg, num_neu) >= num_examples // 3: 
            print('Writing test set to disk.')
            test.to_csv(os.path.join(data_dir, 'test.csv'))
        else:
            print('Test set incomplete. Dataframe not written.')
    return test



if __name__ == "__main__":
    
    test = get_data(data_dir)
    
    bert_path = os.path.join(bert_base_path, model_name)
    model_ckpt = os.path.join(bert_path, ckpt_name)
    do_lower_case = model_name.find("uncased") != -1
    bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, model_ckpt)
    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

    
    bert_params = bert.params_from_pretrained_ckpt(bert_path)
    l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
    in_id = keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    bert_output = l_bert(in_id)[:, 0, :]
    dropout = keras.layers.Dropout(0.5)(bert_output)
    dense = keras.layers.Dense(768, activation="relu")(dropout)
    dropout = keras.layers.Dropout(0.5)(dense)
    pred = keras.layers.Dense(num_categories, activation=None)(dropout)
    model = keras.models.Model(inputs=in_id, outputs=pred)    

    opt = keras.optimizers.Nadam()
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=['SparseCategoricalAccuracy'])
    # bert.load_bert_weights(l_bert, model_ckpt) 
    model.load_weights(os.path.join(log_dir, 'best_model.h5'))    
    print("Reloaded best parameters.")
    model.summary()
    
    
    print('Converting sentences to examples.')        
    X = test['sentence'].to_list()
    X = [" ".join(x.split()[0:max_seq_length]) for x in X]
    X = np.array(X, dtype=object)[:, np.newaxis]
    y = np.array(test.label)
    test_examples = convert_text_to_examples(X, np.zeros(len(X)))
    (
        test_input_ids,
        test_input_masks,
        test_segment_ids,
        test_labels,
    ) = convert_examples_to_features(
                tokenizer, test_examples, max_seq_length=max_seq_length
                )
    print("Predicting.")
    y_pred = model.predict(test_input_ids)
    y_pred = np.argmax(y_pred, axis=1)
    test['prediction'] = y_pred
    BMAC = balanced_accuracy_score(y, y_pred)
    print(BMAC)
    test.to_csv(os.path.join(log_dir, "predictions.csv"))
    

        
   
    