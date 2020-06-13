import os
import argparse
import json
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import bert # https://github.com/kpe/bert-for-tf2/
from onecycle import OneCycleScheduler # https://www.avanwyk.com/tensorflow-2-super-convergence-with-the-1cycle-policy/
from imdb import get_imdb_data
from tweets import get_tweets_data
from amazon import get_reviews_data


parser = argparse.ArgumentParser()

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

parser.add_argument("--experiment_name", type=str, default=current_time, help="Insert string defining your experiment. Defaults to datetime.now()")

parser.add_argument("--task", type=str, required=True, help="One of imdb, reviews, or tweets.")   
parser.add_argument("--subtask", type=str, default="german", help="One of german or multi. Ignored for imdb task.")
parser.add_argument("--ckpt_name", type=str, default="bert_model.ckpt", help="Name of BERT checkpoint to load.")
parser.add_argument("--bert_base_path", type=str, default="D:/bert_models/", help="Where to find BERT models.")
parser.add_argument("--model_name", type=str, default=None, help="Name of BERT model. Default depends on task.")
parser.add_argument("--data_dir", type=str, default="data", help="Data directory.")
parser.add_argument("--log_dir", type=str, default="D:\\logs", help="Log directory.")
# training parameters
parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping.")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum frequence length.")
parser.add_argument("--no_class_weights", action='store_true', help="Don't use class weights.")
parser.add_argument("--num_epochs", type=int, default=3, help="Maximum number of epochs.")
parser.add_argument("--test_size", type=float, default=None, help="Test size. Default depends on task.")
parser.add_argument("--num_categories", type=int, default=None, help="Number of categoroies. Defaults to 2 for imdb, 3 otherwise.")
parser.add_argument("--polarized", action='store_true', help="For reviews data: if true and num_categories=3, count only 1 and 5 as pos/neg")
print('Experiment name is ' + current_time + '.')

# read variables
ARGS = parser.parse_args()
experiment_name = ARGS.experiment_name
batch_size = ARGS.batch_size
learning_rate = ARGS.learning_rate
max_seq_length = ARGS.max_seq_length
ckpt_name = ARGS.ckpt_name
use_class_weights = not ARGS.no_class_weights
num_epochs = ARGS.num_epochs
task = ARGS.task
bert_base_path = ARGS.bert_base_path
num_categories = ARGS.num_categories
model_name = ARGS.model_name
test_size = ARGS.test_size
subtask = ARGS.subtask
data_dir = ARGS.data_dir
log_dir = ARGS.log_dir
patience = ARGS.patience
polarized = ARGS.polarized

if task == "imdb":
    if model_name == None:
        model_name = "uncased_L-12_H-768_A-12"
    if num_categories == None:
        num_categories = 2
elif task == "tweets":
    if model_name == None:
        model_name = "bert_base_german_cased" if subtask == "german" else "multi_cased_L-12_H-768_A-12"
    if num_categories == None:
        num_categories = 3
    if test_size == None:
        test_size = 0.2
elif task == "reviews":
    if model_name == None:
        model_name = "bert_base_german_cased" if subtask == "german" else "multi_cased_L-12_H-768_A-12"
    if num_categories == None:
        num_categories = 3
    if test_size == None:
        test_size = 0.5
else:
    raise Exception('No such task.')

ARGS.model_name = model_name
ARGS.num_categories = num_categories
ARGS.test_size = test_size

log_dir = os.path.join(log_dir, experiment_name)
data_dir = os.path.join(data_dir, task)
        
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
config = vars(ARGS)
json.dump(config, open(os.path.join(log_dir, 'config.json'), 'w'), indent=4, sort_keys=True)

if subtask != 'german' and subtask != 'multi':
    raise Exception("No such subtask.")


def get_data(task, subtask, num_categories, data_dir, tokenizer, max_seq_length, test_size):
    if task == "imdb":
        print("Ignoging test_size for imdb data.")
        return get_imdb_data(data_dir, tokenizer, max_seq_length)
    elif task == "tweets":
        return get_tweets_data(data_dir, subtask, num_categories, tokenizer, max_seq_length, test_size)
    elif task == "reviews":
        return get_reviews_data(data_dir, subtask, num_categories, tokenizer, max_seq_length, test_size, polarized)
    else:
        raise Exception('No such task.')


if __name__ == "__main__":
    bert_path = os.path.join(bert_base_path, model_name)
    model_ckpt = os.path.join(bert_path, ckpt_name)
    do_lower_case = model_name.find("uncased") != -1
    bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, model_ckpt)
    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
    
    (   train_input_ids,
        train_input_masks,
        train_segment_ids,
        train_labels,
        test_input_ids,
        test_input_masks,
        test_segment_ids,
        test_labels
        ) = get_data(task, subtask, num_categories, data_dir, tokenizer, max_seq_length, test_size)
        
    steps = np.ceil(train_input_ids.shape[0] / batch_size) * num_epochs
    lr_schedule = OneCycleScheduler(learning_rate, steps)
    es = EarlyStopping(monitor='val_SparseCategoricalAccuracy', mode='max', verbose=1, patience=patience)
    mc = ModelCheckpoint(os.path.join(log_dir, 'best_model.h5'), monitor='val_SparseCategoricalAccuracy', mode='max', save_best_only=True, save_weights_only=True)
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
    bert.load_bert_weights(l_bert, model_ckpt) 
    
    model.summary()

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                          write_graph=False, write_images=False, update_freq=1000)
    y = np.concatenate([train_labels, test_labels]).flatten()
    wgt = compute_class_weight('balanced', np.unique(y), y)
    if not use_class_weights:
        wgt = (wgt * 0 + 1) / num_categories
    print('Class weights:', wgt)
    model.fit(
        train_input_ids,
        train_labels,
        class_weight=wgt, 
        validation_data=(test_input_ids, test_labels),
        shuffle=True,
        epochs=num_epochs,
        batch_size=batch_size,        
        callbacks=[tensorboard_callback, es, mc, lr_schedule]
    )
    model.load_weights(os.path.join(log_dir, 'best_model.h5'))
    print("Reloaded best parameters.")
    y_pred = model.predict(test_input_ids)
    y_pred = np.argmax(y_pred, axis=1)
    BMAC = balanced_accuracy_score(test_labels, y_pred)
    print(BMAC)