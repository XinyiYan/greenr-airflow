import numpy as np
import pandas as pd
import torch
from torch.utils import data
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split
from process_data import preprocess_text, get_text_processor

class textData(data.Dataset):
    def __init__(self, text, attens, labels=None):
        self.text = text
        self.attens = attens
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        sentence = self.text[idx]
        atten = self.attens[idx]

        if self.labels is None:
            return sentence, atten

        label = self.labels[idx]
        return sentence, atten, torch.tensor(label)


def load_data(file_path, data_sample_per_class=2300):
    data_df = pd.read_csv(file_path)
    data_df.dropna(subset = ["Text"], inplace=True)

    label_0 = data_df.query('Label == 0').sample(n=data_sample_per_class)
    label_1 = data_df.query('Label == 1').sample(n=data_sample_per_class)
    label_2 = data_df.query('Label == 2').sample(n=data_sample_per_class)

    merged_data = pd.concat([label_0, label_1, label_2], ignore_index=True)
    tweets_texts = merged_data['Text'].values
    tweets_labels = (merged_data['Label'].values).astype(np.int)
    return tweets_texts, tweets_labels


def preprocessing_for_bert(input_sents):
    input_ids = []
    attention_masks = []
    
    text_processor = get_text_processor()
    for sent in input_sents:
        encoded_sent = bert_tokenizer.encode_plus(
            text=preprocess_text(text_processor, sent, keep_hashtags=True),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=400,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            truncation=True,
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks


def train_epoch(epoch, model, data_loader, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    predictions, true_labels = [], []

    for i, (inputs, attns, labels) in enumerate(data_loader):
      print("Training...." + str(i))
      inputs = inputs.to(device)
      attns = attns.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      loss, logits = model(input_ids=inputs, attention_mask=attns, labels=labels).to_tuple()
      _, preds = torch.max(logits, dim=1)
      running_loss += loss.item()

      predictions.extend(preds.tolist())
      true_labels.extend(labels.tolist())

      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      scheduler.step()

      if i % 10 == 0:
          print("Train batch:{}, loss:{}".format(i, loss))

    epoch_loss = running_loss / len(data_loader.dataset)
    print("Train epoch:{}, loss:{}".format(epoch, epoch_loss))
    precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    return epoch_loss, fscore


def evaluate_epoch(epoch, model, data_loader):
    model.eval()
    running_loss = 0.0
    predictions, true_labels = [], []

    with torch.no_grad():
      for i, (inputs, attns, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        attns = attns.to(device)
        labels = labels.to(device)

        loss, logits = model(input_ids=inputs, attention_mask=attns, labels=labels).to_tuple()
        _, preds = torch.max(logits, dim=1)
        running_loss += loss.item()

        predictions.extend(preds.tolist())
        true_labels.extend(labels.tolist())

    epoch_loss = running_loss / len(data_loader.dataset)
    print("Eval epoch:{}, loss:{}".format(epoch, epoch_loss))
    print(classification_report(true_labels, predictions))
    precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    print(precision, recall, fscore)
    return epoch_loss, fscore


if __name__ == "__main__":
    np.random.seed(42)

    # training parameter
    batch_size = 8
    epochs = 4

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # split train/test data
    tweets_texts, tweets_labels = load_data("/usr/local/airflow/data/english-anot-shuffled.csv")
    X_train, X_test, y_train, y_test = train_test_split(tweets_texts, tweets_labels, test_size=0.1, random_state=42)

    train_inputs, train_masks = preprocessing_for_bert(X_train)
    val_inputs, val_masks = preprocessing_for_bert(X_test)

    train_dataset = textData(train_inputs, train_masks, y_train)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = textData(val_inputs, val_masks, y_test)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    print('training examples',len(train_dataset))
    print('testing examples', len(val_dataset))
    print('training batch size ',len(train_loader ))
    print('testing batch size', len(val_loader))

    # initialize model
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 3, output_attentions=False, output_hidden_states=False)
    bert_model = bert_model.to(device)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(epochs * len(train_dataset) * 0.1 / batch_size)
    optimizer = AdamW(bert_model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)

    best_f1 = 0.0
    for i in range(epochs):
        train_epoch(i, bert_model, train_loader, optimizer, scheduler)
        epoch_loss, fscore = evaluate_epoch(i, bert_model, val_loader)

        if fscore > best_f1:
            best_f1 = fscore
            print("Model saved!")
            torch.save(bert_model.state_dict(), '/usr/local/airflow/models/best_model_f1_{:.8f}_{}.pth'.format(fscore, i))


