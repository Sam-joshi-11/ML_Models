import pandas as pd 
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np

from sklearn.model_selection import train_test_split

df =pd.read_csv('IMDB Dataset.csv')

df['label'] = df['sentiment'].map({'negative':0,'positive':1})
df = df[['review','label']]
train_df,test_df = train_test_split(df,test_size=0.2,random_state=42,stratify=df['label'])
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
print(train_ds)
print(test_ds)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['review'],padding='max_length',truncation=True,max_length=256)

train_ds = train_ds.map(tokenize,batched=True)
test_ds = test_ds.map(tokenize,batched=True)

print('\nColumns after tokenization:')
print(train_ds.column_names)

print("\nFirst tokenized example:")
print(train_ds[0])

cols = ['input_ids','attention_mask','label']
train_ds.set_format(type='torch',columns=cols)
test_ds.set_format(type='torch',columns=cols)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

accuracy = evaluate.load("accuracy")
precision = evaluate.load('precision')
recall = evaluate.load('recall')
f1 = evaluate.load('f1')

def compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds = np.argmax(logits,axis=-1)
    return {
        'accuracy':accuracy.compute(predictions=preds,references=labels)['accuracy'],
        'precision':precision.compute(predictions=preds,references=labels)['precision'],
        'recall':recall.compute(predictions=preds,references=labels)['recall'],
        'f1':f1.compute(predictions=preds,references=labels)['f1']
    }

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=100
)

trainer = Trainer(
    model=model,    
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
     compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
model.save_pretrained('distilbert_imdb')
tokenizer.save_pretrained('distilbert_imdb')
