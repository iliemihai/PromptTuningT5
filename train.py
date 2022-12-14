from transformers import (T5TokenizerFast, AdamW, get_scheduler)
import torch
from model import T5PromptTuningLM
import pandas as pd
from datasets import Dataset
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from datasets import Dataset as HDataset

class NewsDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.instances = []

        dataset = pd.read_csv(file_path)
        dataset.dropna(inplace=True)
        for line, label in tqdm(zip(dataset["Body"].values[:], dataset["NewsType"].values[:]), total=len(dataset["NewsType"].values[:])):
            instance = {
                        "sentence": line,
                        "label":"\nLABEL:"+label
                       }
            self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]


class T5Summarization(pl.LightningModule):
    def __init__(self, model_name:str="dumitrescustefan/mt5-large-romanian",
                 num_train_epochs:int=10,
                 weight_decay:float=0.01,
                 learning_rate:float=0.01,
                 num_warmup_steps:int=0,
                 n_prompt_tokens:int=20,
                 init_from_vocab:bool=True):
        super().__init__()
        self.num_train_epochs=num_train_epochs
        self.weight_decay=weight_decay
        self.learning_rate=learning_rate
        self.num_warmup_steps=num_warmup_steps
        self.max_train_steps=num_train_epochs
        self.n_prompt_tokens=n_prompt_tokens
        self.init_from_vocab=init_from_vocab
        self.model_name = model_name

        self.model = T5PromptTuningLM.from_pretrained(self.model_name,
                                                          n_tokens=self.n_prompt_tokens,
                                                          initialize_from_vocab=self.init_from_vocab)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name)
        #if self.tokenizer.pad_token is None:
        #    self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        #self.model.resize_token_embeddings(self.model.config.vocab_size + 1)
        # because reshaping, need to refreeze embeddings
        #for name, param in self.model.named_parameters():
        #    if name == "transformer.wte.weight":
        #        param.requires_grad=False

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

    def my_collate(self, batch):
        sentences = []
        summaries = []
        for instance in batch:
            sentences.append(instance["sentence"])
            summaries.append(instance["label"])
        sentences_batch = self.tokenizer(sentences, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        summaries_batch = self.tokenizer(summaries, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

        sentences_batch["input_ids"][sentences_batch["input_ids"][:, :] == self.tokenizer.pad_token_id] = -100
        summaries_batch["input_ids"][summaries_batch["input_ids"][:, :] == self.tokenizer.pad_token_id] = -100

        return sentences_batch, summaries_batch

    def forward(self, sentence):
        outputs = self.model(input_ids=sentence["input_ids"], attention_mask=sentence["attention_mask"],
                             labels=sentence["input_ids"], )
        loss, logits = outputs.loss, outputs.logits
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.train_loss.append(loss.detach().cpu().numpy())
        return loss#{"train_loss":loss}

    def validation_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.val_loss.append(loss.detach().cpu().numpy())
        return loss#{"valid_loss":loss}

    def test_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.test_loss.append(loss.detach().cpu().numpy())
        return loss#{"test_loss":loss}

    def configure_optimizers(self):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n == "soft_prompt.weight"],
                "weight_decay": self.weight_decay,
             }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # data
    train_dataset = NewsDataset("train.csv")
    test_dataset = NewsDataset("test.csv")
    valid_dataset = NewsDataset("validation.csv")

    model = T5Summarization()
    trainer = pl.Trainer(max_epochs=model.max_train_steps)

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True, collate_fn=model.my_collate, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4, shuffle=False, collate_fn=model.my_collate, pin_memory=True, drop_last=True )
    val_loader = DataLoader(valid_dataset, batch_size=16, num_workers=4, shuffle=False, collate_fn=model.my_collate, pin_memory=True, drop_last=True)

    trainer.fit(model, train_loader, val_loader)
    print("Saving prompt...")
    save_dir_path = "./soft_prompt"
    model.model.save_soft_prompt(save_dir_path)
    #trainer.test(test_dataloaders=test_loader)

if __name__ == "__main__":
    cli_main()
