import argparse

import numpy
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from pops import base_directory
from pops.util import read_R, save_R

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', default='pain_pooled')
parser.add_argument('-o', '--output', default='pain_multisite_all')
parser.add_argument('-l', '--locations', nargs='*')

class PostopPainModel(LightningModule):
    def __init__(self, vocab_size, embedding_dim, static_dim, attention_heads, hidden_dim, output_dim, binarized=False, threshold=4, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(vocab_size+1, embedding_dim, 0)
        self.multihead_attention = torch.nn.MultiheadAttention(embedding_dim, attention_heads, batch_first=True, dropout=dropout)
        self.dense_hidden_layer = torch.nn.Linear(embedding_dim + static_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.binarized = binarized
        self.threshold = threshold

    def forward(self, sequences, static):
        x, weights = self.encode(sequences)
        x = torch.concat((x, static), 1)
        x = self.dense_hidden_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x, weights

    def encode(self, x):
        x = self.embedding(x)
        x, weights = self.multihead_attention(x, x, x, need_weights=True)
        x = torch.mean(x, 1)
        return x, weights
    
    def loss(self, batch, batch_idx):
        batch_sequences, batch_demographics, batch_outcomes = batch
        pred, _ = self(batch_sequences, batch_demographics)
        mask = ~torch.isnan(batch_outcomes)
        if self.binarized:
            loss = self.bce(pred[mask], (batch_outcomes[mask]>self.threshold)*1.)
        else:
            loss = self.mse(pred[mask], batch_outcomes[mask]/10)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
    
    def predict_step(self, batch, batch_idx):
        batch_sequences, batch_demographics, batch_outcomes = batch
        embeddings, _ = self.encode(batch_sequences)
        pred, weights = self(batch_sequences, batch_demographics)
        return pred, weights.mean(1), embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


def run(checkpoint="pain_pooled", output="pain_multisite_all", locations=None):
    dataset = load_data()
    demographics, sequences, outcomes, training, validation, location, vocab_size = dataset
    train_loader, val_loader, eval_loader = generate_dataloader(dataset, locations)
    
    model = PostopPainModel(vocab_size, 256, demographics.shape[-1], 16, 256, outcomes.shape[-1])
    checkpoint_callback = ModelCheckpoint(dirpath=str(base_directory+'/data/models/' + checkpoint), save_top_k=1, monitor="val_loss")
    trainer = Trainer(max_epochs=8, accelerator="gpu", devices=1, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    predictions = trainer.predict(model, dataloaders=eval_loader, ckpt_path='best')
    all_pred = torch.concat([x[0] for x in predictions], 0).numpy()
    all_weights = torch.concat([x[1] for x in predictions], 0).numpy()
    all_embeddings = torch.concat([x[2] for x in predictions], 0).numpy()
    
    save_R(base_directory + '/data/results/' + output + '.rds', {'pred': all_pred, 'weights': all_weights, 'embeddings': all_embeddings})

    model = PostopPainModel(vocab_size, 256, demographics.shape[-1], 16, 256, outcomes.shape[-1], binarized=True, threshold=4)
    checkpoint_callback = ModelCheckpoint(dirpath=str(base_directory+'/data/models/' + checkpoint + '_binarized_4'), save_top_k=1, monitor="val_loss")
    trainer = Trainer(max_epochs=8, accelerator="gpu", devices=1, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    predictions = trainer.predict(model, dataloaders=eval_loader, ckpt_path='best')
    all_pred = torch.concat([x[0] for x in predictions], 0).numpy()
    all_weights = torch.concat([x[1] for x in predictions], 0).numpy()
    all_embeddings = torch.concat([x[2] for x in predictions], 0).numpy()
    
    save_R(base_directory + '/data/results/' + output + '_binarized_4.rds', {'pred': all_pred, 'weights': all_weights, 'embeddings': all_embeddings})

    model = PostopPainModel(vocab_size, 256, demographics.shape[-1], 16, 256, outcomes.shape[-1], binarized=True, threshold=6)
    checkpoint_callback = ModelCheckpoint(dirpath=str(base_directory+'/data/models/' + checkpoint + '_binarized_6'), save_top_k=1, monitor="val_loss")
    trainer = Trainer(max_epochs=8, accelerator="gpu", devices=1, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    predictions = trainer.predict(model, dataloaders=eval_loader, ckpt_path='best')
    all_pred = torch.concat([x[0] for x in predictions], 0).numpy()
    all_weights = torch.concat([x[1] for x in predictions], 0).numpy()
    all_embeddings = torch.concat([x[2] for x in predictions], 0).numpy()
    
    save_R(base_directory + '/data/results/' + output + '_binarized_6.rds', {'pred': all_pred, 'weights': all_weights, 'embeddings': all_embeddings})

def generate_dataloader(dataset, train_locations=None, resample=False, augment=False, rng = numpy.random.default_rng(0)):
    if augment:
        demographics, sequences, outcomes, training, validation, location, vocab_size, icd_sequences = dataset
        icd_sequences_t = torch.as_tensor(icd_sequences, dtype=torch.int32)
    else:
        demographics, sequences, outcomes, training, validation, location, vocab_size = dataset

    if train_locations is not None:
        train_subset = training & ~validation & numpy.isin(location, train_locations)
        val_subset = training & validation & numpy.isin(location, train_locations) 
    else:
        train_subset = training & ~validation
        val_subset = training & validation

    demographics_t = torch.as_tensor(demographics, dtype=torch.float32) 
    sequences_t = torch.as_tensor(sequences, dtype=torch.int32)
    outcomes_t = torch.as_tensor(outcomes, dtype=torch.float32)
    
    if resample:
        train_sample = rng.choice(numpy.sum(train_subset), numpy.sum(train_subset), replace=True)
        val_sample = rng.choice(numpy.sum(val_subset), numpy.sum(val_subset), replace=True)

        if augment:
            train_data = TensorDataset(
                torch.concat((sequences_t[train_subset, :][train_sample,:], icd_sequences_t[train_subset,:][train_sample,:]), 0),
                torch.concat((demographics_t[train_subset, :][train_sample,:],demographics_t[train_subset, :][train_sample,:]), 0),
                torch.concat((outcomes_t[train_subset, :][train_sample,:], outcomes_t[train_subset, :][train_sample,:]), 0)
            )
        else:
            train_data = TensorDataset(sequences_t[train_subset, :][train_sample,:], demographics_t[train_subset, :][train_sample,:], outcomes_t[train_subset, :][train_sample,:])
        val_data = TensorDataset(sequences_t[val_subset, :][val_sample,:], demographics_t[val_subset, :][val_sample,:], outcomes_t[val_subset, :][val_sample,:])
    else:
        if augment:
            train_data = TensorDataset(
                torch.concat((sequences_t[train_subset, :], icd_sequences_t[train_subset,:]), 0),
                torch.concat((demographics_t[train_subset, :], demographics_t[train_subset, :]), 0),
                torch.concat((outcomes_t[train_subset, :], outcomes_t[train_subset, :]), 0)
            )
        else:
            train_data = TensorDataset(sequences_t[train_subset, :], demographics_t[train_subset, :], outcomes_t[train_subset, :])
        val_data = TensorDataset(sequences_t[val_subset, :], demographics_t[val_subset, :], outcomes_t[val_subset, :])
    eval_data = TensorDataset(sequences_t, demographics_t, outcomes_t)
    
    train_loader = DataLoader(train_data, 128, shuffle=True)
    val_loader = DataLoader(val_data, 128, shuffle=False)
    eval_loader = DataLoader(eval_data, 128, shuffle=False)
    return train_loader, val_loader, eval_loader

def load_data(return_icd_sequences = False):
    demographic_data = read_R(base_directory + "/data/multisite_data.rds")
    sequence_data = read_R(base_directory + "/data/multisite_icd_cpt.rds")

    demographics = demographic_data['demographics']
    outcomes = demographic_data['outcomes']
    training = demographic_data['training'] == 1
    validation = demographic_data['validation'] == 1
    location = demographic_data['location']

    sequences = sequence_data['sequences']
    icd_sequences = sequence_data['icd_sequences']
    vocab_size = sequence_data['pruned_vocab'].shape[-1]

    if return_icd_sequences:
        return demographics, sequences, outcomes, training, validation, location, vocab_size, icd_sequences
    else:
        return demographics, sequences, outcomes, training, validation, location, vocab_size

if __name__ == "__main__":
    arguments = parser.parse_args()
    run(**vars(arguments))