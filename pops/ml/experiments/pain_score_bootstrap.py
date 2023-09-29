import argparse
import os

import numpy
from tqdm import tqdm

from pops import base_directory
from pops.ml.experiments.pain_score_multisite import PostopPainModel, load_data, generate_dataloader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nbootstrap', default=1, type=int, help="Number of bootstrap iterations")
parser.add_argument('-c', '--checkpoint', default='all', help="Checkpoint prefix")
parser.add_argument('-l', '--locations', nargs='*')
parser.add_argument('-r', '--retrain', action='store_true', help="Whether to refit existing models")
parser.add_argument('-a', '--augment', action='store_true', help="Whether to augment dataset with ICD only data for CPT-free counterfactuals")

def run(retrain = False, locations = None, checkpoint = "all", nbootstrap = 1, augment = True):
    if augment:
        dataset = load_data(return_icd_sequences=True)
    else:
        dataset = load_data()
    rng = numpy.random.default_rng(0)
    
    # Fit point estimate
    fit_model(base_directory + "/data/models/pain_bootstrap", checkpoint, dataset, locations, resample=False, retrain=retrain, augment=augment)

    # Fit bootstrap
    for i in tqdm(range(nbootstrap)):
        fit_model(base_directory + "/data/models/pain_bootstrap", f"{checkpoint}_{i}", dataset, locations, rng=rng, retrain=retrain, augment=augment)

def fit_model(dirpath, filename, dataset, locations=None, resample=True, rng=numpy.random.default_rng(0), retrain=False, augment = True):
    if not os.path.isfile(dirpath + "/" + filename + ".ckpt") or retrain:
        if augment:
            demographics, sequences, outcomes, training, validation, location, vocab_size, icd_sequences = dataset
        else:
            demographics, sequences, outcomes, training, validation, location, vocab_size = dataset
        train_loader, val_loader, eval_loader = generate_dataloader(dataset, locations, resample=resample, rng=rng, augment=augment)
        
        model = PostopPainModel(vocab_size, 256, demographics.shape[-1], 16, 256, outcomes.shape[-1])
        checkpoint_callback = ModelCheckpoint(dirpath=dirpath, filename=filename, save_top_k=1, monitor="val_loss")
        trainer = Trainer(max_epochs=4, accelerator="gpu", devices=1, callbacks=[checkpoint_callback])
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    args = parser.parse_args()
    run(**vars(args))
