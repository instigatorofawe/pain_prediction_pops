import argparse

import torch
import numpy
from tqdm import tqdm

from pops import base_directory
from pops.ml.experiments.pain_score_multisite import PostopPainModel
from pops.util import save_R, read_R

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--nbootstrap', default=500, type=int, help="Number of bootstrap iterations")
parser.add_argument('-c', '--checkpoint', default='all', help="Checkpoint prefix")
parser.add_argument('-l', '--locations', nargs='*')
parser.add_argument('-o', '--output', default='pain_bootstrap_synthetic_all.rds')

def load_data():
    data = read_R(base_directory + '/data/pain_synthetic_patients.rds')
    demographics_t = torch.as_tensor(data['demographics'], dtype=torch.float32)
    sequences_t = torch.as_tensor(data['sequences'], dtype=torch.int32)
    return demographics_t, sequences_t

def run(nbootstrap=500, checkpoint='all', locations=None, output='pain_bootstrap_synthetic_all.rds'):
    demographics, sequences = load_data()
    predictions = numpy.empty((nbootstrap, sequences.shape[0], 5))
    weights = numpy.empty((nbootstrap, sequences.shape[0], sequences.shape[1]))
    counterfactuals = numpy.empty(numpy.concatenate(([nbootstrap], sequences.shape, [5])))

    counterfactual_sequences = sequences.unsqueeze(-1).tile((1, 1, sequences.shape[-1]))
    for j in range(sequences.shape[-1]):
        counterfactual_sequences[:,j,j] = 0

    for i in tqdm(range(nbootstrap)):
        model = PostopPainModel.load_from_checkpoint(base_directory + "/data/models/pain_bootstrap/" + f"{checkpoint}_{i}.ckpt")
        
        with torch.no_grad():
            model.eval()
            predictions[i,:], current_weights = model(sequences, demographics)
            weights[i,:] = current_weights.mean(1)

            for j in range(sequences.shape[-1]):
                counterfactuals[i,:,j,:], _ = model(counterfactual_sequences[:,:,j], demographics)
    
    save_R(base_directory + f'/data/results/{output}',
            {'pred': predictions, 'counterfactuals': counterfactuals, 'weights': weights})


if __name__ == "__main__":
    args = parser.parse_args()
    run(**vars(args))
