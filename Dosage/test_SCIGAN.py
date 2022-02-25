# Copyright (c) 2020, Ioana Bica

import argparse
import os
import shutil
import tensorflow as tf
import json

from data_simulation import get_dataset_splits, TCGA_Data
from SCIGAN import SCIGAN_Model
from utils.evaluation_utils import compute_eval_metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_treatments", default=3, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    parser.add_argument("--save_dataset", default=True)
    parser.add_argument("--validation_fraction", default=0.1, type=float)
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--model_name", default="scigan")
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--h_dim", default=64, type=int)
    parser.add_argument("--rep", default=2, type=int)
    parser.add_argument("--h_inv_eqv_dim", default=64, type=int)
    parser.add_argument("--num_dosage_samples", default=5, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)

    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()

    dataset_params = dict()
    dataset_params['num_treatments'] = args.num_treatments
    dataset_params['treatment_selection_bias'] = args.treatment_selection_bias
    dataset_params['dosage_selection_bias'] = args.dosage_selection_bias
    dataset_params['save_dataset'] = args.save_dataset
    dataset_params['validation_fraction'] = args.validation_fraction
    dataset_params['test_fraction'] = args.test_fraction
    save_path = 'saved_models/' + args.model_name + '/tr' +str(args.num_treatments) + '/trb' + str(args.treatment_selection_bias) +'/dob' +str(args.dosage_selection_bias)
    export_dir = save_path + '/model'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(export_dir)
    
    result = {}
    result['in'] = []
    result['out'] = []
    for r in range(args.rep):
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        data_class = TCGA_Data(dataset_params)
        dataset = data_class.dataset
        dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)

        params = {'num_features': dataset_train['x'].shape[1], 'num_treatments': args.num_treatments,
                'num_dosage_samples': args.num_dosage_samples, 'export_dir': export_dir,
                'alpha': args.alpha, 'batch_size': args.batch_size, 'h_dim': args.h_dim,
                'h_inv_eqv_dim': args.h_inv_eqv_dim}

        model_baseline = SCIGAN_Model(params)

        model_baseline.train(Train_X=dataset_train['x'], Train_T=dataset_train['t'], Train_D=dataset_train['d'],
                            Train_Y=dataset_train['y_normalized'], verbose=args.verbose)

        print('-----------------Test------------------')
        mise, dpe, pe, ate = compute_eval_metrics(dataset, dataset_test['x'], num_treatments=args.num_treatments,
                                            num_dosage_samples=args.num_dosage_samples, model_folder=export_dir)
        print("Mise: %s" % str(mise))
        print("DPE: %s" % str(dpe))
        print("PE: %s" % str(pe))
        print("ATE: %s" % str(ate))
        result['out'].append(ate)

        mise, dpe, pe, ate = compute_eval_metrics(dataset, dataset_val['x'], num_treatments=args.num_treatments,
                                            num_dosage_samples=args.num_dosage_samples, model_folder=export_dir)
        print('-----------------Val------------------')
        print("Mise: %s" % str(mise))
        print("DPE: %s" % str(dpe))
        print("PE: %s" % str(pe))
        print("ATE: %s" % str(ate))
        result['in'].append(ate)

        with open(save_path + '/result.json', 'w') as fp:
            json.dump(result, fp)
