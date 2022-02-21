# Copyright (c) 2020, Ioana Bica

import numpy as np
from scipy.integrate import romb
import tensorflow as tf
from torch.autograd import Variable
import matplotlib.pyplot as plt
from data_simulation import get_patient_outcome
from scipy.optimize import minimize
import torch

def plt_adrf(x, y_t, y=None):
    c1 = 'gold'
    c2 = 'grey'
    c3 = '#d7191c'
    c4 = 'red'
    c0 = '#2b83ba'
    #plt.plot(x, y_t, marker='', ls='-', label='Truth', linewidth=4, color=c1)
    plt.scatter(x, y_t, marker='*', label='Truth', alpha=0.9, zorder=3, color=c1, s=15)
    if y is not None:
        plt.scatter(x, y, marker='+', label='TransTTE', alpha=0.9, zorder=3, color='#d7191c', s=15)
    plt.grid()
    plt.legend()
    plt.xlabel('Treatment')
    plt.ylabel('Response')
    plt.savefig("transtee.pdf", bbox_inches='tight')
    plt.close()

def sample_dosages(batch_size, num_treatments, num_dosages):
    dosage_samples = np.random.uniform(0., 1., size=[batch_size, num_treatments, num_dosages])
    return dosage_samples

def get_model_predictions(num_treatments, test_data, model):
    x = Variable(torch.from_numpy(test_data['x']).cuda().detach()).float()
    t = Variable(torch.from_numpy(test_data['t']).cuda().detach()).float()
    d = Variable(torch.from_numpy(test_data['d']).cuda().detach()).float()
    I_logits = model(x, t, d)
    return I_logits.cpu().detach().numpy()


def get_true_dose_response_curve(news_dataset, patient, treatment_idx):
    def true_dose_response_curve(dosage):
        y = get_patient_outcome(patient, news_dataset['metadata']['v'], treatment_idx, dosage)
        return y

    return true_dose_response_curve


def compute_eval_metrics(dataset, test_patients, num_treatments, num_dosage_samples, model_folder, model, train=False):
    mises = []
    ites = []
    dosage_policy_errors = []
    policy_errors = []
    pred_best = []
    pred_vals = []
    true_best = []

    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 1. / num_integration_samples
    treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

    for patient in test_patients:
        if train and len(pred_best) > 10:
            return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)
        for treatment_idx in range(num_treatments):
            test_data = dict()
            test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
            test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
            test_data['d'] = treatment_strengths
 
            pred_dose_response = get_model_predictions(num_treatments=num_treatments, test_data=test_data, model=model)
            # pred_dose_response = pred_dose_response * (
            #         dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
            #                         dataset['metadata']['y_min']

            true_outcomes = [get_patient_outcome(patient, dataset['metadata']['v'], treatment_idx, d) for d in
                                treatment_strengths]
            
            # if len(pred_best) < num_treatments and train == False:
            #     #print(true_outcomes)
            #     print([item[0] for item in pred_dose_response])
            mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
            inter_r = np.array(true_outcomes) - pred_dose_response.squeeze()
            ite = np.mean(inter_r ** 2)
            mises.append(mise)
            ites.append(ite)

            best_encountered_x = treatment_strengths[np.argmax(pred_dose_response)]

            def pred_dose_response_curve(dosage):
                test_data = dict()
                test_data['x'] = np.expand_dims(patient, axis=0)
                test_data['t'] = np.expand_dims(treatment_idx, axis=0)
                test_data['d'] = np.expand_dims(dosage, axis=0)

                ret_val = get_model_predictions(num_treatments=num_treatments, test_data=test_data, model=model)
                # ret_val = ret_val * (dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                #             dataset['metadata']['y_min']
                return ret_val

            true_dose_response_curve = get_true_dose_response_curve(dataset, patient, treatment_idx)

            min_pred_opt = minimize(lambda x: -1 * pred_dose_response_curve(x),
                                    x0=[best_encountered_x], method="SLSQP", bounds=[(0, 1)])

            max_pred_opt_y = - min_pred_opt.fun
            max_pred_dosage = min_pred_opt.x
            max_pred_y = true_dose_response_curve(max_pred_dosage)

            min_true_opt = minimize(lambda x: -1 * true_dose_response_curve(x),
                                    x0=[0.5], method="SLSQP", bounds=[(0, 1)])
            max_true_y = - min_true_opt.fun
            max_true_dosage = min_true_opt.x

            dosage_policy_error = (max_true_y - max_pred_y) ** 2
            dosage_policy_errors.append(dosage_policy_error)

            pred_best.append(max_pred_opt_y)
            pred_vals.append(max_pred_y)
            true_best.append(max_true_y)
            

        selected_t_pred = np.argmax(pred_vals[-num_treatments:])
        selected_val = pred_best[-num_treatments:][selected_t_pred]
        selected_t_optimal = np.argmax(true_best[-num_treatments:])
        optimal_val = true_best[-num_treatments:][selected_t_optimal]
        policy_error = (optimal_val - selected_val) ** 2
        policy_errors.append(policy_error)

    return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)