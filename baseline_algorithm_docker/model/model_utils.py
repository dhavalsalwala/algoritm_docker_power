from typing import Dict

import numpy as np
import tqdm


def epkl_reg(preds: np.array, epsilon=1e-10) -> np.array:
    """
    Taken from here: https://github.com/yandex-research/shifts.
    Modification/additions:
    Rename vars variable to vars_.
    :param: preds: array [n_models, n_samples, 2] - mean and var along last axis.
    """
    means = preds[:, :, 0]
    vars_ = preds[:, :, 1] + epsilon
    logvars = np.log(vars_)

    avg_means = np.mean(means, axis=0)
    avg_second_moments = np.mean(means * means + vars_, axis=0)

    inv_vars = 1.0 / vars_
    avg_inv_vars = np.mean(inv_vars, axis=0)
    mean_inv_var = inv_vars * means
    avg_mean_inv_var = np.mean(mean_inv_var, axis=0)
    avg_mean2_inv_var = np.mean(means * mean_inv_var + logvars, axis=0) + np.log(2 * np.pi)

    epkl = 0.5 * (avg_second_moments * avg_inv_vars - 2 * avg_means * avg_mean_inv_var + avg_mean2_inv_var)

    return epkl


def ensemble_uncertainties_regression(preds: np.array) -> Dict:
    """
    Taken from here: https://github.com/yandex-research/shifts
    :param: preds: array [n_models, n_samples, 2] - last dim is mean, var
    """
    epkl = epkl_reg(preds=preds)
    var_mean = np.var(preds[:, :, 0], axis=0)
    mean_var = np.mean(preds[:, :, 1], axis=0)

    uncertainty = {'tvar': var_mean + mean_var,
                   'mvar': mean_var,
                   'varm': var_mean,
                   'epkl': epkl}

    return uncertainty


def get_distributions_params(model, x):
    """
    Return the parameters of the normal distributions
    :param model: The model
    :param x: The input tensor
    :return: array [num_samples x 2],
             where
             num_samples = number of rows in data_norm
             2 = [mean, std]
    """
    distrs = model.forward(x)
    return np.concatenate([distrs.loc.detach().numpy().reshape(-1, 1),
                           distrs.scale.detach().numpy().reshape(-1, 1)],
                          axis=1)


def get_ensemble_predictions(model_list, data_norm, multi_runs):
    """
    Gets the normalized predictions of an ensemble of models
    :param model_list: List of models to form an ensemble
    :param data_norm: Torch tensor of the input data
    :param multi_runs: Number of sequential runs per member of the ensemble
    :return: An array with the ensemble predictions with size [(ensemble_size * multi_runs) x num_samples x 2],
            where
            ensemble_size = number of models in models_list
            multi_runs = number of sequential runs per member of the ensemble
            num_samples = number of rows in data_norm
            2 = [mean, variance]
    """
    all_preds = []
    for member in tqdm.tqdm(range(len(model_list))):  # Iterate over the members of the ensemble
        for run in range(multi_runs):  # Each member will perform 'multi_runs' inferences
            preds = np.asarray(get_distributions_params(model=model_list[member], x=data_norm))
            all_preds.append(preds)
    all_preds = np.stack(all_preds, axis=0)

    # convert std to variance
    all_preds[:, :, 1] = np.square(all_preds[:, :, 1])

    return all_preds
