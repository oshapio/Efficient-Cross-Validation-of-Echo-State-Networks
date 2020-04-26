import numpy as np

def get_rmse(truth, pred):
    return np.sqrt(np.mean((truth.flatten() - pred.flatten()) ** 2))

def get_nrmse(truth, pred, use_axis = 1):
    target_avg = np.expand_dims(np.mean(truth, axis = use_axis),use_axis)
    nrmse_up  = np.mean(np.square(np.linalg.norm(pred - truth)))
    nrmse_down = np.mean(np.square(np.linalg.norm(truth-target_avg)))

    return np.sqrt(nrmse_up / nrmse_down)