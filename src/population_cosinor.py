import numpy as np

# Load in a bunch of different cosinor models

class PopulationCosinorModel():
    def __init__(self, N_components = 2):
        self.N_components = N_components

    def fit(self, t, y, ids, period = 24 * 3600, remove_outliers = False, alpha = 0.1):
        # Will need to add in a function here to limit to only the ids that have enough data
        pass


# for i_idx, icu_id in enumerate(unique_icu):
#     curr_data = data_in.loc[data_in.ICUSTAY_ID == icu_id, :]
#     y_est, params, p_zero_amp[i_idx], residual_tests[i_idx, :] = cosinor.N_cosinor_model(N, curr_data.ts, curr_data.t,
#                                                                                          input_params, False)
#     h_zero_amp[i_idx] = p_zero_amp[i_idx] < input_params['alpha']
#     data_in.loc[data_in.ICUSTAY_ID == icu_id, 'cosine'] = y_est
#
#     R_squared_adjusted[i_idx] = np.corrcoef(curr_data.ts, y_est)[0, 1]
#     if np.isinf(R_squared_adjusted[i_idx]):
#         R_squared_adjusted[i_idx] = np.nan
#
#     params_store[i_idx, :] = list(params.values())