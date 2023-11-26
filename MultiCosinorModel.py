import numpy as np
from scipy.stats import f as fdist
import matplotlib.pyplot as plt

class MultiCosinorModel():
    def __init__(self, N_components = 2, alpha = 0.05):
        '''
        Class for a multi component cosinor
        '''
        self.N_components = N_components
        self.alpha = alpha


        # The input parameters
        self.period = None # Time period of cosinor in seconds
        self.y = None
        self.t = None

        # Computed params
        self.tau = None

        # The fitted params
        self.M = None
        self.A = None
        self.phi = None
        self.beta = None
        self.gamma = None
        self.ortho = None
        self.bathy = None


    def _cosinor_func(self, A, phi, tau, t):
        return A * np.cos((2*np.pi*t/tau) + phi)

    def fit(self, t, y, period = 24 * 3600, remove_outliers = False, do_residual_tests = False):
        '''
        Fit the multicomponent cosinor to the observed data
        '''

        self.t, self.y = self._parse_t_y(t=t, y=y)


        self.tau = []
        for i_n in range(1, self.N_components+1):
            self.tau.append(period/i_n)

        # ---------------------------- Computing S ---------------------------- #
        # Define the cosine and sine
        cs_vec = []
        sn_vec = []

        x_sc = [np.ones_like(t)]

        for i in range(self.N_components):
            cs_vec.append(np.cos((2 * np.pi / self.tau[i]) * self.t))
            sn_vec.append(np.sin((2 * np.pi / self.tau[i]) * self.t))
            x_sc.append(cs_vec[i])
            x_sc.append(sn_vec[i])


        S = np.zeros((2 * self.N_components + 1, 2 * self.N_components + 1))
        for i in range(len(x_sc)):
            row = []
            for j in range(len(x_sc)):
                row.append(np.sum(x_sc[i] * x_sc[j]))
            S[i] = row

        S[0][0] = len(self.y)

        # ---------------------------- Computing d ---------------------------- #

        d = [np.sum(self.y)]
        for i in range(self.N_components):
            d.append(np.sum(self.y * cs_vec[i]))
            d.append(np.sum(self.y * sn_vec[i]))

        # ---------------------------- Fit u = d/S ---------------------------- #
        u = np.linalg.solve(S, d)

        # ----------------------- Computing parameters ------------------------ #

        M = u[0]
        A, phi, beta, gamma = [], [], [], []
        for i in range(self.N_components):
            beta.append(u[2 * i + 1])
            gamma.append(u[2 * i + 2])

        self.set_params(M, beta, gamma)

        if do_residual_tests:
            self._residual_tests()


        ## Check for outliers defined as errors significantly greater than 3 SD
        if remove_outliers:
            # Check if the data fits
            _, p = self.goodness_of_fit()
            if p > self.alpha:
                residuals = self.y - self.transform(self.t)
                ind_ignore = np.abs(residuals - np.mean(residuals)) > 3 * np.std(residuals)
                if np.any(ind_ignore):
                    self.fit(self.t[~ind_ignore], self.y[~ind_ignore], period = period, remove_outliers = False)

    def _parse_t_y(self, t, y):
        # Remove NaN values from y and t
        non_nan_mask = ~np.isnan(y)
        t = t[non_nan_mask]
        y = y[non_nan_mask]
        return t, y


    def set_params(self, M, beta, gamma):
        A, phi = get_A_phi(beta, gamma)
        # Assign
        self.M = M
        self.A = A
        self.phi = phi
        self.beta = beta
        self.gamma = gamma

        # determine orthophase and bathyphase
        t_1000 = np.linspace(0, 24 * 3600, 1000)
        phase_1000 = np.linspace(0, 2 * np.pi, 1000)
        y_est_1000 = self.transform(t_1000)
        i_min = np.argmin(y_est_1000)
        self.ortho = phase_1000[i_min]
        i_max = np.argmax(y_est_1000)
        self.bathy = phase_1000[i_max]

    def transform(self, t):
        '''
        Transform a new observation at a timepoint
        '''
        y_est = np.zeros_like(t) + self.M
        for i_N in range(self.N_components):
            y_est += self._cosinor_func(self.A[i_N], self.phi[i_N], self.tau[i_N], t)
        return y_est

    def fit_tranform(self,t, y, period = 24 * 3600, remove_outliers = False ):
        self.fit(t=t, y=y, period = period, remove_outliers = remove_outliers )
        return self.transform(t)


    def goodness_of_fit(self):
        '''
        Compute the goodness of fit statistic
        '''
        p_num_params = 2 * self.N_components + 1
        y_est = self.transform(self.t)
        num_data_points = len(y_est)
        RSS = np.nansum((self.y - y_est) ** 2)
        MSS = np.nansum((np.nanmean(self.y) - y_est) ** 2)
        F_statistic = (MSS / (p_num_params - 1)) / (RSS / (num_data_points - p_num_params))
        p_zero_amp_rhythm = fdist.pdf(F_statistic, p_num_params - 1, num_data_points - p_num_params)

        return F_statistic, p_zero_amp_rhythm

    def _residual_tests(self):
        residuals = y[non_nan_mask] - y_est[non_nan_mask]
        if len(residuals) >= 4:
            h_normality, p_normality = adtest(residuals)
        else:
            h_normality, p_normality = 1, 1
            h_idependence, p_idependence = runstest(residuals)
            h_mean, p_mean = ztest(residuals, 0, np.std(residuals))
            residual_tests = np.array([p_normality, p_idependence, p_mean])

        # else:
        #     residual_tests = np.zeros(3)

        return residual_tests()


    def plot_cosinor_fit(self, ylab='Value'):
        # Fix the looping t
        sorted_indices = np.argsort(self.t)

        _,p=self.goodness_of_fit()

        fig, ax = plt.subplots()
        y_est = self.transform(self.t)
        ax.scatter(self.t[sorted_indices]/3600, self.y[sorted_indices], c='r', label='Data')
        ax.plot(self.t[sorted_indices]/3600, y_est[sorted_indices], c='k', label=f'Cosinor Fit (p={p:.2f})')
        ax.legend()
        ax.set_xlabel('Time (hrs)')
        ax.set_ylabel(ylab)
        plt.grid()
        plt.show()




def get_A_phi(beta_vec, gamma_vec):
    '''
    Based on the beta and gamma parameters compute the correct A and phi signs
    '''
    N = len(beta_vec)
    A = np.empty((N, 1))
    phi = np.empty((N, 1))

    for n_idx in range(N):
        beta = beta_vec[n_idx]
        gamma = gamma_vec[n_idx]
        A[n_idx] = np.sqrt(beta**2 + gamma**2)

        theta = np.arctan(np.abs(gamma/beta))

        # Calculate acrophase (phi) -- phase to peak
        a = np.sign(beta)
        b = np.sign(gamma)
        if (a == 1 or a == 0) and b == 1:
            phi[n_idx] = -theta
        elif a == -1 and (b == 1 or b == 0):
            phi[n_idx] = -np.pi + theta
        elif (a == -1 or a == 0) and b == -1:
            phi[n_idx] = -np.pi - theta
        elif a == 1 and (b == -1 or b == 0):
            phi[n_idx] = -2*np.pi + theta
        elif a == 0 and b == 0:
            phi[n_idx] = 0

    return A, phi


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('VitalSigns24Hr.csv')
    df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'])


    def get_t_seconds(dt):
        return dt.hour * 3600 + dt.minute * 60 + dt.second


    df['t'] = df.apply(lambda row: get_t_seconds(row['CHARTTIME']), axis=1)
    all_ids = df['ICUSTAY_ID'].unique()
    curr_df = df.loc[df['ICUSTAY_ID'] == all_ids[0]]


    csn = MultiCosinorModel(N_components=2)
    csn.fit(curr_df['t'], curr_df['SBP'], remove_outliers=True)
    csn.goodness_of_fit()
    csn.plot_cosinor_fit(ylab='SBP (mmHg)')