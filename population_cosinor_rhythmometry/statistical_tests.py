import math
from scipy.stats import norm, anderson

def andersonDarlingTest(data, distribution='norm'):
    if len(data) <4:
        return 1, 1

    result = anderson(data, dist=distribution)

    statistic = result.statistic
    critical_values = result.critical_values
    significance_levels = result.significance_level

    # Find the critical value corresponding to the Anderson-Darling statistic
    critical_value = None
    for i in range(len(significance_levels)):
        if statistic < critical_values[i]:
            critical_value = critical_values[i]
            break

    # Calculate the p-value
    p_value = significance_levels[i]

    return statistic, p_value

def runsTest(data):
    '''
    Test for randomness in a series.
    '''
    n = len(data)
    median_threshold = sorted(data)[n // 2] if n % 2 != 0 else (sorted(data)[n // 2 - 1] +
                                                                sorted(data)[n // 2]) / 2

    run_count, positive_count, negative_count = 0, 0, 0

    # Evaluate the sequence for the initiation of runs
    for i in range(1, len(data)):
        if (data[i] >= median_threshold and data[i - 1] < median_threshold) or \
                (data[i] < median_threshold and data[i - 1] >= median_threshold):
            run_count += 1

        if data[i] >= median_threshold:
            positive_count += 1
        else:
            negative_count += 1

    expected_runs = ((2 * positive_count * negative_count) / (positive_count + negative_count)) + 1
    standard_deviation = math.sqrt((2 * positive_count * negative_count * (2 * positive_count * negative_count -
                                                                             positive_count - negative_count)) /
                                   (((positive_count + negative_count) ** 2) * (positive_count + negative_count - 1)))

    # Calculate the Z-score
    z_score = (run_count - expected_runs) / standard_deviation

    # Calculate the two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return z_score, p_value


def zTest(data, population_mean, population_std_dev):
    n = len(data)

    sample_mean = sum(data) / n

    z_score = (sample_mean - population_mean) / (population_std_dev / math.sqrt(n))

    # Calculate the two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return z_score, p_value
