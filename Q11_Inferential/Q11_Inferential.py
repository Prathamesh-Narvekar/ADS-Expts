# pip install pandas numpy scipy statsmodels

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats.api as sms

# Load dataset
df = pd.read_csv("diabetes.csv")

# Select the 'Glucose' column
glucose_data = df['Glucose']

# 1. 95% Confidence Interval for the mean of Glucose
conf_interval = sms.DescrStatsW(glucose_data).tconfint_mean()
print("95% Confidence Interval for Glucose Mean:", conf_interval)

# 2. One-Sample t-test: Is the mean glucose different from 120?
t_stat_one, p_val_one = stats.ttest_1samp(glucose_data, popmean=120)
print("\nOne-Sample t-test (H0: mean = 120):")
print(f"  t-statistic = {t_stat_one:.3f}")
print(f"  p-value     = {p_val_one:.3f}")
if p_val_one < 0.05:
    print("  Result: Reject the null hypothesis (significant difference).")
else:
    print("  Result: Fail to reject the null hypothesis (no significant difference).")

# 3. Two-Sample t-test: Glucose levels for diabetic (1) vs non-diabetic (0)
glucose_0 = df[df['Outcome'] == 0]['Glucose']
glucose_1 = df[df['Outcome'] == 1]['Glucose']
t_stat_two, p_val_two = stats.ttest_ind(glucose_0, glucose_1, equal_var=False)

print("\nTwo-Sample t-test (Glucose: Outcome 0 vs 1):")
print(f"  t-statistic = {t_stat_two:.3f}")
print(f"  p-value     = {p_val_two:.3e}")
if p_val_two < 0.05:
    print("  Result: Significant difference between groups.")
else:
    print("  Result: No significant difference between groups.")



# OR


import numpy as np
import scipy.stats as stats
import statsmodels.stats.api as sms

# Sample data: e.g., weights of people
data = [68, 70, 72, 71, 69, 67, 73, 70, 69, 71]

# Convert to numpy array
data = np.array(data)

# Mean and Standard Deviation
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)  # Sample standard deviation
n = len(data)

print(f"Sample Mean: {sample_mean:.2f}")
print(f"Sample Standard Deviation: {sample_std:.2f}")

# 1. Confidence Interval for Mean (95%)
conf_interval = sms.DescrStatsW(data).tconfint_mean()
print(f"95% Confidence Interval for Mean: {conf_interval}")

# 2. One-sample t-test
# H0: μ = 70 (population mean)
t_stat, p_value = stats.ttest_1samp(data, popmean=70)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < 0.05:
    print("Result: Reject the null hypothesis (significant difference).")
else:
    print("Result: Fail to reject the null hypothesis (no significant difference).")
