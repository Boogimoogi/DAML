import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro

df = pd.read_csv('motor.csv')

##Q3-a
print (df['Lifespan'].mean())
"""

Average lifespan of the given data is 1008.0 hours.
The mean briefly describes the performance of the given motors.
Also can be used to compare motors work above average or below average.

"""


##Q3-b

plt.figure(figsize=(10, 6))
plt.hist(df['Lifespan'], bins='auto', color='skyblue', edgecolor='black')
plt.title('Distribution of Motor Lifespans')
plt.xlabel('Lifespan (hours)')
plt.ylabel('Counts of motors')
plt.grid(axis='y', alpha=0.75)

plt.show()
"""

"""

##Q3-c
"""
There are 5 motors reached or exceeeded the expected lifespan.

"""

##Q3-d
print (df['Lifespan'].describe())
"""
The standard deviation of lifespan is 42.373996 hours.
The standard deviation is a measure of the amount of variation or dispersion of the set of motors,
which can help further understand which motros are more reliable or less reliable.
"""

##Q3-e
lifespan_data = df["Lifespan"]

#Perform Shapiro-Wilk test
shapiro_result = shapiro(lifespan_data)
#Display the result
shapiro_test_statistic, shapiro_test_p_value = shapiro_result
print(f"Shapiro-Wilk Test Statistic: {shapiro_test_statistic}, P-value: {shapiro_test_p_value}")

"""
Accordiong to the Shapiro-Wilk test, the Static is about 0.917 and the p-value is about 0.3339.
The result shows that the data seems to be normally distributed. However, there are only 10 data given, 
which is not enough to make a critical conclusion that it's normally distributed. 

"""
