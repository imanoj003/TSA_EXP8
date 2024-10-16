# Ex.No: 08  MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
## Developer name : Manoj M
## Reg no : 212221240027
## Date: 16/10/24


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2.  Load student performance data into a DataFrame and set StudentID as the index.
3. Visualize the final grades against the StudentID using a line plot.
4.  Use ARIMA with order (0, 0, 2) to build the Moving Average model with 2 lag residuals.
5. Predict the next 5 grades using the forecast() method of the fitted model.
6. Plot both the original grades and forecasted values on the same graph for comparison.

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Create the student performance dataset
df = pd.read_csv('/content/student_performance.csv')

df = pd.DataFrame(data)

# Treat StudentID as the time index for simplicity
df.set_index('StudentID', inplace=True)

# Plot the original data
plt.plot(df.index, df['FinalGrade'], marker='o')
plt.title('Final Grades Over Student IDs')
plt.xlabel('Student ID')
plt.ylabel('Final Grade')
plt.grid(True)
plt.show()


# Fit ARIMA model with MA(q=2) component
ma_model = ARIMA(df['FinalGrade'], order=(0, 0, 2)).fit()

# Predict the next 5 final grades using the MA model
ma_predictions = ma_model.forecast(steps=5)

print("Moving Average Model Predictions for next 5 students:\n", ma_predictions)

# Plot the original data and MA predictions
plt.plot(df.index, df['FinalGrade'], label='Original Final Grades', marker='o')
plt.plot(range(len(df) + 1, len(df) + 6), ma_predictions, label='MA Forecasted Grades', marker='x')
plt.title('Moving Average Model Forecast')
plt.xlabel('Student ID')
plt.ylabel('Final Grade')
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT:

# Final Grade

![Untitled](https://github.com/user-attachments/assets/1a7d8270-1962-421e-bb02-24322642dadd)
```
Moving Average Model Predictions for next 5 students:
 10    82.468068
11    80.056671
12    80.069559
13    80.069559
14    80.069559
```
Name: predicted_mean, dtype: float64
# Moving Average Model Forecast
![Untitled](https://github.com/user-attachments/assets/90f949f1-6f0c-4867-b73f-f0ad8a701977)


### RESULT:
Thus the have successfully implemented the Moving Average Model and Exponential smoothing using python.
