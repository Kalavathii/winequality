# winequality
Developed a machine learning model to predict wine quality based on chemical properties using XGBClassifier and Logistic Regression algorithms like scikit learn. Delivered insights on key  quality factors, with applications in quality control and consumer recommendations.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')
