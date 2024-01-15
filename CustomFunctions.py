import warnings
from IPython.display import display
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time


def printLarge(a):
    #Capture the current max columns/rows settings
    maxCol = pd.get_option("display.max_columns")
    maxRow = pd.get_option("display.max_rows")
    precision = pd.get_option("display.precision")

    #Display the dataframe
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_rows", 2000)
    pd.set_option('display.precision', 2)
    display(a)

    #Reset the options
    pd.set_option("display.max_columns", maxCol)
    pd.set_option("display.max_rows", maxRow)
    pd.set_option('display.precision', precision)


def test_train_comparison(y_train, pred_train, y_test, pred_test):
    #Compare all classifications between train, test and predictions
    a = np.unique(pred_train, return_counts=True)
    a = pd.DataFrame(a[1], index=a[0], columns=['Freq'])
    b = np.unique(pred_test, return_counts=True)
    b = pd.DataFrame(b[1], index=b[0], columns=['Freq'])
    df = pd.DataFrame([])
    df['y_train'] = y_train.value_counts()
    df['pred_train'] = a
    df['error_train'] = df['y_train'] - df['pred_train']
    df['y_test'] = y_test.value_counts()
    df['pred_test'] = b
    df['error_test'] = df['y_test'] - df['pred_test']
    display(df.style.format('{:,}').background_gradient())


def classification_report_IDS(y_true, y_prediction):
    '''My version of the sklearn's classification_report function. Instead of the F1 score, it returns the F2 score 
    and Matthews Correlation Coefficient (MCC)'''
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import fbeta_score
        
    #Calculate all metrics
    p, r, f2, s = precision_recall_fscore_support(y_true, y_prediction, beta=2.0, average=None)
    d = {'precision': p.round(2), 'recall': r.round(2), 'f2-score':f2.round(2), 'support':s}
    classLabels = np.sort(y_true.unique())
    t = pd.DataFrame(d, index=classLabels)
    t['support'] = t['support'].map('{:,.0f}'.format)
    m = matthews_corrcoef(y_true, y_prediction)
    
    #Print results
    print(t)
    print(f'\nMatthews Correlation Coefficient: {m: .2f}')
    print(f'F2 Macro score: {np.average(f2): .1%}')


class TimerError(Exception):
    '''A custom exception for reporting errors of the Timer class'''
    
class Timer:
    def __init__(self):
        self._start_time = None
        self._start_cpu_time = None


    def start(self):
        """Start a new timer"""
        
        #Init class variables
        self._start_time = time.perf_counter()
        self._start_cpu_time = time.process_time()


    def stop(self, timerName=''):
        """Stop the timer, and report the elapsed time"""

        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        #Take stop measurements
        elapsed_time = time.perf_counter() - self._start_time
        elapsed_cpu_time = time.process_time() - self._start_cpu_time

        if (timerName == ''):
            print(f"Elapsed time: {elapsed_time:0.2f} seconds ({elapsed_cpu_time:0.2f} CPU seconds)")
        else:
            print(f"{timerName} took {elapsed_time:0.2f} seconds ({elapsed_cpu_time:0.2f} CPU seconds)")

        #Restart the timer
        self.start()