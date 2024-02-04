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



def plot_distribution(X):
    """Plots the distribution of X using these 3 types of plots:
            1) Distribution Plot
            2) Box Plot
            3) Probability Plot
       X needs to be a Pandas Series
    """
    print('Feature is of type: ', X.dtype)
    print('5 sample values from this feature:', X.unique()[0:5])
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4));
    sns.histplot(X, ax=ax[0], kde=True, stat="density", kde_kws=dict(cut=3))
    #sns.distplot(X, ax=ax[0]);
    sns.boxplot(X, ax=ax[1]);
    stats.probplot(X, plot=ax[2]);
    plt.show();



def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names


def beep():
    import winsound

    winsound.Beep(2093, 180)
    winsound.Beep(2093, 180)
    winsound.Beep(3136, 180)
    winsound.Beep(3136, 180)
    winsound.Beep(3520, 180)
    winsound.Beep(3520, 180)
    winsound.Beep(3136, 720)
    winsound.Beep(2794, 180)
    winsound.Beep(2794, 180)
    winsound.Beep(2637, 180)
    winsound.Beep(2637, 180)
    winsound.Beep(2349, 180)
    winsound.Beep(2349, 180)
    winsound.Beep(2093, 720)


def classification_report_IDS(y_true, y_prediction):
    '''My version of the sklearn's classification_report function. Instead of the F1 score, it returns the F2 score'''
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