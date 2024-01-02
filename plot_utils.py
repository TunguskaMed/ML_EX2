from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import os
import warnings
import numpy as np
import datetime
from typing import List
from sklearn.decomposition import PCA
import plotly.express as px

def plot_confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray, model_name:str,accuracy:float,plot_save_path:str,plot_show:bool,plot_save:bool,labels_name:List[str]=None,labels_list:List = None,cf_matrix_filename:str = ""):
    """
    Plot the confusion matrix for a classification model.

    Args:
    y_true(np.ndarray): Ground truth labels.
    y_pred(np.ndarray): Predicted labels.
    labels_name (List[str]): List of label names.
    model_name (str): Name of the model for display purposes.
    accuracy (float): Accuracy score of the model.
    plot_save_path (str): Path to save the confusion matrix plot.
    plot_show (bool): Whether to display the confusion matrix plot.
    plot_save (bool): Whether to save the confusion matrix plot.

    Raises:
    ValueError: If plot_save is set to True but plot_save_path is not specified.
    ValueError: If y_pred or y_true are not NumPy array.

    """
    if not isinstance(y_pred, np.ndarray):
            raise TypeError("y_pred must be a NumPy array.")
    if not isinstance(y_true, np.ndarray):
            raise TypeError("y_true must be a NumPy array.")
    
    if plot_show == None and plot_save == None:
        warnings.warn("Called 'plot_confusion_matrix' but plot_show and plot_save are None",RuntimeWarning)
        return
    if plot_save and plot_save_path is None:
        raise ValueError("If plot_save is set to True, plot_save_path must be specified.")
    if plot_save is None and plot_save_path:
        warnings.warn("If plot_save is set to False, plot_save_path should not be specified.",RuntimeWarning)
    
    cm = confusion_matrix(y_true, y_pred,labels=labels_list,normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
    
    # Set the figure size in inches for 720p resolution
    fig, ax = plt.subplots(figsize=(1280/100, 720/100), dpi=100)
    
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set(
        title='Confusion Matrix ' + model_name,
        xlabel='Predicted',
        ylabel='Actual '
    )
    plt.xticks(rotation=45)
    plt.grid(False)
    
    # Save the figure in the current working directory with 720p resolution
    if plot_save:
        plt.savefig(os.path.join(plot_save_path, "confusion_matrix" + cf_matrix_filename + str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")) + ".jpg"), dpi=100)
    if plot_show:
        plt.show()
    plt.clf()
    plt.close()

def plot_samples_3d_gpu(X, y,normalization_method,plot_save_path=None, marker_size=5,plot_show = True, plot_save = False):
    
    if not isinstance(X, np.ndarray):
            raise TypeError("y_pred must be a NumPy array.")
    if not isinstance(y, np.ndarray):
            raise TypeError("y_true must be a NumPy array.")
    
    if plot_show == None and plot_save == None:
        warnings.warn("Called 'plot_samples_3d_gpu' but plot_show and plot_save are None",RuntimeWarning)
        return
    if plot_save and plot_save_path is None:
        raise ValueError("If plot_save is set to True, plot_save_path must be specified.")
    if plot_save is None and plot_save_path:
        warnings.warn("If plot_save is set to False, plot_save_path should not be specified.",RuntimeWarning)
    
    pca = PCA(n_components=3)
    X = pca.fit_transform(normalization_method.fit_transform(X))#x_train 

    y = np.reshape(y,(y.shape[0],1))
    combined_data = np.concatenate([X, y], axis=1)

    # Create a pandas DataFrame
    column_names = ['feature_1', 'feature_2', 'feature_3', 'label']
    df = pd.DataFrame(combined_data, columns=column_names)
    df["label"] = df["label"].astype(str)
    fig = px.scatter_3d(df, x='feature_1', y='feature_2', z='feature_3',
              color='label')
    fig.update_traces(marker=dict(size=3,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    
    fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 30),itemsizing = 'constant'))

    if plot_show:
        fig.show()
    if plot_save:
        fig.write_html(plot_save_path)



