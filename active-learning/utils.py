import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def create_plot(n_cols, steps, interval, metrices_names, **kwargs):
    x_steps = np.arange(1, steps+1)
    fig, ax = plt.subplots(len(metrices_names)//n_cols + 1, n_cols, figsize=[20, 10])
    fig.autofmt_xdate(rotation=70)
    
    for title_of_alg, metrices in kwargs.items():
        for i, metric in enumerate(zip(*metrices)):
            ax[i//n_cols][i%n_cols].plot(x_steps*interval, metric, label=title_of_alg)
    for i in range(len(metrices_names)):
        #dolacz informacje o podpisach
        first_idx = i//n_cols
        second_idx = i%n_cols
        ax[first_idx][second_idx].legend()
        ax[first_idx][second_idx].grid(True)
        ax[first_idx][second_idx].set_title(metrices_names[i].upper()+"_TEST")
        ax[first_idx][second_idx].set_xlabel("DATA SIZE")
        ax[first_idx][second_idx].set_ylabel(metrices_names[i])
        ax[first_idx][second_idx].set_xticks(interval*np.arange(1, steps + 1))


def create_validation_data(x_train, y_train):
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42) 
    return (x_train, y_train),(x_validation, y_validation) 
            
##TODO
#Wymysl dodatkowa szosta miare, i popraw poprzednie