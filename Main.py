import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter,\
                      WeibullFitter,\
                      ExponentialFitter,\
                      LogNormalFitter,\
                      LogLogisticFitter 



def compare_aft_models(dataframe, duration_col, event_col, epsilon=0.00001):
    # Avoid non-positive durations
    dataframe[duration_col] = np.maximum(dataframe[duration_col], epsilon)

    aft_fitters = [
        WeibullAFTFitter(),
        LogNormalAFTFitter(),
        LogLogisticAFTFitter()
        ]

    fig, ax = plt.subplots(figsize=(16, 9))

    for model in aft_fitters:
        model.fit(dataframe, duration_col=duration_col, event_col=event_col)

        label = f"{model.__class__.__name__}"
        
        model.print_summary()
        print('*' * 145)
        
        plt.plot(model.predict_survival_function(dataframe.loc[1]), label=label)

    plt.legend()
    plt.title('Comparison of Models with ATF fitters', fontsize=16)
    plt.xlabel('Tenure', fontsize=14)
    plt.ylabel('Survival Probability', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()


def compare_usual_models(dataframe, duration_col, event_col, epsilon=0.00001):
    # Avoid non-positive durations
    dataframe[duration_col] = np.maximum(dataframe[duration_col], epsilon)

    fitters = [
        WeibullFitter(),
        ExponentialFitter(),
        LogNormalFitter(),
        LogLogisticFitter()
    ]

    fig, ax = plt.subplots(figsize=(16, 9))

    for model in fitters:
        model.fit(durations=dataframe[duration_col], event_observed=dataframe[event_col])

        label = f"{model.__class__.__name__}"
        
        print("The AIC value for", model.__class__.__name__, "is", model.AIC_)
        print("The BIC value for", model.__class__.__name__, "is", model.BIC_)
        
        model.plot_survival_function(ax=ax, label=label)

        model.print_summary()
        print('*' * 145)

    plt.legend()
    plt.title('Comparison of Models with usual fitters', fontsize=16)
    plt.xlabel('Tenure', fontsize=14)
    plt.ylabel('Survival Probability', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()