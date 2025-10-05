import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def create_myplot(data_path):
    data=pd.read_csv(data_path)

    sns.lineplot(data=data, x='list_length', y='accuracy', hue='condition')
    plt.show()