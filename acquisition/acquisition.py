import pandas as pd


def get_vienna_data():
    vienna_weekdays = pd.read_csv('datasets/vienna_weekdays.csv', sep=',')
    vienna_weekdays['weekend'] = False
    vienna_weekend = pd.read_csv('datasets/vienna_weekends.csv', sep=',')
    vienna_weekend['weekend'] = True
    vienna = pd.concat([vienna_weekend, vienna_weekdays], ignore_index=True, sort=False)
    return vienna
