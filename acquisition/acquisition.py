import pandas as pd


def test():
    print("acquisition test")


def get_vienna_data():
    vienna_weekdays = pd.read_csv('datasets/vienna_weekdays.csv', sep=',')
    vienna_weekdays['weekend'] = False
    vienna_weekend = pd.read_csv('datasets/vienna_weekends.csv', sep=',')
    vienna_weekend['weekend'] = True
    vienna_weekdays['city'] = 'Vienna'
    vienna_weekend['city'] = 'Vienna'
    vienna = pd.concat([vienna_weekend, vienna_weekdays], ignore_index=True, sort=False)
    return vienna
