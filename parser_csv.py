import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def parser_csv_files(dir_name):
    """
    function to extract training and testing data from csv files
    :param dir_name: string with the name of the directory that contains the csv files
    :return: dictionary - {name_file : [train_data, test_data]}
    """
    file_names = os.listdir(dir_name)
    dict_data = {}
    for file_name in file_names:
        if file_name[-4:] == '.csv':
            with open(os.path.join(dir_name, file_name)) as csvfile:
                csvreader = csv.reader(csvfile)
                train_data = []
                test_data = []
                full_data = []
                for ind_row, row in enumerate(csvreader):
                    # skip column names
                    if ind_row < 2:
                        continue
                    # check for empty data
                    if row[0] == '' or row[1] == '' or row[2] == '' or row[3] == '':
                        break
                    train_data.append(float(row[3]))
                    test_data.append(float(row[1]))

            full_data.append(train_data)
            full_data.append(test_data)
            dict_data[file_name] = full_data

    return dict_data


def data_visualization(dict_data):
    """
    function to plot graphs training and testing data
    :param dict_data: dictionary - {name_file : [train_data, test_data]}
    :return: 0
    """
    for file_name, data in dict_data.items():
        fig, ax = plt.subplots()
        x = range(1, len(data[0]) + 1)
        ax.plot(x, data[0], label='train error')
        ax.plot(x, data[1], label='test error')
        ax.legend()
        fig.savefig(fr'src\pars_graphs\loss\{file_name}.png')
        plt.close()


def division_csv_files(dir_name):
    """
    function to division into 4 parts csv files
    :param dir_name: string with the name of the directory that contains the csv files
    :return: 0
    """
    file_names = os.listdir(dir_name)
    for file_name in file_names:
        if file_name[-4:] == '.csv':
            with open(os.path.join(dir_name, file_name)) as csvfile:
                df = pd.read_csv(csvfile)
                row_count_tick = len(df) // 4
                if row_count_tick < 5:
                    continue
                for i, row in df.iterrows():
                    if i == 0:
                        continue
                    if row[0] is np.NaN or row[1] is np.NaN or row[2] is np.NaN or row[3] is np.NaN:
                        break
                    if i < row_count_tick + 1:
                        with open(os.path.join(dir_name, (file_name[:-4] + '_1.csv')), mode='a', newline='') as wr_file:
                            employee_writer = csv.writer(wr_file, delimiter=',')
                            employee_writer.writerow(row.values)
                    if row_count_tick + 1 <= i < row_count_tick * 2 + 1:
                        with open(os.path.join(dir_name, (file_name[:-4] + '_2.csv')), mode='a', newline='') as wr_file:
                            employee_writer = csv.writer(wr_file, delimiter=',')
                            employee_writer.writerow(row.values)
                    if row_count_tick * 2 + 1 <= i < row_count_tick * 3 + 1:
                        with open(os.path.join(dir_name, (file_name[:-4] + '_3.csv')), mode='a', newline='') as wr_file:
                            employee_writer = csv.writer(wr_file, delimiter=',')
                            employee_writer.writerow(row.values)
                    if row_count_tick * 3 + 1 <= i < row_count_tick * 4 + 1:
                        with open(os.path.join(dir_name, (file_name[:-4] + '_4.csv')), mode='a', newline='') as wr_file:
                            employee_writer = csv.writer(wr_file, delimiter=',')
                            employee_writer.writerow(row.values)


def division_csv_files_2(dir_name):
    """
    function to division into 2 parts csv files
    :param dir_name: string with the name of the directory that contains the csv files
    :return: 0
    """
    file_names = os.listdir(dir_name)
    for file_name in file_names:
        if file_name[-4:] == '.csv':
            with open(os.path.join(dir_name, file_name)) as csvfile:
                df = pd.read_csv(csvfile)
                row_count_tick = len(df) // 2
                if row_count_tick < 5:
                    continue
                for i, row in df.iterrows():
                    if i == 0:
                        continue
                    if row[0] is np.NaN or row[1] is np.NaN or row[2] is np.NaN or row[3] is np.NaN:
                        break
                    if i < row_count_tick + 1:
                        with open(os.path.join(dir_name, (file_name[:-4] + '_1.csv')), mode='a', newline='') as wr_file:
                            employee_writer = csv.writer(wr_file, delimiter=',')
                            employee_writer.writerow(row.values)
                    else:
                        with open(os.path.join(dir_name, (file_name[:-4] + '_2.csv')), mode='a', newline='') as wr_file:
                            employee_writer = csv.writer(wr_file, delimiter=',')
                            employee_writer.writerow(row.values)
