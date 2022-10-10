import numpy as np
import os
from parser_csv import parser_csv_files
import feature_extraction
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from pandas_profiling import ProfileReport


def plot_history_predict(probs):
    fig, axes = plt.subplots()

    fig.set_figwidth(10)
    fig.set_figheight(10)

    axes.scatter(range(6, len(probs) + 6), probs)

    fig.suptitle('Все признаки + файлы разделенные на 2 части)', fontsize=25)

    axes.set_title('Сlassifier prediction', fontsize=20)
    axes.set(yticks=[0, 1, 2])
    axes.set_yticklabels(['normal', 'overfit', 'underfit'], fontsize=15)
    axes.set_xlabel('epoch', fontsize=15)

    plt.savefig(r'res\15_predict_3_19.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    # # OLD FEATURES
    # class_names = ['normal', 'overfitting', 'underfitting']
    # acc_feature = []
    # loss_feature = []
    # for class_name in class_names:
    #     # extracting and saving features
    #     acc_data = parser_csv_files(os.path.join(r'src\acc_double', class_name))
    #     for file_name in acc_data.keys():
    #         train_val_data = acc_data[file_name]
    #         acc_feature.append([*feature_extraction.acc_all_features(train_val_data), class_names.index(class_name)])
    #     loss_data = parser_csv_files(os.path.join(r'src\loss_double', class_name))
    #     for file_name in loss_data.keys():
    #         train_val_data = loss_data[file_name]
    #         loss_feature.append([*feature_extraction.loss_all_features(train_val_data), class_names.index(class_name)])
    #
    # # converting data to DataFrame format
    # acc_colum_names = feature_extraction.acc_feature_names()
    # acc_colum_names.append('target')
    # loss_colum_names = feature_extraction.loss_feature_names()
    # loss_colum_names.append('target')
    # df_acc = pd.DataFrame(np.array(acc_feature), columns=acc_colum_names)
    # df_loss = pd.DataFrame(np.array(loss_feature), columns=loss_colum_names)


    # # NEW FEATURES
    # class_names = ['normal', 'overfitting', 'underfitting']
    # acc_feature = []
    # loss_feature = []
    # for class_name in class_names:
    #     # extracting and saving features
    #     acc_data = parser_csv_files(os.path.join(r'src\acc_double', class_name))
    #     for file_name in acc_data.keys():
    #         train_val_data = acc_data[file_name]
    #         if len(train_val_data[0]) > 4:
    #             acc_feature.append([*feature_extraction.f_features(train_val_data[0]),
    #                                 *feature_extraction.f_features(train_val_data[1]),
    #                                 class_names.index(class_name)])
    #     loss_data = parser_csv_files(os.path.join(r'src\loss_double', class_name))
    #     for file_name in loss_data.keys():
    #         train_val_data = loss_data[file_name]
    #         if len(train_val_data[0]) > 4:
    #             loss_feature.append([*feature_extraction.f_features(train_val_data[0]),
    #                                  *feature_extraction.f_features(train_val_data[1]),
    #                                  class_names.index(class_name)])
    #
    # # converting data to DataFrame format
    # df_acc = pd.DataFrame(np.array(acc_feature), columns=['F1_train', 'F2_train', 'F3_train',
    #                                                       'F1_test', 'F2_test', 'F3_test',
    #                                                       'target'])
    # df_loss = pd.DataFrame(np.array(loss_feature), columns=['F1_train', 'F2_train', 'F3_train',
    #                                                         'F1_test', 'F2_test', 'F3_test',
    #                                                         'target'])


    # ALL FEATURES
    class_names = ['normal', 'overfitting', 'underfitting']
    acc_feature = []
    loss_feature = []
    for class_name in class_names:
        # extracting and saving features
        acc_data = parser_csv_files(os.path.join(r'src\acc_all', class_name))
        for file_name in acc_data.keys():
            train_val_data = acc_data[file_name]
            acc_feature.append([*feature_extraction.acc_all_features(train_val_data),
                                *feature_extraction.f_features(train_val_data),
                                class_names.index(class_name)])
        loss_data = parser_csv_files(os.path.join(r'src\loss_all', class_name))
        for file_name in loss_data.keys():
            train_val_data = loss_data[file_name]
            loss_feature.append([*feature_extraction.loss_all_features(train_val_data),
                                 *feature_extraction.f_features(train_val_data),
                                 class_names.index(class_name)])

    # converting data to DataFrame format
    acc_colum_names = feature_extraction.acc_feature_names()
    loss_colum_names = feature_extraction.loss_feature_names()
    df_acc = pd.DataFrame(np.array(acc_feature), columns=acc_colum_names)
    df_loss = pd.DataFrame(np.array(loss_feature), columns=loss_colum_names)




    # saving data
    # df_acc.to_csv(r'src\df_acc.csv')
    # df_loss.to_csv(r'src\df_loss.csv')

    # normalization data
    # Scale only columns that have values greater than 1
    # to_scale_acc = [col for col in df_acc.columns if df_acc[col].max() > 1 and col != 'target']
    # mms_acc = MinMaxScaler()
    # scaled_acc = mms_acc.fit_transform(df_acc[to_scale_acc])
    # scaled_acc = pd.DataFrame(scaled_acc, columns=to_scale_acc)
    #
    # # Replace original columns with scaled ones
    # for col in scaled_acc:
    #     df_acc[col] = scaled_acc[col]
    #
    # # Scale only columns that have values greater than 1
    # to_scale_loss = [col for col in df_loss.columns if df_loss[col].max() > 1 and col != 'target']
    # mms_loss = MinMaxScaler()
    # scaled_loss = mms_loss.fit_transform(df_loss[to_scale_loss])
    # scaled_loss = pd.DataFrame(scaled_loss, columns=to_scale_loss)
    #
    # # Replace original columns with scaled ones
    # for col in scaled_loss:
    #     df_loss[col] = scaled_loss[col]

    # saving data
    # df_acc.to_csv(r'src\df_acc_norm.csv')
    # df_loss.to_csv(r'src\df_loss_norm.csv')



    # remove anomalies
    # cols = df_acc.columns
    # for col in cols:
    #     q1 = df_acc[col].quantile(0.25)
    #     q3 = df_acc[col].quantile(0.75)
    #     iqr = q3 - q1
    #     # df_acc = df_acc[(df_acc[col] > (q1 - 1.5 * iqr)) & (df_acc[col] < (q3 + 1.5 * iqr))]
    #     df_acc = df_acc[(df_acc[col] > (q1 - 3 * iqr)) & (df_acc[col] < (q3 + 3 * iqr))]

    # SMOTE acc
    x_acc = df_acc.drop('target', axis=1)
    y_acc = df_acc['target']
    sm = SMOTE(random_state=42)
    x_sm_acc, y_sm_acc = sm.fit_resample(x_acc, y_acc)

    # report acc
    # profile_acc = ProfileReport(x_sm_acc, title="Pandas Profiling Report Acc", explorative=True)
    # profile_acc.to_file("report_acc_2.html")


    # # remove anomalies
    # cols = df_loss.columns
    # for col in cols:
    #     q1 = df_loss[col].quantile(0.25)
    #     q3 = df_loss[col].quantile(0.75)
    #     iqr = q3 - q1
    #     # df_loss = df_loss[(df_loss[col] > (q1 - 1.5 * iqr)) & (df_loss[col] < (q3 + 1.5 * iqr))]
    #     df_loss = df_loss[(df_loss[col] > (q1 - 3 * iqr)) & (df_loss[col] < (q3 + 3 * iqr))]

    # SMOTE loss
    x_loss = df_loss.drop('target', axis=1)
    y_loss = df_loss['target']
    sm = SMOTE(random_state=42)
    x_sm_loss, y_sm_loss = sm.fit_resample(x_loss, y_loss)

    # report loss
    # profile_loss = ProfileReport(x_sm_loss, title="Pandas Profiling Report Loss", explorative=True)
    # profile_loss.to_file("report_loss_2.html")



    # plotting correlation matrix
    # ax = sns.heatmap(x_loss.corr(), annot=False, linewidths=1)
    # plt.savefig('corr_matrix_loss.png')

    # training a Decision Tree Classifier acc
    # dt_clf_acc = DecisionTreeClassifier(random_state=17)
    # cv_score_acc = cross_val_score(dt_clf_acc, x_sm_acc, y_sm_acc, cv=5, scoring='accuracy')
    # print('Mean accuracy for acc clf: ', sum(cv_score_acc) / len(cv_score_acc))
    # cv_predict_acc = cross_val_predict(dt_clf_acc, x_sm_acc, y_sm_acc, cv=5)
    # print('Confusion matrix for acc clf:')
    # print(confusion_matrix(y_sm_acc, cv_predict_acc))
    # print('Precision for acc clf: ', end='')
    # print(precision_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('Recall for acc clf: ', end='')
    # print(recall_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('F1 score for acc clf: ', end='')
    # print(f1_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print()
    #
    # # training a Decision Tree Classifier loss
    # dt_clf_loss = DecisionTreeClassifier(random_state=17)
    # cv_score_loss = cross_val_score(dt_clf_loss, x_sm_loss, y_sm_loss, cv=5, scoring='accuracy')
    # print('Mean accuracy for loss clf: ', sum(cv_score_loss) / len(cv_score_loss))
    # cv_predict_loss = cross_val_predict(dt_clf_loss, x_sm_loss, y_sm_loss, cv=5)
    # print('Confusion matrix for loss clf:')
    # print(confusion_matrix(y_sm_loss, cv_predict_loss))
    # print('Precision for loss clf: ', end='')
    # print(precision_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('Recall for loss clf: ', end='')
    # print(recall_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('F1 score for loss clf: ', end='')
    # print(f1_score(y_sm_loss, cv_predict_loss, average='weighted'))

    # dt_clf_acc.fit(x_sm_acc.values, y_sm_acc)
    # for sample, target in zip(x_acc.iterrows(), y_acc):
    #     print(dt_clf_acc.predict([list(sample[1])]))
    #     print(target)

    # dt_clf_loss.fit(x_sm_loss.values, y_sm_loss)
    # for sample, target in zip(x_loss.iterrows(), y_loss):
    #     if dt_clf_loss.predict([list(sample[1])])[0] != target:
    #         print(1)

    # # training a SVC acc
    # svc_clf_acc = SVC()
    # cv_score_acc = cross_val_score(svc_clf_acc, x_sm_acc, y_sm_acc, cv=5, scoring='accuracy')
    # print('Mean accuracy for acc clf: ', sum(cv_score_acc) / len(cv_score_acc))
    # cv_predict_acc = cross_val_predict(svc_clf_acc, x_sm_acc, y_sm_acc, cv=5)
    # print('Confusion matrix for acc clf:')
    # print(confusion_matrix(y_sm_acc, cv_predict_acc))
    # print('Precision for acc clf: ', end='')
    # print(precision_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('Recall for acc clf: ', end='')
    # print(recall_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('F1 score for acc clf: ', end='')
    # print(f1_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print()
    #
    # # training a SVC loss
    # svc_clf_loss = SVC()
    # cv_score_loss = cross_val_score(svc_clf_loss, x_sm_loss, y_sm_loss, cv=5, scoring='accuracy')
    # print('Mean accuracy for loss clf: ', sum(cv_score_loss) / len(cv_score_loss))
    # cv_predict_loss = cross_val_predict(svc_clf_loss, x_sm_loss, y_sm_loss, cv=5)
    # print('Confusion matrix for loss clf:')
    # print(confusion_matrix(y_sm_loss, cv_predict_loss))
    # print('Precision for loss clf: ', end='')
    # print(precision_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('Recall for loss clf: ', end='')
    # print(recall_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('F1 score for loss clf: ', end='')
    # print(f1_score(y_sm_loss, cv_predict_loss, average='weighted'))

    # training a SVC with polynomial Features acc
    # polynomial_svc_clf_acc = Pipeline([
    #     ('poly_feature', PolynomialFeatures(degree=3)),
    #     ('scaler', StandardScaler()),
    #     ('svm_clf', LinearSVC(C=10, loss='hinge'))
    # ])
    # cv_score_acc = cross_val_score(polynomial_svc_clf_acc, x_sm_acc, y_sm_acc, cv=5, scoring='accuracy')
    # print('Mean accuracy for acc clf: ', sum(cv_score_acc) / len(cv_score_acc))
    # cv_predict_acc = cross_val_predict(polynomial_svc_clf_acc, x_sm_acc, y_sm_acc, cv=5)
    # print('Confusion matrix for acc clf:')
    # print(confusion_matrix(y_sm_acc, cv_predict_acc))
    # print('Precision for acc clf: ', end='')
    # print(precision_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('Recall for acc clf: ', end='')
    # print(recall_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('F1 score for acc clf: ', end='')
    # print(f1_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print()
    #
    # # training a SVC with polynomial Features loss
    # polynomial_svc_clf_loss = Pipeline([
    #     ('poly_feature', PolynomialFeatures(degree=3)),
    #     ('scaler', StandardScaler()),
    #     ('svm_clf', LinearSVC(C=10, loss='hinge'))
    # ])
    # cv_score_loss = cross_val_score(polynomial_svc_clf_loss, x_sm_loss, y_sm_loss, cv=5, scoring='accuracy')
    # print('Mean accuracy for loss clf: ', sum(cv_score_loss) / len(cv_score_loss))
    # cv_predict_loss = cross_val_predict(polynomial_svc_clf_loss, x_sm_loss, y_sm_loss, cv=5)
    # print('Confusion matrix for loss clf:')
    # print(confusion_matrix(y_sm_loss, cv_predict_loss))
    # print('Precision for loss clf: ', end='')
    # print(precision_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('Recall for loss clf: ', end='')
    # print(recall_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('F1 score for loss clf: ', end='')
    # print(f1_score(y_sm_loss, cv_predict_loss, average='weighted'))

    # svc_clf_acc = Pipeline([
    #     ('poly_feature', PolynomialFeatures(degree=3)),
    #     ('svm_clf', SVC())
    # ])
    # cv_score_acc = cross_val_score(svc_clf_acc, x_sm_acc, y_sm_acc, cv=5, scoring='accuracy')
    # print('Mean accuracy for acc clf: ', sum(cv_score_acc) / len(cv_score_acc))
    # cv_predict_acc = cross_val_predict(svc_clf_acc, x_sm_acc, y_sm_acc, cv=5)
    # print('Confusion matrix for acc clf:')
    # print(confusion_matrix(y_sm_acc, cv_predict_acc))
    # print('Precision for acc clf: ', end='')
    # print(precision_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('Recall for acc clf: ', end='')
    # print(recall_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('F1 score for acc clf: ', end='')
    # print(f1_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print()
    #
    # # training a SVC with polynomial Features loss
    # svc_clf_loss = Pipeline([
    #     ('poly_feature', PolynomialFeatures(degree=3)),
    #     ('svm_clf', SVC())
    # ])
    # cv_score_loss = cross_val_score(svc_clf_loss, x_sm_loss, y_sm_loss, cv=5, scoring='accuracy')
    # print('Mean accuracy for loss clf: ', sum(cv_score_loss) / len(cv_score_loss))
    # cv_predict_loss = cross_val_predict(svc_clf_loss, x_sm_loss, y_sm_loss, cv=5)
    # print('Confusion matrix for loss clf:')
    # print(confusion_matrix(y_sm_loss, cv_predict_loss))
    # print('Precision for loss clf: ', end='')
    # print(precision_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('Recall for loss clf: ', end='')
    # print(recall_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('F1 score for loss clf: ', end='')
    # print(f1_score(y_sm_loss, cv_predict_loss, average='weighted'))

    # svc_clf_acc = KNeighborsClassifier()
    # cv_score_acc = cross_val_score(svc_clf_acc, x_sm_acc, y_sm_acc, cv=5, scoring='accuracy')
    # print('Mean accuracy for acc clf: ', sum(cv_score_acc) / len(cv_score_acc))
    # cv_predict_acc = cross_val_predict(svc_clf_acc, x_sm_acc, y_sm_acc, cv=5)
    # print('Confusion matrix for acc clf:')
    # print(confusion_matrix(y_sm_acc, cv_predict_acc))
    # print('Precision for acc clf: ', end='')
    # print(precision_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('Recall for acc clf: ', end='')
    # print(recall_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('F1 score for acc clf: ', end='')
    # print(f1_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print()
    #
    # # training a KNeighborsClassifier loss
    # svc_clf_loss = KNeighborsClassifier()
    # cv_score_loss = cross_val_score(svc_clf_loss, x_sm_loss, y_sm_loss, cv=5, scoring='accuracy')
    # print('Mean accuracy for loss clf: ', sum(cv_score_loss) / len(cv_score_loss))
    # cv_predict_loss = cross_val_predict(svc_clf_loss, x_sm_loss, y_sm_loss, cv=5)
    # print('Confusion matrix for loss clf:')
    # print(confusion_matrix(y_sm_loss, cv_predict_loss))
    # print('Precision for loss clf: ', end='')
    # print(precision_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('Recall for loss clf: ', end='')
    # print(recall_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('F1 score for loss clf: ', end='')
    # print(f1_score(y_sm_loss, cv_predict_loss, average='weighted'))

    # lg_clf_acc = LogisticRegression()
    # cv_score_acc = cross_val_score(lg_clf_acc, x_sm_acc, y_sm_acc, cv=5, scoring='accuracy')
    # print('Mean accuracy for acc clf: ', sum(cv_score_acc) / len(cv_score_acc))
    # cv_predict_acc = cross_val_predict(lg_clf_acc, x_sm_acc, y_sm_acc, cv=5)
    # print('Confusion matrix for acc clf:')
    # print(confusion_matrix(y_sm_acc, cv_predict_acc))
    # print('Precision for acc clf: ', end='')
    # print(precision_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('Recall for acc clf: ', end='')
    # print(recall_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('F1 score for acc clf: ', end='')
    # print(f1_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print()
    #
    # # training a KNeighborsClassifier loss
    # lg_clf_loss = LogisticRegression()
    # cv_score_loss = cross_val_score(lg_clf_loss, x_sm_loss, y_sm_loss, cv=5, scoring='accuracy')
    # print('Mean accuracy for loss clf: ', sum(cv_score_loss) / len(cv_score_loss))
    # cv_predict_loss = cross_val_predict(lg_clf_loss, x_sm_loss, y_sm_loss, cv=5)
    # print('Confusion matrix for loss clf:')
    # print(confusion_matrix(y_sm_loss, cv_predict_loss))
    # print('Precision for loss clf: ', end='')
    # print(precision_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('Recall for loss clf: ', end='')
    # print(recall_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('F1 score for loss clf: ', end='')
    # print(f1_score(y_sm_loss, cv_predict_loss, average='weighted'))

    # gb_clf_acc = svc_clf_loss = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('svm_clf', GradientBoostingClassifier())
    # ])
    # cv_score_acc = cross_val_score(gb_clf_acc, x_sm_acc, y_sm_acc, cv=5, scoring='accuracy')
    # print('Mean accuracy for acc clf: ', sum(cv_score_acc) / len(cv_score_acc))
    # cv_predict_acc = cross_val_predict(gb_clf_acc, x_sm_acc, y_sm_acc, cv=5)
    # print('Confusion matrix for acc clf:')
    # print(confusion_matrix(y_sm_acc, cv_predict_acc))
    # print('Precision for acc clf: ', end='')
    # print(precision_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('Recall for acc clf: ', end='')
    # print(recall_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print('F1 score for acc clf: ', end='')
    # print(f1_score(y_sm_acc, cv_predict_acc, average='weighted'))
    # print()
    #
    # # training a GradientBoostingClassifier loss
    # gb_clf_loss = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('svm_clf', GradientBoostingClassifier())
    # ])
    # cv_score_loss = cross_val_score(gb_clf_loss, x_sm_loss, y_sm_loss, cv=5, scoring='accuracy')
    # print('Mean accuracy for loss clf: ', sum(cv_score_loss) / len(cv_score_loss))
    # cv_predict_loss = cross_val_predict(gb_clf_loss, x_sm_loss, y_sm_loss, cv=5)
    # print('Confusion matrix for loss clf:')
    # print(confusion_matrix(y_sm_loss, cv_predict_loss))
    # print('Precision for loss clf: ', end='')
    # print(precision_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('Recall for loss clf: ', end='')
    # print(recall_score(y_sm_loss, cv_predict_loss, average='weighted'))
    # print('F1 score for loss clf: ', end='')
    # print(f1_score(y_sm_loss, cv_predict_loss, average='weighted'))


    rf_clf_acc = svc_clf_loss = Pipeline([
        # ('scaler', StandardScaler()),
        ('svm_clf', RandomForestClassifier())
    ])
    cv_score_acc = cross_val_score(rf_clf_acc, x_sm_acc, y_sm_acc, cv=5, scoring='accuracy')
    print('Mean accuracy for acc clf: ', sum(cv_score_acc) / len(cv_score_acc))
    cv_predict_acc = cross_val_predict(rf_clf_acc, x_sm_acc, y_sm_acc, cv=5)
    print('Confusion matrix for acc clf:')
    print(confusion_matrix(y_sm_acc, cv_predict_acc))
    print('Precision for acc clf: ', end='')
    print(precision_score(y_sm_acc, cv_predict_acc, average='weighted'))
    print('Recall for acc clf: ', end='')
    print(recall_score(y_sm_acc, cv_predict_acc, average='weighted'))
    print('F1 score for acc clf: ', end='')
    print(f1_score(y_sm_acc, cv_predict_acc, average='weighted'))
    print()

    # training a GradientBoostingClassifier loss
    rf_clf_loss = Pipeline([
        # ('scaler', StandardScaler()),
        ('svm_clf', RandomForestClassifier())
    ])
    cv_score_loss = cross_val_score(rf_clf_loss, x_sm_loss, y_sm_loss, cv=5, scoring='accuracy')
    print('Mean accuracy for loss clf: ', sum(cv_score_loss) / len(cv_score_loss))
    cv_predict_loss = cross_val_predict(rf_clf_loss, x_sm_loss, y_sm_loss, cv=5)
    print('Confusion matrix for loss clf:')
    print(confusion_matrix(y_sm_loss, cv_predict_loss))
    print('Precision for loss clf: ', end='')
    print(precision_score(y_sm_loss, cv_predict_loss, average='weighted'))
    print('Recall for loss clf: ', end='')
    print(recall_score(y_sm_loss, cv_predict_loss, average='weighted'))
    print('F1 score for loss clf: ', end='')
    print(f1_score(y_sm_loss, cv_predict_loss, average='weighted'))

    rf_clf_acc = RandomForestClassifier()
    rf_clf_acc.fit(x_sm_acc, y_sm_acc)

    with open('rf_clf_acc_main.pickle', 'wb') as f:
        pickle.dump(rf_clf_acc, f)

    rf_clf_loss = RandomForestClassifier()
    rf_clf_loss.fit(x_sm_loss, y_sm_loss)

    with open('rf_clf_loss_main.pickle', 'wb') as f:
        pickle.dump(rf_clf_loss, f)


    # stds_acc = StandardScaler()
    # stds_acc.fit_transform(x_sm_acc)
    #
    # stds_loss = StandardScaler()
    # stds_loss.fit_transform(x_sm_loss)


    # with open('rf_clf_loss_all_f_files_2.pickle', 'rb') as f:
    #     clf_loss = pickle.load(f)
    # with open('rf_clf_acc_all_f_files_2.pickle', 'rb') as f:
    #     clf_acc = pickle.load(f)
    # with open(r'res\history_3_19.pickle', 'rb') as f:
    #     history = pickle.load(f)
    #
    # data_acc = [history['accuracy'], history['val_accuracy']]
    # data_loss = [history['loss'], history['val_loss']]
    #
    # res_acc_predict = []
    # res_proba = []
    # for epoch in range(5, len(data_acc[0])):
    #     # # OLD FEATURES
    #     # features_acc = feature_extraction.acc_all_features([data_acc[0][:epoch], data_acc[1][:epoch]])
    #     # features_loss = feature_extraction.loss_all_features([data_loss[0][:epoch], data_loss[1][:epoch]])
    #
    #     # # NEW FEATURES
    #     # features_acc = [*feature_extraction.f_features(data_acc[0][:epoch]),
    #     #                 *feature_extraction.f_features(data_acc[1][:epoch])]
    #     #
    #     # features_loss = [*feature_extraction.f_features(data_loss[0][:epoch]),
    #     #                  *feature_extraction.f_features(data_loss[1][:epoch])]
    #
    #     # ALL FEATURES
    #     features_acc = [*feature_extraction.acc_all_features([data_acc[0][:epoch], data_acc[1][:epoch]]),
    #                     *feature_extraction.f_features(data_acc[:epoch]),
    #                     *feature_extraction.f_features(data_acc[:epoch])]
    #
    #     features_loss = [*feature_extraction.loss_all_features([data_loss[0][:epoch], data_loss[1][:epoch]]),
    #                      *feature_extraction.f_features(data_loss[:epoch]),
    #                      *feature_extraction.f_features(data_loss[:epoch])]
    #
    #     prob_acc = clf_acc.predict_proba([features_acc])
    #     prob_loss = clf_loss.predict_proba([features_loss])
    #
    #     sum_probs = sum(prob_acc, prob_loss)[0]
    #     res_class = np.argmax(sum_probs)
    #     res_proba.append(res_class)
    #
    #     # res_acc_predict.append(*clf_acc.predict([features_acc]))
    #     # res_loss_predict.append(*clf_loss.predict([features_loss]))
    #
    # plot_history_predict(res_proba)
