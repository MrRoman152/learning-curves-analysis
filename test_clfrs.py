import numpy as np
import feature_extraction
import pickle
import matplotlib.pyplot as plt
import os


def plot_history_predict(probs, history_train, name_file):
    fig, axes = plt.subplots(1, 3)

    fig.set_figwidth(30)
    fig.set_figheight(10)

    axes[0].scatter(range(6, len(probs) + 6), probs)
    axes[0].set_title('Сlassifier prediction', fontsize=20)
    axes[0].set(yticks=[0, 1, 2])
    axes[0].set_yticklabels(['normal', 'overfit', 'underfit'], fontsize=15)
    axes[0].set_xlabel('epoch', fontsize=15)

    # axes[1].plot(history_train['mae'], label='accuracy')
    # axes[1].plot(history_train['val_mae'], label='val_accuracy', linestyle='--')
    axes[1].plot(history_train['accuracy'], label='accuracy')
    axes[1].plot(history_train['val_accuracy'], label='val_accuracy', linestyle='--')
    axes[1].legend(loc='lower right')
    axes[1].set_title('Accuracy', fontsize=20)
    axes[1].set_xlabel('epoch', fontsize=15)

    axes[2].plot(history_train['loss'], label='loss')
    axes[2].plot(history_train['val_loss'], label='val_loss', linestyle='--')
    axes[2].legend(loc='upper right')
    axes[2].set_title('Loss', fontsize=20)
    axes[2].set_xlabel('epoch', fontsize=15)

    plt.savefig(rf'history\res\{name_file}.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    with open('rf_clf_loss_main.pickle', 'rb') as f:
        clf_loss = pickle.load(f)
    with open('rf_clf_acc_main.pickle', 'rb') as f:
        clf_acc = pickle.load(f)

    file_name = None
    dir_name = r'history\16'
    file_names = os.listdir(dir_name)
    for file_name in file_names:
        if file_name[-7:] == '.pickle':
            with open(os.path.join(dir_name, file_name), 'rb') as f:
                history = pickle.load(f)

                print(history.keys())
                # data_acc = [history['mae'], history['val_mae']]
                data_acc = [history['accuracy'], history['val_accuracy']]
                data_loss = [history['loss'], history['val_loss']]

                res_acc_predict = []
                res_proba = []
                for epoch in range(5, len(data_acc[0])):
                    features_acc = [*feature_extraction.acc_all_features([data_acc[0][:epoch], data_acc[1][:epoch]]),
                                    *feature_extraction.f_features(data_acc[:epoch])]

                    features_loss = [*feature_extraction.loss_all_features([data_loss[0][:epoch], data_loss[1][:epoch]]),
                                     *feature_extraction.f_features(data_loss[:epoch])]

                    prob_acc = clf_acc.predict_proba([features_acc])
                    prob_loss = clf_loss.predict_proba([features_loss])

                    sum_probs = sum(prob_acc, prob_loss)[0]
                    res_class = np.argmax(sum_probs)
                    res_proba.append(res_class)

                plot_history_predict(res_proba, history, file_name)
