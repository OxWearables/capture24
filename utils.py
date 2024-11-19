from datetime import datetime, timedelta, time
from pandas.plotting import register_matplotlib_converters
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML
import numpy as np
import sklearn.metrics as metrics
import os
import pickle
import sys


def check_files_exist(dir, files):
    """ Ensure all expected files exist in dir """
    return all(os.path.exists(os.path.join(dir, file)) for file in files)


def ordered_unique(x):
    """ Return unique elements, maintaining order of appearance """
    return x[np.sort(np.unique(x, return_index=True)[1])]


def save_dict(dict, savefile):
    """ Save dict file as .pkl """
    file_extension = savefile.rsplit(".", 1)[-1].lower()

    if file_extension == "pkl":
        dirname = os.path.dirname(savefile)

        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(savefile, 'wb') as file:
            pickle.dump(dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        raise ValueError("Unsupported file type. Only '.pkl' is supported.")


def load_dict(savefile):
    """ Load .pkl file as dict """
    file_extension = savefile.rsplit(".", 1)[-1].lower()

    if file_extension == "pkl":
        with open(savefile, 'rb') as file:
            return pickle.load(file)

    else:
        raise ValueError("Unsupported file type. Only '.pkl' is supported.")


def display_df(df, caption=""):
    """ Display dataframe in notebook or console, with caption """
    if "ipykernel_launcher" in sys.argv[0]:
        if caption:
            df = df.style.set_caption(caption)

        df_html = df.to_html().replace('\\n', '<br>')
        display(HTML(df_html))

    else:
        if caption:
            print(caption)

        print(df)


# Class code (Doherty labels)
CLASS_CODE = {'sleep': 0, 'sedentary': 1, 'tasks-light': 2, 'walking': 3, 'moderate': 4}
# List of classes, ordered by code
CLASSES = ['sleep', 'sedentary', 'tasks-light', 'walking', 'moderate']
NUM_CLASSES = len(CLASSES)
# Colors to be used for each class, ordered by code
COLORS = ['blue', 'red', 'darkorange', 'lightgreen', 'green']


def encode_one_hot(y):
    '''
    0 -> 1,0,0,0,0
    1 -> 0,1,0,0,0
    2 -> 0,0,1,0,0
    3 -> 0,0,0,1,0
    4 -> 0,0,0,0,1
    '''
    return (y.reshape(-1, 1) == np.arange(NUM_CLASSES)).astype(int)


def train_hmm(Y_pred, y_true):
    ''' Estimate prior, emission, and transition matrices
    https://en.wikipedia.org/wiki/Hidden_Markov_model '''
    if Y_pred.ndim == 1 or Y_pred.shape[1] == 1:
        Y_pred = encode_one_hot(Y_pred)

    prior = np.mean(y_true.reshape(-1, 1) == np.arange(NUM_CLASSES), axis=0)
    emission = np.vstack(
        [np.mean(Y_pred[y_true == i], axis=0) for i in range(NUM_CLASSES)]
    )
    transition = np.vstack(
        [np.mean(y_true[1:][(y_true == i)[:-1]].reshape(-1, 1) == np.arange(NUM_CLASSES), axis=0)
            for i in range(NUM_CLASSES)]
    )
    return prior, emission, transition


def viterbi(y_pred, prior, emission, transition):
    ''' https://en.wikipedia.org/wiki/Viterbi_algorithm '''
    small_number = 1e-16

    def log(x):
        return np.log(x + small_number)

    num_obs = len(y_pred)
    probs = np.zeros((num_obs, NUM_CLASSES))
    probs[0, :] = log(prior) + log(emission[:, y_pred[0]])
    for j in range(1, num_obs):
        for i in range(NUM_CLASSES):
            probs[j, i] = np.max(
                log(emission[i, y_pred[j]]) +
                log(transition[:, i]) +
                probs[j - 1, :])  # probs already in log scale
    viterbi_path = np.zeros_like(y_pred)
    viterbi_path[-1] = np.argmax(probs[-1, :])
    for j in reversed(range(num_obs - 1)):
        viterbi_path[j] = np.argmax(
            log(transition[:, viterbi_path[j + 1]]) +
            probs[j, :])  # probs already in log scale

    return viterbi_path


def compute_scores(y_true, y_pred):
    ''' Compute a bunch of scoring functions '''
    confusion = metrics.confusion_matrix(y_true, y_pred)
    per_class_recall = metrics.recall_score(y_true, y_pred, average=None)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    balanced_acuracy = metrics.balanced_accuracy_score(y_true, y_pred)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    return {
        'confusion': confusion,
        'per_class_recall': per_class_recall,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acuracy,
        'kappa': kappa,
    }


def print_scores(scores):
    print("Accuracy score:", scores['accuracy'])
    print("Balanced accuracy score:", scores['balanced_accuracy'])
    print("Cohen kappa score:", scores['kappa'])
    print("\nPer-class recall scores:")
    print(
        "sleep      : {}\n"
        "sedentary  : {}\n"
        "tasks-light: {}\n"
        "walking    : {}\n"
        "moderate   : {}".format(*scores['per_class_recall'])
    )
    print("\nConfusion matrix:\n", scores['confusion'])


# ----------------------------------------
#  Function to plot activity timeseries
# ----------------------------------------
register_matplotlib_converters()


def plot_activity(x, y, t):
    ''' Plot activity timeseries '''
    BACKGROUND_COLOR = '#d3d3d3'  # lightgray

    def split_by_timegap(group, seconds=30):
        subgroupIDs = (group.index.to_series().diff() > timedelta(seconds=seconds)).cumsum()
        subgroups = group.groupby(by=subgroupIDs)
        return subgroups

    convert_date = np.vectorize(
        lambda day, x: matplotlib.dates.date2num(datetime.combine(day, x)))
    timeseries = pd.DataFrame(data={'x': x, 'y': y, 't': t})
    timeseries.set_index('t', inplace=True)
    timeseries['x'] = timeseries['x'].rolling(window=12, min_periods=1).mean()  # ! inplace?
    ylim_min, ylim_max = np.min(x), np.max(x)
    groups = timeseries.groupby(timeseries.index.date)
    fig, axs = plt.subplots(nrows=len(groups) + 1)
    for ax, (day, group) in zip(axs, groups):
        for _, subgroup in split_by_timegap(group):
            _t = convert_date(day, subgroup.index.time)
            _ys = [(subgroup['y'] == i).astype('int') * ylim_max for i in range(NUM_CLASSES)]
            ax.plot(_t, subgroup['x'], c='k')
            ax.stackplot(_t, _ys, colors=COLORS, alpha=.5, edgecolor='none')

        ax.get_xaxis().grid(True, which='major', color='grey', alpha=0.5)
        ax.get_xaxis().grid(True, which='minor', color='grey', alpha=0.25)
        ax.set_xlim((datetime.combine(day, time(0, 0, 0, 0)),
                     datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0))))
        ax.set_xticks(pd.date_range(start=datetime.combine(day, time(0, 0, 0, 0)),
                                    end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
                                    freq='4H'))
        ax.set_xticks(pd.date_range(start=datetime.combine(day, time(0, 0, 0, 0)),
                                    end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
                                    freq='1H'), minor=True)
        ax.set_ylim((ylim_min, ylim_max))
        ax.get_yaxis().set_ticks([])  # hide y-axis lables
        ax.spines['top'].set_color(BACKGROUND_COLOR)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor(BACKGROUND_COLOR)

        ax.set_title(
            day.strftime("%A,\n%d %B"), weight='bold',
            x=-.2, y=.5,
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation='horizontal',
            transform=ax.transAxes,
            fontsize='medium',
            color='k'
        )

    # legends
    axs[-1].axis('off')
    legend_patches = []
    legend_patches.append(mlines.Line2D([], [], color='k', label='acceleration'))
    for color, label in zip(COLORS, CLASSES):
        legend_patches.append(mpatches.Patch(facecolor=color, label=label, alpha=0.5))
    axs[-1].legend(handles=legend_patches, bbox_to_anchor=(0., 0., 1., 1.),
                   loc='center', ncol=min(3, len(legend_patches)), mode="best",
                   borderaxespad=0, framealpha=0.6, frameon=True, fancybox=True)
    axs[-1].spines['left'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['bottom'].set_visible(False)

    # format x-axis to show hours
    fig.autofmt_xdate()
    hours = [(str(hr) + 'am') if hr <= 12 else (str(hr - 12) + 'pm') for hr in range(0, 24, 4)]
    axs[0].set_xticklabels(hours)
    axs[0].tick_params(labelbottom=False, labeltop=True, labelleft=False)

    return fig, axs
