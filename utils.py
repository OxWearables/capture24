from IPython.display import display, HTML
import numpy as np
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
