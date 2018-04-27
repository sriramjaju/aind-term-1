import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    X_lengths = test_set.get_all_Xlengths()

    for test_X, test_lengths in X_lengths.values():
        log_l = {}  # to save loglikelihood of words
        max_score = float('-inf')
        best_guess = None

        for word, model in models.items():
            try:
                logL_score = model.score(test_X, test_lengths)
                log_l[word] = logL_score

                if logL_score > max_score:
                    best_guess = word
                    max_score = logL_score

            except Exception:
                log_l[word] = float('-inf')

        probabilities.append(log_l)
        guesses.append(best_guess)


    return probabilities, guesses

