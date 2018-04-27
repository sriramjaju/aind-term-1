import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
        L is likelihood of fitted model
        p is complexity
        p * logN is the penalty term, increases with higher p to penalise complexity and avoid overfitting
        N is size of data set
        -2 * logL : decreazsed with higher p
        p * logN : increases with higher p
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        min_bic_score = None
        min_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n)
                # calculate the score of the model
                log_likelihood = hmm_model.score(self.X, self.lengths)

                num_data_points = sum(self.lengths)
                num_feature = len(self.X[0])
                num_free_params = (n ** 2) + (2 * n * num_feature) - 1

                score_bic = -2*log_likelihood + num_free_params * math.log(num_data_points)

                if min_bic_score is None or min_bic_score > score_bic:
                    min_bic_score = score_bic
                    min_model = hmm_model

            except Exception as e:
                # print('except', e)
                pass
        # Lower the BIC score the better the model
        return min_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_DIC_score = float('-inf')
        best_model = None
        # Loop through range of components
        for n in range(self.min_n_components, self.max_n_components+1):
        #Try and find best model if it exists
            try:
                hmm_model = self.base_model(n)
                logL_scores = []

                # Loop through all words
                for word, (X_others, lengths_others) in self.hwords.items():
                    # For words that aren't this word, compute logL score
                    if word != self.this_word:
                        logL_scores.append(hmm_model.score(X_others, lengths_others))

                # Find average logL score of other words
                average_logL_others = np.mean(logL_scores)

                # Assign DIC score (model score for 'this_word' - average score of others)
                DIC_score = hmm_model.score(self.X, self.lengths) - average_logL_others

                if DIC_score > best_DIC_score:
                    best_DIC_score = DIC_score
                    best_model = hmm_model
            # If best model doesn't exist, pass
            except:
                pass
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_CV_score = float('-inf')
        best_num_components = None
        # Define split method and number of splits
        folds = min(3, len(self.sequences))
        fold_count = 0
        total_logL_score = 0
        # Loop through all components
        for n in range(self.min_n_components, self.max_n_components+1):
            fold_count = 0
            total_logL_score = 0
            try:
                # Split sequences into training and test sets
                if folds > 1:
                    split_method = KFold(n_splits=folds)

                    for cv_train, cv_test in split_method.split(self.sequences):
                        # Training KFolds
                        X_train, lengths_train = combine_sequences(cv_train, self.sequences)
                        # Testing KFolds
                        X_test, lengths_test = combine_sequences(cv_test, self.sequences)

                        # Build HMM model for training set
                        hmm_model = GaussianHMM(n_components=n,
                                                covariance_type="diag",
                                                n_iter=1000,
                                                random_state=self.random_state,
                                                verbose=False).fit(X_train, lengths_train)

                        # Calculae logL score of model
                        logL_score = hmm_model.score(X_test, lengths_test)
                        # Update the total logL score
                        total_logL_score += logL_score
                        # Increase fold count for each step in the loop
                        fold_count += 1
                else:
                    hmm_model = GaussianHMM(n_components=n,
                                            covariance_type="diag",
                                            n_iter=1000,
                                            random_state=self.random_state,
                                            verbose=False).fit(self.X, self.lengths)

                    logL_score = hmm_model.score(self.X, self.lengths)
                    total_logL_score += logL_score
                    fold_count += 1


                # Average the logL score across the number of folds
                CV_score = total_logL_score/fold_count

                # Assign best CV score and number of components to use
                if CV_score > best_CV_score:
                    best_CV_score = CV_score
                    best_num_components = n
            # If a model doesn't exist, pass
            except:
               pass
        return self.base_model(best_num_components)

