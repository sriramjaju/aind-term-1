SelectorBIC takes argument of ModelSelector instance of base class with attributes such as: this_word, min_n_components, max_n_components
        Loop from min_n_components to max_n_components
        Find the lowest BIC score as the better model
    Maximising the likelihood of data and penalising large-size models
    Use of penalty term instead of cross-validation
    Parameters of HMM are of 2 types, transition probabilities and emission probabilities

# num_free_params = initial_state_occupation_probs + transition_probs + emission_probs
                #                 = num_states + num_states * (num_states - 1) + (num_states * num_data_points * 2)
                #                 = (num_states ** 2) + (2 * num_states * num_data_points)
