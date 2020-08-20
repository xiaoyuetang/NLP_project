def compute_forward_score(x, feature_dict, states):
    '''
    Uses the forward algorithm to compute the score for a given sequence.

    Inputs:
        x (list[str]): Input sequence.
        feature_dict (dict[str] -> float): Dictionary that maps a given feature to its score.
        states (list[str]): List of unique states.
    Outputs:
        forward_score (float): Forward score for this sequence.
    '''

    n = len(x)  # Number of words
    d = len(states)  # Number of states
    forward_scores = np.zeros((n, d))

    # Start forward pass
    # Compute START transition scores
    for i, current_y in enumerate(states):
        transition_key = f"transition:START+{current_y}"
        emission_key = f"emission:{current_y}+{x[0]}"
        transition_score = feature_dict.get(transition_key, -10 ** 8)
        emission_score = feature_dict.get(emission_key, -10 ** 8)
        # Sum exponentials
        forward_scores[0, i] = transition_score + emission_score

    # Recursively compute best scores based on transmission and emission scores at each node
    for i in range(1, n):
        for k, current_y in enumerate(states):
            temp_score = 0
            for j, prev_y in enumerate(states):
                transition_key = f"transition:{prev_y}+{current_y}"
                emission_key = f"emission:{current_y}+{x[i]}"

                transition_score = feature_dict.get(transition_key, -10 ** 8)
                emission_score = feature_dict.get(emission_key, -10 ** 8)

                # Sum exponentials
                temp_score += np.exp(min(emission_score + transition_score + forward_scores[i - 1, j], 700))

            # Add to forward scores array
            forward_scores[i, k] = np.log(temp_score) if temp_score else -10 ** 8

    # Compute for STOP
    forward_prob = 0
    for j, prev_y in enumerate(states):
        transition_key = f"transition:{prev_y}+STOP"
        transition_score = feature_dict.get(transition_key, -10 ** 8)
        # Sum exponentials
        overall_score = np.exp(min(transition_score + forward_scores[n - 1, j], 700))
        forward_prob += overall_score
    # End forward pass

    alpha = np.log(forward_prob) if forward_prob else -700
    return forward_scores, alpha