import numpy as np
import json

################################
# CONSTS
################################

REVIEWS = 0
BOT_ACTION = 1
USER_DECISION = 2


################################

def correct_action(information):
    if information["hotel_value"] >= 8:
        return 1
    else:
        return 0


def random_action(information):
    return np.random.randint(2)


def user_rational_action(information):
    if information["bot_message"] >= 8:
        return 1
    else:
        return 0


def user_picky(information):
    if information["bot_message"] >= 9:
        return 1
    else:
        return 0


def user_sloppy(information):
    if information["bot_message"] >= 7:
        return 1
    else:
        return 0


def user_short_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or (information["previous_rounds"][-1][BOT_ACTION] >= 8 and
                information["previous_rounds"][-1][REVIEWS].mean() >= 8) \
            or (information["previous_rounds"][-1][BOT_ACTION] < 8 and
                information["previous_rounds"][-1][REVIEWS].mean() < 8):  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def user_picky_short_t4t(information):
    if information["bot_message"] >= 9 or ((information["bot_message"] >= 8) and (
            len(information["previous_rounds"]) == 0 or (
            information["previous_rounds"][-1][REVIEWS].mean() >= 8))):  # good hotel
        return 1
    else:
        return 0


def user_hard_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"]])) == 1:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def history_and_review_quality(history_window, quality_threshold):
    def func(information):
        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][
                                    -history_window:]])) == 1:  # cooperation from *result's* perspective
            if information["bot_message"] >= quality_threshold:  # good hotel from user's perspective
                return 1
            else:
                return 0
        else:
            return 0

    return func


def topic_based(positive_topics, negative_topics, quality_threshold):
    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"].loc[topic]) * 2 / (rank + 1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"].loc[topic]) * 2 / (rank + 1)
        if review_personal_score >= quality_threshold:  # good hotel from user's perspective
            return 1
        else:
            return 0

    return func


def LLM_based(is_stochastic):
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}

    if is_stochastic:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(np.random.rand() <= review_llm_score)

        return func
    else:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(review_llm_score >= 0.5)

        return func


# Our strategies
# justification: Hertwig, R., & Erev, I. (2009). "The description-experience gap in risky choice."
def cumulative_review_score_adjustment(
        information):
    past_reviews = information["previous_rounds"]
    if not past_reviews:
        return user_rational_action(information)
    successes = sum(1 for r in past_reviews if (r[USER_DECISION] == 1 and r[REVIEWS].mean() >= 8) or
                    (r[USER_DECISION] == 0 and r[REVIEWS].mean() < 8))
    success_rate = successes / len(past_reviews)
    base_threshold = 8
    if success_rate > 0.7:
        threshold = 9
    elif success_rate < 0.3:
        threshold = 7
    else:
        threshold = base_threshold

    if information["bot_message"] >= threshold:
        return 1
    else:
        return 0


# justification Kahneman, D., & Tversky, A. (1979). "Prospect Theory: An Analysis of Decision under Risk." Econometrica, 47(2), 263-291.
# Festinger, L. (1957). "A Theory of Cognitive Dissonance." Stanford University Press.
def loss_aversion_decision_making(information, recent_rounds=3, loss_penalty=0.5):
    past_reviews = information["previous_rounds"][-recent_rounds:]
    if not past_reviews:
        return user_rational_action(information)

    losses = sum(1 for r in past_reviews if (r[USER_DECISION] == 1 and r[REVIEWS].mean() < 8) or
                 (r[USER_DECISION] == 0 and r[REVIEWS].mean() >= 8))

    threshold_adjustment = losses * loss_penalty
    threshold = 8 - threshold_adjustment

    if information["bot_message"] >= threshold:
        return 1
    else:
        return 0


# Ebbinghaus, H. (1885). "Memory: A Contribution to Experimental Psychology." Teachers College, Columbia University.
# Festinger, L. (1957). "A Theory of Cognitive Dissonance." Stanford University Press.
def sequential_consistency_decay(information, decay_rate=0.9, consistency_threshold=0.7):  # with justification
    past_reviews = information["previous_rounds"]
    if not past_reviews:
        return user_rational_action(information)

    consistency_scores = []
    for i, review in enumerate(past_reviews):
        actual_quality = review[REVIEWS].mean()
        bot_recommendation = review[BOT_ACTION]
        consistency = 1 if (bot_recommendation >= 8 and actual_quality >= 8) or (
                bot_recommendation < 8 and actual_quality < 8) else 0
        consistency_scores.append(consistency * (decay_rate ** i))

    weighted_consistency = sum(consistency_scores) / len(past_reviews)

    current_bot_message = information["bot_message"]
    if weighted_consistency >= consistency_threshold and current_bot_message >= 8:
        return 1
    else:
        return 0


# Sutton, R. S., & Barto, A. G. (1998). "Reinforcement Learning: An Introduction." MIT Press.
# Thrun, S. (1992). "Efficient exploration in reinforcement learning." Technical Report, CMU-CS-92-102, Carnegie Mellon University.
def contextual_reinforcement_exploration_exploitation(information, base_exploration_rate=0.2, success_boost=0.1,
                                                      failure_penalty=0.1):
    past_reviews = information["previous_rounds"]
    if not past_reviews:
        return random_action(information)

    successes = sum(1 for r in past_reviews if (r[USER_DECISION] == 1 and r[REVIEWS].mean() >= 8) or
                    (r[USER_DECISION] == 0 and r[REVIEWS].mean() < 8))
    failures = sum(1 for r in past_reviews if (r[USER_DECISION] == 1 and r[REVIEWS].mean() < 8) or
                   (r[USER_DECISION] == 0 and r[REVIEWS].mean() >= 8))

    exploration_rate = base_exploration_rate - (successes * success_boost) + (failures * failure_penalty)
    exploration_rate = max(0.05, min(0.95, exploration_rate))

    if np.random.rand() < exploration_rate:
        return random_action(information)
    else:
        successful_threshold = 8
        if information["bot_message"] >= successful_threshold:
            return 1
        else:
            return 0


# Unjustified strats
def vengeful(information, lie_penalty=0.5):
    past_reviews = information["previous_rounds"]
    if not past_reviews:  # If no history, default behavior
        return user_rational_action(information)

    # number of lies
    lies = sum(1 for r in past_reviews if (r[BOT_ACTION] < 8 <= r[REVIEWS].mean()) or
               (r[BOT_ACTION] >= 8 > r[REVIEWS].mean()))
    if information["bot_message"] - lies * lie_penalty >= 8:
        return 1
    else:
        return 0


def adaptive_trust_review_discrepancy(information, discrepancy_threshold=1.5):
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}

    current_bot_score = information["bot_message"]
    current_llm_score = proba2go[
        information["review_id"]]
    discrepancy = abs(current_bot_score - current_llm_score)

    if discrepancy <= discrepancy_threshold:
        return 1 if current_bot_score >= 8 else 0
    else:
        return 1 if current_llm_score >= 8 else 0


def learning_from_outcome_patterns(information):
    success_threshold = 8
    pattern_success = False

    if len(information["previous_rounds"]) < 2:
        return user_rational_action(information)

    last_decision = information["previous_rounds"][-1][USER_DECISION]
    last_to_last_decision = information["previous_rounds"][-2][USER_DECISION]
    last_score = information["previous_rounds"][-1][REVIEWS].mean()

    if last_to_last_decision != last_decision:
        if last_decision == 1 and last_score >= success_threshold:
            pattern_success = True
        elif last_decision == 0 and last_score < success_threshold:
            pattern_success = True

    current_score = information["bot_message"]
    if pattern_success:
        if current_score >= success_threshold:
            return 1 if last_decision == 0 else 0
        else:
            return 0 if last_decision == 1 else 1
    else:
        return user_rational_action(information)
