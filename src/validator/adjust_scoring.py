
from typing import Dict, List


def conditional_power_scaling(score_dict: Dict[int, float]) -> Dict[int, float]:
    """
    Adjusts the distribution of scores, such that the best miners are rewarded significantly more than the rest.
    This ensures that it's profitable to run a high-end model, in comparison to cheap models.

    Args:
        score_dict: 
            A dictionary mapping miner UIDs to their scores.

    Returns:
        Dict[int, float]:
            A dictionary mapping miner UIDs to their adjusted scores.
    """
    scaling_factor = 0.2
    mean_score = sum(score_dict.values()) / len(score_dict)

    transformed_scores = []
    for uid, score in score_dict.items():
        if score > mean_score:
            # Apply a power < 1 to scores above the mean to raise them, retaining separation
            transformed_score = (score / mean_score) ** (1 - scaling_factor)
        else:
            # Apply a power > 1 to scores below the mean to lower them, retaining separation
            transformed_score = (score / mean_score) ** (1 + scaling_factor)

        transformed_scores.append(transformed_score)

    for uid, t in zip(score_dict.keys(), transformed_scores):
        score_dict[uid] = (t / max(transformed_scores))

    return score_dict


def normalize_scores(scores: List[float]) -> List[float]:
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        # If all scores are the same, give all ones
        return [1] * len(scores)

    # Normalize scores from 0 to 1
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]

    return normalized_scores
