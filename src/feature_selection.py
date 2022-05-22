from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectPercentile
import numpy as np

def k_best_f(feature_mx: np.ndarray, tr_label: list, n: int) -> Tuple[np.ndarray, List[int]]:
    """
    args:
        - feature_mx: a (num_instance, num_feature) np ndarray
        - tr_label: list of training labels
        - n: kept k features
    returns:
        - a reduced (num_instance, k_best_feature) np ndarray
        - the list of kept feature indices

    select n best features based on mutual information
    """
    selector = SelectKBest(mutual_info_classif, k=n)
    r_feature = selector.fit_transform(feature_mx, tr_label)
    indices = selector.get_support(indices=True).tolist()
    return r_feature, indices


def k_perc_best_f(feature_mx: np.ndarray, tr_label: list, p: int) -> Tuple[np.ndarray, List[int]]:
    """
    args:
        - feature_mx: a (num_instance, num_feature) np ndarray
        - tr_label: list of training labels
        - p: kept p percentile of features
    returns:
        - a reduced (num_instance, k_best_precentile_feature) np ndarray
        - the list of kept feature indices

    select p percentile best features based on mutual information
    """
    selector = SelectPercentile(mutual_info_classif, percentile=p)
    r_feature = selector.fit_transform(feature_mx, tr_label)
    indices = selector.get_support(indices=True).tolist()
    return r_feature, indices


def prune_test(feature_mx: np.ndarray, indices: list) -> np.ndarray:
    """
    args:
        - feature_mx: a (num_instance, num_feature) np ndarray
        - indices: list of kept feature indices
    returns:
        a reduced (num_instance, kept_feature) np ndarray

    prune the feature mx of test set after performing feature selection on the training data
    """
    p_feature = feature_mx[:, indices]
    return p_feature
