# stdlib
import platform
from typing import Any, Dict
from logging import warning
# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# synthcity absolute
from synthcity.metrics.core import MetricEvaluator

from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.mlp import MLP
from synthcity.utils.serialization import load_from_file, save_to_file

from syntheval.metrics.core.metric import MetricClass
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from syntheval.utils.nn_distance import _knn_distance

def exact_match(real_baseline_df, synthetic_df, chunk_size = 2000, rtol=5e-2, atol=1e-5):

    real_num = real_baseline_df.select_dtypes(include=[np.number]).to_numpy()
    syn_num = synthetic_df.select_dtypes(include=[np.number]).to_numpy()


    count = 0
    for start in range(0, len(syn_num), chunk_size):

        end = start + chunk_size
        syn_chunk = syn_num[start:end]  # Shape: (chunk_size, num_columns)

        # Compute broadcasted isclose
        is_close = np.isclose(syn_chunk[:, None, :], real_num[None, :, :], rtol=rtol, atol=atol, equal_nan=True)

        # Check if all columns match
        colwise_match = np.any(np.all(is_close, axis=2), axis=0)  # shape: (chunk_size,)
        count += np.sum(colwise_match)

    return count

class AttackEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.AttackEvaluator
        :parts: 1

    Evaluating the risk of attribute inference attack.

    This class evaluates the risk of a type of privacy attack, known as attribute inference attack.
    In this setting, the attacker has access to the synthetic dataset as well as partial information about the real data
    (quasi-identifiers). The attacker seeks to uncover the sensitive attributes of the real data using these two pieces
    of information.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "attack"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_leakage(
        self,
        classifier_template: Any,
        classifier_args: Dict,
        regressor_template: Any,
        regressor_args: Dict,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

        if len(X_gt.sensitive_features) == 0:
            return {}

        output = []
        for col in X_gt.sensitive_features:
            if col not in X_syn.columns:
                continue

            target = X_syn[col]
            keys_data = X_syn.drop(columns=[col])

            # TODO: use a common limit for categorical features
            if len(target.unique()) < 15:
                task_type = "classification"
                encoder = LabelEncoder()
                target = encoder.fit_transform(target)
                if "n_units_out" in classifier_args:
                    classifier_args["n_units_out"] = len(np.unique(target))
                model = classifier_template(**classifier_args)
            else:
                task_type = "regression"
                model = regressor_template(**regressor_args)

            model.fit(keys_data.values, np.asarray(target))

            test_target = X_gt[col]
            if task_type == "classification":
                test_target = encoder.transform(test_target)

            test_keys_data = X_gt.drop(columns=[col])

            preds = model.predict(test_keys_data.values)

            if task_type == "classification":
                output.append(
                    (np.asarray(preds) == np.asarray(test_target)).sum() / (len(preds) + 1)
                )
            else:
                # check if near or not
                output.append(
                    np.mean(np.isclose(np.asarray(preds), np.asarray(test_target), rtol=5e-2, atol=1e-5))
                )

        if len(output) == 0:
            return {}

        results = {self._reduction: self.reduction()(output)}

        save_to_file(cache_file, results)

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)[self._reduction]


class DataLeakageMLP(AttackEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.DataLeakageMLP
        :parts: 1

    Data leakage test using a neural net.
    """

    @staticmethod
    def name() -> str:
        return "data_leakage_mlp"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        return self._evaluate_leakage(
            MLP,
            {
                "task_type": "classification",
                "n_units_in": X_gt.shape[1] - 1,
                "n_units_out": 0,
                "random_state": self._random_state,
            },
            MLP,
            {
                "task_type": "regression",
                "n_units_in": X_gt.shape[1] - 1,
                "n_units_out": 1,
                "random_state": self._random_state,
            },
            X_gt,
            X_syn,
        )


class DataLeakageXGB(AttackEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.DataLeakageXGB
        :parts: 1

    Data leakage test using XGBoost
    """

    @staticmethod
    def name() -> str:
        return "data_leakage_xgb"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        return self._evaluate_leakage(
            XGBClassifier,
            {
                "n_jobs": -1,
                "eval_metric": "logloss",
            },
            XGBRegressor,
            {"n_jobs": -1},
            X_gt,
            X_syn,
        )


class DataLeakageLinear(AttackEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.DataLeakageLinear
        :parts: 1


    Data leakage test using a linear model
    """

    @staticmethod
    def name() -> str:
        return "data_leakage_linear"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        return self._evaluate_leakage(
            LogisticRegression,
            {"random_state": self._random_state},
            LinearRegression,
            {},
            X_gt,
            X_syn,
        )
    
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

def membership_inference(train_df, holdout_df, synth_df, known_columns, percentile_threshold=0.5):
    # Combine training and syn for consistent normalization
    reference_df = pd.concat([train_df, synth_df])

    # Fit one shared MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(reference_df[known_columns])

    # Normalize all datasets using same scaler
    train_norm = pd.DataFrame(scaler.transform(train_df[known_columns]), columns=known_columns)
    holdout_norm = pd.DataFrame(scaler.transform(holdout_df[known_columns]), columns=known_columns)
    synth_norm = pd.DataFrame(scaler.transform(synth_df[known_columns]), columns=known_columns)

    # Fit kNN model on synthetic data
    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(synth_norm)

    # Compute distances to nearest synthetic record
    train_distances, _ = nbrs.kneighbors(train_norm)
    holdout_distances, _ = nbrs.kneighbors(holdout_norm)

    # Predict membership: 1 if distance < threshold
    combine = np.concatenate([train_distances.flatten(), holdout_distances.flatten()])
    threshold = np.percentile(combine, percentile_threshold)  # Use percentile to set threshold
    print(f"Threshold for membership inference: {threshold}, min: {combine.min()}, max: {combine.max()}")
    train_preds = (train_distances.flatten() < threshold).astype(int)
    holdout_preds = (holdout_distances.flatten() < threshold).astype(int)

    # Combine predictions and ground truth
    y_true = np.concatenate([np.ones(len(train_preds)), np.zeros(len(holdout_preds))])
    y_pred = np.concatenate([train_preds, holdout_preds])

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)


    return {
        # 'train_pred': train_preds,
        # 'holdout_pred': holdout_preds,
        # 'true_labels': y_true,
        # 'predicted_labels': y_pred,
        'MIA_accuracy': accuracy,
        # 'balanced_accuracy': balanced_accuracy,
        # 'precision': precision,
        # 'recall': recall,
        # 'f1_score': f1,
        # 'specificity': specificity

    }

def _adversarial_score(real, fake, cat_cols, metric):
    """Function for calculating adversarial score
    
    Args:
        real (DataFrame) : Real dataset
        fake (DataFrame) : Synthetic dataset
        cat_cols (List[str]) : list of strings
        metric (str) : keyword literal for NN module

    Returns:
        float : Adversarial score
    
    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> _adversarial_score(real, fake, [], 'euclid')
        0.0
    """
    left = np.mean(_knn_distance(real, fake, cat_cols, 1, metric)[0] > _knn_distance(real, real, cat_cols, 1, metric)[0])
    right = np.mean(_knn_distance(fake, real, cat_cols, 1, metric)[0] > _knn_distance(fake, fake, cat_cols, 1, metric)[0])
    return 0.5 * (left + right)

def evaluate_dataset_nnaa(real, fake, num_cols, cat_cols, metric, n_resample):
    """Helper function for running adversarial score multiple times if the 
    datasets have much different sizes.
    
    Args:
        real (DataFrame) : Real dataset
        fake (DataFrame) : Synthetic dataset
        num_cols (List[str]) : list of strings
        cat_cols (List[str]) : list of strings
        metric (str) : keyword literal for NN module
        n_resample (int) : number of resample rounds to run if datasets are of different sizes
    
    Returns:
        float, float: Average adversarial score and standard error
    
    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> evaluate_dataset_nnaa(real, fake, [], [], 'euclid', 1)
        (0.0, 0.0)
    """

    real_fake = len(real)/len(fake)
    fake_real = len(fake)/len(real)

    if any([real_fake > 1, fake_real > 1]):
        aa_lst = []
        for _ in range(n_resample):
            temp_r = real if real_fake < 1 else real.sample(n=len(fake))
            temp_f = fake if fake_real < 1 else fake.sample(n=len(real))
            aa_lst.append(_adversarial_score(temp_r, temp_f, cat_cols, metric))

        avg = np.mean(aa_lst)
        err = np.std(aa_lst, ddof=1)/np.sqrt(len(aa_lst))
    else:
        avg = _adversarial_score(real, fake, cat_cols, metric)
        err = 0.0

    return avg, err

class NearestNeighbourAdversarialAccuracy(MetricClass):
    """The Metric Class is an abstract class that interfaces with 
    SynthEval. When initialised the class has the following attributes:

    Attributes:
    self.real_data : DataFrame
    self.synt_data : DataFrame
    self.hout_data : DataFrame
    self.cat_cols  : list of strings
    self.num_cols  : list of strings
    
    self.nn_dist   : string keyword
    
    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'nnaa'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, n_resample=30) -> dict:
        """Implementation heavily inspired by original paper
        
        Args:
            n_resample (int) : number of resample rounds to run if datasets are of different sizes
        
        Returns:
            dict: Average adversarial score and standard error
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> N = NearestNeighbourAdversarialAccuracy(real, fake, cat_cols=[], num_cols=[], nn_dist='euclid', do_preprocessing=False)
            >>> N.evaluate(n_resample = 1)
            {'avg': 0.0, 'err': 0.0}
        """

        avg, err = evaluate_dataset_nnaa(self.real_data,self.synt_data,self.num_cols,self.cat_cols,self.nn_dist,n_resample)

        self.results = {'avg': avg, 'err': err}

        if self.hout_data is not None:
            avg_h, err_h = evaluate_dataset_nnaa(self.hout_data,self.synt_data,self.num_cols,self.cat_cols,self.nn_dist,n_resample)
            diff = avg_h - avg
            err_diff = np.sqrt(err_h**2+err**2)

            self.results['priv_loss'] = diff
            self.results['priv_loss_err'] = err_diff

        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
| Nearest neighbour adversarial accuracy   :   %.4f  %.4f   |""" % (self.results['avg'], self.results['err'])
        return string

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0 
        """
        if self.results != {}:
            output =  [{'metric': 'nnaa', 'dim': 'u', 
                        'val': self.results['avg'], 
                        'err': self.results['err'], 
                        'n_val': 1-self.results['avg'], 
                        'n_err': self.results['err'], 
                        }]
            if self.hout_data is not None:
                output.extend([{'metric': 'priv_loss_nnaa', 'dim': 'p', 
                        'val': self.results['priv_loss'], 
                        'err': self.results['priv_loss_err'], 
                        'n_val': 1-abs(self.results['priv_loss']), 
                        'n_err': self.results['priv_loss_err'], 
                        }])
            return output
        else: pass

    def extra_formatted_output(self) -> dict:
        """Bit for printing the privacy loss together with the other privacy metrics"""
        if (self.results != {} and self.hout_data is not None):
            string = """\
| Privacy loss (diff. in NNAA)             :   %.4f  %.4f   |""" % (self.results['priv_loss'], self.results['priv_loss_err'])
            return {'privacy': string}
        else: pass
