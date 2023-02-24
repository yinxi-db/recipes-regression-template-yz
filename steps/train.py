"""
This module defines the following routines used by the 'train' step of the regression recipe:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model recipe.
"""
from typing import Dict, Any


from typing import Dict, Any

def estimator_fn(estimator_params: Dict[str, Any] = None):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(random_state=42, **estimator_params)

def my_early_stop_fn(*args):
  from hyperopt.early_stop import no_progress_loss
  return no_progress_loss(10)(*args)