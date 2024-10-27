# models/RandomSearchCVWrapper.py

from sklearn.model_selection import RandomizedSearchCV


class RandomSearchCVWrapper:
    """
    A wrapper class for sklearn's RandomizedSearchCV to simplify
    hyperparameter tuning.
    """

    def __init__(
        self,
        model,
        param_distributions,
        n_iter=20,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    ):
        self.model = model
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring  # Add scoring parameter
        self.random_state = random_state
        self.best_model = None
        self.best_params_ = None

    def tune_hyperparameters(self, X, y):
        """
        Performs RandomizedSearchCV to find the best hyperparameters
        and returns the best model.
        """
        random_search = RandomizedSearchCV(
            self.model,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,  # Use scoring parameter
            random_state=self.random_state,
            verbose=2,  # Add verbosity for better tracking
        )
        random_search.fit(X, y)
        self.best_params_ = random_search.best_params_
        self.best_model = random_search.best_estimator_
        print("Best parameters:", self.best_params_)
        return self.best_model  # Return the best model
