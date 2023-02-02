import optuna


class StopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consecutive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consecutive_pruned_count += 1
        else:
            self._consecutive_pruned_count = 0

        if self._consecutive_pruned_count >= self.threshold:
            study.stop()
