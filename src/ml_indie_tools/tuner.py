import copy
import logging


class MLTuner:
    """Simple hyper parameter tuner

    Sample `search_space`:

    .. code-block:: python

         param_space_minimal_prm = {
            "dense_layers": [4, 8, 12],
            "dense_neurons":[256, 512, 768],
            "learning_rate": [0.001, 0.002],
            "regu1": [1e-8, 1e-7]
        }

    :param search_space: Dictionary defining the search space.
    :param progress_callback: Callback function that is called after each iteration with updated search space as parameter.
    """

    def __init__(self, search_space=None, progress_callback=None):
        # XXX Get rid of search_space?!

        self.log = logging.getLogger("MLTuner")
        self.progress_callback = progress_callback
        if search_space is None:
            self.search_space = {}
            self.search_space["best_ev"] = 0
            self.search_space["is_first"] = True
            self.search_space["progress"] = 0
        else:
            self.search_space = search_space
            self.search_space["is_first"] = False

    def tune(self, param_space, eval_func):
        """Tune hyper parameters

        Example parameter space:

        .. code-block:: python

            param_space = {
                "dense_layers": [4, 8, 12],
                "dense_neurons":[256, 512, 768],
                "learning_rate": [0.001, 0.002],
                "regu1": [1e-8, 1e-7]
            }

        `eval_func` is called with a dictionary of hyper parameters with exactly one value for each key, e.g.:

        .. code-block:: python

             params={
                "dense_layers": 8,
                "dense_neurons": 256,
                "learning_rate": 0.001,
                "regu1": 1e-8
            }

        :param param_space: Dictionary defining the search space.
        :param eval_func: Function that is called to evaluate the hyper parameters.
        """
        if "best_params" not in self.search_space:
            self.search_space["best_params"] = {}
        for key in param_space:
            if key not in self.search_space["best_params"]:
                self.search_space["best_params"][key] = param_space[key][0]
        p_cnt = 0
        for key in param_space:
            params = copy.deepcopy(self.search_space["best_params"])
            vals = param_space[key]
            for val in vals:
                if self.search_space["is_first"] is False:
                    if val == self.search_space["best_params"][key]:
                        continue  # Was already tested.
                else:
                    self.search_space["is_first"] = False
                if p_cnt < self.search_space["progress"]:
                    p_cnt += 1
                    self.log.debug(f"Fast forwarding: {key} {val}")
                    continue
                else:
                    p_cnt += 1
                self.search_space["progress"] += 1
                params[key] = val
                self.log.debug(f"Testing: {key}={val} with {params}")
                ev = eval_func(params)
                self.log.debug(f"Eval: {ev}")
                if ev > self.search_space["best_ev"]:
                    self.search_space["best_ev"] = ev
                    self.search_space["best_params"] = copy.deepcopy(params)
                    self.log.info(f"Best parameter set with ev={ev}: {params}")
                if self.progress_callback is not None:
                    self.progress_callback(self.search_space)
        self.log.info(
            f"Best parameter set with {self.search_space['best_ev']} eval: {self.search_space['best_params']}"
        )
        return self.search_space["best_params"]
