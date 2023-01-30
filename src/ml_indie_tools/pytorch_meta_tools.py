import copy
import os
import json
import time
import logging

import torch


class ModelJanitor:
    def __init__(self, model_path, params, updatable_keys=[]):
        self.log = logging.getLogger("ModelJanitor")
        self.model_path = model_path
        if os.path.exists(model_path) is False:
            os.makedirs(model_path)
        self.params = params
        self.updatable_keys = updatable_keys

        suffix = self.expand_name_template(params)
        self.model_meta_filename = os.path.join(
            self.model_path, f"model_meta_{suffix}.json"
        )

    @staticmethod
    def expand_name_template(params):
        exp = copy.copy(params["meta_name_template"])
        for key in params:
            src = "{" + key + "}"
            dst = f"{params[key]}"
            exp = exp.replace(src, dst).replace("[", "(").replace("]", ")")
        return exp

    def save_model_metadata(self, params, current_epoch):
        params["current_epoch"] = current_epoch
        try:
            with open(self.model_meta_filename, "w") as f:
                f.write(json.dumps(params))
        except Exception as e:
            print(f"Failed to store model metadata to {self.model_meta_filename}: {e}")
            return False
        return True

    def read_model_metadata(self):
        try:
            with open(self.model_meta_filename, "r") as f:
                params = json.load(f)
        except Exception as e:
            print(
                f"Cannot access project meta-data at {self.model_meta_filename}: {e}, starting anew."
            )
            return None
        return params

    def is_metadata_compatible(self, current_params, saved_params):
        is_valid = True
        keys = set(list(current_params.keys()) + list(saved_params.keys()))
        for key in keys:
            if key in self.updatable_keys:
                continue
            if key not in saved_params:
                print(
                    f"Key {key} not available in last checkpoint model_meta, current_params[{key}]: {current_params[key]},"
                )
                print(
                    "cannot import incompatible model. Put key in `updatable_keys` list, if irrelevant."
                )
                is_valid = False
            elif key not in current_params:
                print(
                    f"Key {key} not available in params, last checkpoint saved_params[{key}]: {saved_params[key]},"
                )
                print(
                    "cannot import incompatible model. Put key in `updatable_keys` list, if irrelevant."
                )
                is_valid = False
            elif saved_params[key] != current_params[key]:
                print(
                    f"Last checkpoint saved_params[{key}]: {saved_params[key]} != current_params[{key}]: {current_params[key]},"
                )
                print(
                    "cannot import incompatible model. Put key in `updatable_keys` list, if irrelevant."
                )
                is_valid = False
        if is_valid is False:
            print("Aborting import.")
            return False
        return True

    # -----------------  Model Checkpointing -----------------

    def save_checkpoint(self, model, optimizer, params, current_epoch, current_loss):
        params["current_epoch"] = current_epoch
        params["current_loss"] = current_loss
        state = {
            "params": params,
            "model_states": model.state_dict(),
            "optimizer_states": optimizer.state_dict(),
        }
        filename = "model.pt"
        save_file = os.path.join(self.model_path, filename)
        torch.save(state, save_file)

    def save_history(self, history):
        filename = "history.json"
        save_file = os.path.join(self.model_path, filename)
        try:
            with open(save_file, "w") as f:
                json.dump(history, f)
        except Exception as e:
            print(f"Failed to write training history file {save_file}, {e}")

    def load_history(self):
        filename = "history.json"
        load_file = os.path.join(self.model_path, filename)
        try:
            with open(load_file, "r") as f:
                history = json.load(f)
        except Exception as _:
            print(f"Starting new history file {load_file}")
            return [], time.time()
        if len(history) > 0:
            start = history[-1]["timestamp"]
        return history, start

    def load_checkpoint(self, model, optimizer, params):
        filename = "model.pt"
        load_file = os.path.join(self.model_path, filename)
        # latest = os.path.join(self.model_path, "latest.json")
        if not os.path.exists(load_file):
            print(load_file)
            print("No saved state, starting from scratch.")
            return 0, 0
        state = torch.load(load_file)
        params_new = state["params"]
        if self.is_metadata_compatible(params, params_new) is False:
            self.log.warning("Metadata incompatible, starting from scratch.")
            return 0, 0
        params = params_new
        model.load_state_dict(state["model_states"])
        optimizer.load_state_dict(state["optimizer_states"])
        epoch = params["current_epoch"]
        loss = params["current_loss"]
        model.train()
        print(
            f"Continuing from saved state epoch={epoch+1}, loss={loss:.3f}"
        )  # Save is not necessarily on epoch boundary, so that's approx.
        self.log.info(f"Continuing from saved state epoch={epoch+1}, loss={loss:.3f}")
        return epoch, loss
