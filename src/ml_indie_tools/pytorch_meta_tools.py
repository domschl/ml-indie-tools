import os
import torch


def ml_metadata_compatible(current_params, saved_params, updatable_keys=[]):
    is_valid = True
    keys = set(list(current_params.keys()) + list(saved_params.keys()))
    for key in keys:
        if key in updatable_keys:
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
        print("Incompatible metadata.")
        return False
    return True


def ml_get_model_filename(model_path, filename="model.pt"):
    return os.path.join(model_path, filename)


def ml_checkpoint_available(model_path, filename="model.pt"):
    load_file = ml_get_model_filename(model_path, filename=filename)
    return os.path.exists(load_file)


def ml_save_checkpoint(self, params, model, optimizer, current_epoch, current_loss):
    params["current_epoch"] = current_epoch
    params["current_loss"] = current_loss
    state = {
        "params": params,
        "model_states": model.state_dict(),
        "optimizer_states": optimizer.state_dict(),
    }
    save_file = ml_get_model_filename(self.model_path)
    torch.save(state, save_file)


def ml_load_model_metadata_from_checkpoint(self, device=None):
    load_file = ml_get_model_filename(self.model_path)
    if not os.path.exists(load_file):
        return None
    if device is None:
        state = torch.load(load_file)
    else:
        state = torch.load(load_file, map_location=device)
    return state["params"]


def ml_load_checkpoint(self, params, model, optimizer, device=None):
    load_file = self.get_model_filename()
    if not os.path.exists(load_file):
        print(load_file)
        print("No saved state, starting from scratch.")
        return 0, 0
    if device is None:
        state = torch.load(load_file)
    else:
        state = torch.load(load_file, map_location=device)
    params_new = state["params"]
    if self.is_metadata_compatible(params, params_new) is False:
        self.log.warning("Metadata incompatible, starting from scratch.")
        return 0, 0
    params = params_new
    model.load_state_dict(state["model_states"])
    optimizer.load_state_dict(state["optimizer_states"])
    for g in optimizer.param_groups:  # Allow for different learning rates
        g["lr"] = params["learning_rate"]
    epoch = params["current_epoch"]
    loss = params["current_loss"]
    print(
        f"Continuing from saved state epoch={epoch+1}, loss={loss:.3f}"
    )  # Save is not necessarily on epoch boundary, so that's approx.
    self.log.info(f"Continuing from saved state epoch={epoch+1}, loss={loss:.3f}")
    return epoch, loss
