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


def ml_save_checkpoint(
    params,
    model,
    optimizer,
    current_epoch,
    current_loss,
    file_path,
):
    params["current_epoch"] = current_epoch
    params["current_loss"] = current_loss
    state = {
        "params": params,
        "model_states": model.state_dict(),
        "optimizer_states": optimizer.state_dict(),
    }
    torch.save(state, file_path)


def ml_load_model_metadata_from_checkpoint(file_path, device=None):
    if not os.path.exists(file_path):
        return None
    if device is None:
        state = torch.load(file_path)
    else:
        state = torch.load(file_path, map_location=device)
    return state["params"]


def ml_load_checkpoint(params, model, optimizer, file_path, device=None):
    if not os.path.exists(file_path):
        print(f"No saved state, no {file_path}, starting from scratch.")
        return None
    if device is None:
        state = torch.load(file_path)
    else:
        state = torch.load(file_path, map_location=device)
    params_new = state["params"]
    if ml_metadata_compatible(params, params_new) is False:
        print("Metadata incompatible, starting from scratch.")
        return None
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
    return params
