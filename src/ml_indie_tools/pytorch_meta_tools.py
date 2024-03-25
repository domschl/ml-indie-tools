import os
import torch


def metadata_compatible(current_params, saved_params, updatable_keys=[], log=None):
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
            if log is not None:
                log.info(
                    f"Key {key} not available in last checkpoint model_meta, current_params[{key}]: {current_params[key]},"
                )
                log.info(
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
            if log is not None:
                log.info(
                    f"Key {key} not available in params, last checkpoint saved_params[{key}]: {saved_params[key]},"
                )
                log.info(
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
            if log is not None:
                log.info(
                    f"Last checkpoint saved_params[{key}]: {saved_params[key]} != current_params[{key}]: {current_params[key]},"
                )
                log.info(
                    "cannot import incompatible model. Put key in `updatable_keys` list, if irrelevant."
                )
            is_valid = False
    if is_valid is False:
        print("Incompatible metadata.")
        if log is not None:
            log.info("Incompatible metadata.")
        return False
    else:
        if log is not None:
            log.info("Compatible metadata.")
    return True


def get_model_filename(model_path, filename="model.pt"):
    return os.path.join(model_path, filename)


def save_checkpoint(
    params, model, optimizer, current_epoch, current_loss, file_path, log=None
):
    params["current_epoch"] = current_epoch
    params["current_loss"] = current_loss

    state = {
        "params": params,
        "optimizer_states": optimizer.state_dict(),
    }

    # Really? This fixes the fact that compiled models store their stuff in a _different_ place!
    if hasattr(model, "_orig_mod"):  # means, model was compiled!
        state["model_states"] = model._orig_mod.state_dict()
    else:  # models was not compiled, 'standard' case.
        state["model_states"] = model.state_dict()

    torch.save(state, file_path)
    if log is not None:
        log.info(f"Saved model to {file_path}")


def load_model_metadata_from_checkpoint(
    params, updatable_params, file_path, device=None, log=None
):
    if not os.path.exists(file_path):
        if log is not None:
            log.info(
                f"No saved state, no {file_path}, starting with default state: {params}"
            )
        return params
    if device is None:
        state = torch.load(file_path)
    else:
        state = torch.load(file_path, map_location=device)
    new_params = state["params"]
    del state
    if metadata_compatible(params, new_params, updatable_params, log) is False:
        if log is not None:
            log.info(f"Metadata incompatible, starting with default state: {params}")
        return params
    for key in updatable_params:
        if key in params:
            new_params[key] = params[key]
    if log is not None:
        log.info(f"Loaded model metadata from {file_path}, {new_params}")
    return new_params


def load_checkpoint(
    params, model, optimizer, file_path, updatable_keys, device=None, log=None
):
    if not os.path.exists(file_path):
        print(f"No saved state, no {file_path}, starting from scratch.")
        if log is not None:
            log.info(
                f"No saved state, no {file_path}, starting new model from scratch with default params {params}."
            )
        return None
    if device is None:
        state = torch.load(file_path)
    else:
        state = torch.load(file_path, map_location=device)
    params_new = state["params"]
    if metadata_compatible(params, params_new, updatable_keys, log) is False:
        print("Metadata incompatible, starting from scratch.")
        del state  # Free memory
        if log is not None:
            log.info(
                f"Metadata incompatible, starting new model with default params {params}."
            )
        return params
    params_old = params
    params = params_new
    model.load_state_dict(state["model_states"])
    optimizer.load_state_dict(state["optimizer_states"])
    for g in optimizer.param_groups:  # Allow for different learning rates
        g["lr"] = params_old["learning_rate"]
    for key in updatable_keys:
        params[key] = params_old[key]
    epoch = params["current_epoch"]
    loss = params["current_loss"]
    print(
        f"Continuing from saved state epoch={epoch+1}, loss={loss:.3f}"
    )  # Save is not necessarily on epoch boundary, so that's approx.
    del state  # Free memory
    if log is not None:
        log.info(f"Continuing from saved state epoch={epoch+1}, loss={loss:.3f}")
    return params
