from pytorch_lightning import LightningDataModule, LightningModule
import torch
import importlib
import warnings
import yaml
import os
from huggingface_hub import hf_hub_download

# from datasets.lightning_data_module import LightningDataModule
# from training.lightning_module import LightningModule

def _expand_env_vars(obj):
    """Recursively expand environment variables in config strings.
    
    Supports both ${VAR} and $VAR syntax via os.path.expandvars.
    Example: "${CHECKPOINT_DIR}/checkpoints" -> "/path/to/checkpoints/checkpoints"
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj

def load_config(config_path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Expand environment variables in all string values
    config = _expand_env_vars(config)
    return config

def build_data(config, data_path, kwargs=None) -> "LightningDataModule":
    module_name, cls_name = config["data"]["class_path"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    data_module_cls = getattr(module, cls_name)
    kwargs = config["data"].get("init_args", {})  
    
    data = data_module_cls(
        path=data_path,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **kwargs,
    ).setup()
    return data


def __load_local_state_dict(ckpt_path, map_location):
    state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict

def __build_model(config, num_classes, img_size, device, model_kwargs_override=None) -> "LightningModule":
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)

    # If available, pass the list of "stuff" classes
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = (
        lit_cls(
            img_size=img_size,
            num_classes=num_classes,
            network=network,
            **model_kwargs,
        )
        .eval()
        .to(device)
    )

    return model

def build_model_ckpt(config, ckpt_path, num_classes, img_size, device) -> "LightningModule":
    model = __build_model(config, num_classes, img_size, device)

    # Load the pretrained weights from the local checkpoint
    state_dict = __load_local_state_dict(ckpt_path, map_location=device)
    incompatible = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint: {ckpt_path}")
    print("Missing keys:", len(incompatible.missing_keys))
    print("Unexpected keys:", len(incompatible.unexpected_keys))

    return model

def build_model_hf(config, num_classes, img_size, device) -> "LightningModule":
    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")

    if name is None:
        raise ValueError("Model name not specified in config for Hugging Face Hub loading.")

    state_dict_path = hf_hub_download(
        repo_id=f"tue-mps/{name}",
        filename="pytorch_model.bin",
    )

    # With dino V3 the checkpoint loading is handles by the model class
    is_dinov3 = "dinov3" in name

    model_kwargs = {}
    if is_dinov3:
        model_kwargs["ckpt_path"] = state_dict_path
        model_kwargs["delta_weights"] = True

    model = __build_model(config, num_classes, img_size, device, model_kwargs_override=model_kwargs)

    if not is_dinov3:
        state_dict = torch.load(
            state_dict_path, map_location=f"cuda:{device}", weights_only=True
        )
        model.load_state_dict(state_dict, strict=False)

    return model

