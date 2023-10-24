import logging
import torch


def load_model_chkpt(load_name: str, load_type: str, model: torch.nn.Module = None):
    match load_type:
        case "pt":  # PyTorch type checkpoint
            state_dict = torch.load(load_name)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict=state_dict)
        case "pl":  # Lightning type checkpoint
            src_state_dict = torch.load(load_name)["state_dict"]
            tgt_state_dict = model.state_dict()
            new_tgt_state_dict = {}
            for k, v in src_state_dict.items():
                if "model." in k:
                    possible_tgt_k = ".".join(k.split(".")[1:])
                else:
                    possible_tgt_k = k
                if possible_tgt_k in tgt_state_dict:
                    new_tgt_state_dict[possible_tgt_k] = v
            model.load_state_dict(state_dict=new_tgt_state_dict)
        case _:
            raise ValueError(
                "Only support loading PyTorch or Lightning model checkpoint."
            )

    return model
