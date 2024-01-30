import logging
import os
import pickle
import time

import toml
import torch
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import TensorBoardLogger

from models.modeling_opt_lora import OPTLoraForCausalLM, OPTLoraForQuestionAnswering, OPTLoraForSequenceClassification
from tools.checkpoint_load import load_model_chkpt
import pl_model_wrapper

logger = logging.getLogger(__name__)


def test(
    model: torch.nn.Module | torch.fx.GraphModule,
    tokenizer,
    model_info,  # dataclass of model's task type and name
    data_module: pl.LightningDataModule,  # for preparing and loading datasets for pl trainer
    dataset_info,  # dataclass including e.g. number of classes for the pl model wrapper
    task,  # to decide the pl model wrapper of which type should be used
    optimizer,  # optimizer for pl trainer
    learning_rate,  # lr for optimizer. lr_scheduler is default as CosineAnnealingLR
    weight_decay,  # weight_decay for optimizer
    pl_trainer_args,  # args for pl trainer; include e.g. "max_epochs" for setting up lr_scheduler
    auto_requeue,  # for setting up SLURMEnvironment, environment for distributed launch
    save_path,  # path for saving checkpoints
    load_name,  # path to the saved checkpoint
    load_type,  # model checkpoint's type: ['pt', 'pl']
):
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        tb_logger = TensorBoardLogger(save_dir=save_path, name="logs_test")
        pl_trainer_args["callbacks"] = []
        pl_trainer_args["logger"] = tb_logger

    if auto_requeue is not None:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    pl_trainer_args["plugins"] = plugins

    wrapper_pl_model: pl.LightningModule = pl_model_wrapper.get_model_wrapper(
        model_info, task
    )

    # load model from pl checkpoint
    if load_name is None:
        raise ValueError(
            "Path to checkpoint required for resuming training. Please use --load PATH."
        )
    model = load_model_chkpt(load_name, load_type=load_type, model=model)

    # if load_type != "pl":
    #     raise ValueError("Load-type pl is required for resuming training. Please use --load-type pl.")
    logger.warning(
        f"Restore model state from pl checkpoint {load_name}. Entered hyperparameter configuration ignored."
    )

    # if model_info.is_lora:
    #     mark_only_lora_as_trainable(model, bias="none")
    #     if model_info.is_ags:
    #         mark_ags_as_trainable(model)
    #     print_trainable_parameters(model)

    pl_model = wrapper_pl_model.load_from_checkpoint(load_name, model=model)

    logger.warning(f"Resuming hyperparameters: {pl_model.hparams}")

    trainer = pl.Trainer(**pl_trainer_args)

    res_val: dict[str, dict[str, float]] = {}
    # Run validation split
    original_val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]
    res_val["original"] = {
        **original_val_metrics,
        "acc_reduction": 0.0,
        "acc_reduction_rate": 0.0,
    }

    # Run validation with lora matrix modified
    with torch.no_grad():
        assert (
                type(model) is OPTLoraForCausalLM
                or type(model) is OPTLoraForSequenceClassification
                or type(model) is OPTLoraForQuestionAnswering
        )
        model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering
        num_heads: int = model.model.decoder.layers[0].self_attn.num_heads
        head_dim: int = model.model.decoder.layers[0].self_attn.head_dim
        cnt = 0
        for name, param in model.named_parameters():
            if cnt >= 2: break
            if "lora_A" not in name and "lora_B" not in name:
                continue
            if "lora_A" in name:
                continue
            cnt += 1
            print(f"Test count {cnt}")
            corr_name: str = name.replace("lora_B", "lora_A")
            delta_w = torch.matmul(
                param.data,
                model.state_dict()[corr_name].data,  # B (d_out, r) * A (r, d_in)
            )  # shape: (out_features, in_features)

            alpha = 0.9

            if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                # Split by heads
                delta_w_shape = delta_w.size()
                delta_w = delta_w.view(
                    num_heads, head_dim, -1
                )  # shape: (num_heads, d_head, d_model)

                for i in range(num_heads):
                    new_delta_w = torch.cat(
                        (delta_w[:i], (alpha * delta_w[i]).unsqueeze(0), delta_w[i+1:])
                    ).view(delta_w_shape)
                    param.data = new_delta_w

                    logger.warning(f"Apply alpha={alpha} to head {i} of {name}")
                    new_val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]
                    acc_reduction = original_val_metrics["val_acc_epoch"] - new_val_metrics["val_acc_epoch"]
                    res_val[f"{name}_head.{i}"] = {
                        **new_val_metrics,
                        "acc_reduction": acc_reduction,
                        "acc_reduction_rate": acc_reduction / original_val_metrics["val_acc_epoch"],
                    }
            else:
                new_delta_w = alpha * delta_w
                param.data = new_delta_w

                logger.warning(f"Apply alpha={alpha} to {name}")
                new_val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]
                acc_reduction = original_val_metrics["val_acc_epoch"] - new_val_metrics["val_acc_epoch"]
                res_val[name] = {
                    **new_val_metrics,
                    "acc_reduction": acc_reduction,
                    "acc_reduction_rate": acc_reduction / original_val_metrics["val_acc_epoch"],
                }

            param.data = delta_w

    t = time.strftime("%H-%M")
    log_path = f"{save_path}/imp-{t}.toml"
    with open(log_path, "w+") as f:
        toml.dump(res_val, f)

    # if dataset_info.test_split_available:
    #     # Testing
    #     trainer.test(pl_model, datamodule=data_module)
    # elif dataset_info.pred_split_available:
    #     # Predicting, save to predicted_result.pkl
    #     predicted_results = trainer.predict(pl_model, datamodule=data_module)
    #     pred_save_name = os.path.join(save_path, "predicted_result.pkl")
    #     with open(pred_save_name, "wb") as f:
    #         pickle.dump(predicted_results, f)
    #     logger.info(f"Predicted results is saved to {pred_save_name}")
    # else:
    #     raise ValueError(
    #         f"Test or pred split not available for dataset {data_module.name}"
    #     )
