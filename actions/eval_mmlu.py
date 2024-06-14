import ast
import json
import logging

import torch
import wandb
from accelerate import (
    dispatch_model,
    infer_auto_device_map,
)
from lm_eval.evaluator import simple_evaluate as evaluate_harness_downstream
from lm_eval.utils import make_table as harness_make_table

from tools.checkpoint_load import load_model_chkpt

logger = logging.getLogger(__name__)


def run_evaluate_harness_downstream(
        model: torch.nn.Module | torch.fx.GraphModule,
        load_name,  # path to the saved checkpoint
        load_type,  # model checkpoint's type: ['pt', 'pl']
        save_path,  # path for saving checkpoints
        eval_config: dict,
):
    logger.warning("Running MMLU testing")
    # Load model from checkpoint
    if load_name is None:
        raise ValueError(
            "Path to checkpoint required for resuming training. Please use --load PATH."
        )
    model = load_model_chkpt(load_name, load_type=load_type, model=model)
    logger.info(f"Loading model state from checkpoint {load_name}.")

    # Dispatch model
    device_map = create_device_map(
        model,
        eval_config.get("max_memory", None),
        eval_config.get("device_map", None),
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model = dispatch_model(model, device_map=device_map)

    # Run evaluation with lm-eval
    results = evaluate_harness_downstream(
        model,
        tasks=eval_config["datasets"],
        num_fewshot=eval_config.get("num_fewshot", 0),
        use_cache=not eval_config.get("no_cache", False),
        batch_size=eval_config.get("batch_size", None),
    )

    # Save result as JSON
    dumped = json.dumps(results, indent=4)
    save_path = save_path.joinpath("harness_results.json")
    if save_path.exists():
        save_path = save_path.parent.joinpath(
            f"harness_results_{len(list(save_path.glob('harness_results_*.json')))}.json"
        )
    with open(save_path, "w") as f:
        f.write(dumped)
    logger.info(f"results saved to {save_path}")
    logger.info("\n" + harness_make_table(results))

    # Log with WandB
    table = wandb.Table(columns=["dataset", "accuracy"])
    task_cnt = 0
    accu_sum = 0
    for task in eval_config["datasets"]:
        task_acc = results["results"][task]["acc"]
        accu_sum += task_acc
        task_cnt += 1
        table.add_data(task, task_acc)
        wandb.run.summary[f"{task}_acc"] = task_acc
    wandb.run.summary["avg_harness_acc"] = accu_sum / task_cnt
    wandb.log({"harness_downstream_results": table})


def create_device_map(model, max_memory, device_map) -> dict[str, int]:
    if max_memory is not None:
        max_memory = ast.literal_eval(max_memory.removeprefix(":ast:"))

    if device_map is not None:
        if device_map == "auto":
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=model._no_split_modules,
                max_memory=max_memory,
            )
        elif isinstance(device_map, str):
            device_map = ast.literal_eval(device_map.removeprefix(":ast:"))
        else:
            assert isinstance(device_map, dict)
    elif max_memory is not None:
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=model._no_split_modules,
        )
    else:
        device_map = infer_auto_device_map(
            model, no_split_module_classes=model._no_split_modules
        )
    return device_map
