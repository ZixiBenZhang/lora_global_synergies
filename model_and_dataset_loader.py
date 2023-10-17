import datasets
import pytorch_lightning as pl


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def setup_dataset(args) -> pl.LightningDataModule:
    if args.dataset.lower() in datasets.get_dataset_config_names("glue"):
        task_name = args.dataset.lower()
        dataset, dataset_info = _load_glue(args.dataset.lower)
    else:
        raise NotImplementedError("Currently only GLUE datasets are supported.")
