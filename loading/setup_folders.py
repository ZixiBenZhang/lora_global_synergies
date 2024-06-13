import time
from pathlib import Path


def setup_folder(args) -> Path:
    project = None
    if args.project is not None:
        project = args.project
    else:
        # No project name is given; so we construct one structured as follows:
        # {MODEL-NAME}_{TASK-TYPE}_{DATASET-NAME}_{TIMESTAMP}
        project = "{}_{}_{}_{}".format(
            args.model.replace("/", "-"),
            args.task,
            args.dataset.replace("/", "-") if args.mmlu_mode is None else args.dataset.replace("/", "-") + "-mmlu",
            time.strftime("%Y-%m-%d"),
        )
        setattr(args, "project", project)

    output_dir = Path(args.project_dir) / project
    # self.logger.info(f"Project will be created at {output_dir}")
    return output_dir
