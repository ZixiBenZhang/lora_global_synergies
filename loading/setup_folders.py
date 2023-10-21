import time
from pathlib import Path


def setup_folder() -> Path:
    project = None
    if self.args.project is not None:
        project = self.args.project
    else:
        # No project name is given; so we construct one structured as follows:
        # {MODEL-NAME}_{TASK-TYPE}_{DATASET-NAME}_{TIMESTAMP}
        project = "{}_{}_{}_{}".format(
            self.args.model.replace("/", "-"),
            self.args.task,
            self.args.dataset,
            time.strftime("%Y-%m-%d"),
        )
        setattr(self.args, "project", project)

    output_dir = Path(self.args.project_dir) / project
    # self.logger.info(f"Project will be created at {output_dir}")
    return output_dir
