import os
import neptune
from typing import Literal


def init_neptune(tags: list, algo: Literal["ras", "aas"] = "ras", mode: Literal["async", "sync", "offline", "read-only", "debug"] = "async", cfg_path: str = os.getenv('CONFIG_FILE_PATH')) -> neptune.Run:
    # Get tags
    tags = list(map(str, tags))
    # Initialize neptune run
    run = neptune.init_run(
        project=os.getenv('NEPTUNE_PROJECT'),
        api_token=os.getenv('NEPTUNE_TOKEN'),
        tags=tags,
        mode=mode
    )
    # Upload source code
    if algo == "ras":
        run["src"].upload_files([
            './rl_autoschedular',
            './utils',
            './train_ras.py'
        ])
    else:
        run["src"].upload_files([
            './aas',
            './utils',
            './train_aas.py'
        ])
    # Upload config
    run["config"].upload(cfg_path)
    return run
