# Simple script to run all the wandb syncs for prior runs
# Say we're using uv as in the main code, and some runs for a day are logged to outputs/2025-07-08
# uv run push_logs_to_wandb.py --base_output_path=./outputs/2025-07-08
# has some control logic for what you want to sync,
# but that's from a different repo's artifact mgmt style. fixme

import os
import glob

DRY_RUN = True
# DRY_RUN = False

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_output_path", type=str, required=True)
args = parser.parse_args()

BASE_OUTPUT_PATH = args.base_output_path

ENUM_DIR = os.listdir(BASE_OUTPUT_PATH)

ENUM_DIR = sorted(ENUM_DIR)
EXP_LIST = ENUM_DIR

key_strings = [
    "",
]

exclude_strings = [
    # "",
]


def filter_exp_list(exp_list, key_strings, exclude_strings=[]):
    filtered_exp_list = []
    for exp in exp_list:
        for key_string in key_strings:
            if key_string in exp and all(
                [exclude_string not in exp for exclude_string in exclude_strings]
            ):
                filtered_exp_list.append(exp)
    return filtered_exp_list


EXP_LIST = filter_exp_list(EXP_LIST, key_strings, exclude_strings)


print(len(EXP_LIST))
for exp in EXP_LIST:
    print(exp)

for EXP in EXP_LIST:
    RUN_DIR = EXP
    wandb_parents = glob.glob(f"{BASE_OUTPUT_PATH}/{RUN_DIR}/**/wandb", recursive=True)
    wandb_parents = [
        pth.replace("/wandb", "") for pth in wandb_parents if pth.endswith("/wandb")
    ]
    assert len(wandb_parents) == 1

    command = f"""\
    cd {wandb_parents[0]} && \
    wandb sync --sync-all .
    """
    if DRY_RUN:
        print(command)
    else:
        os.system(command)
