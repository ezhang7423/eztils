#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# % pip install eztils
import argparse
import os
import sys
import time

import numpy as np

from eztils import abspath
from eztils.run_parallel import BaseHyperParameters, calculate_split


class HyperParameters(BaseHyperParameters):
    _0_gamma = [0.9999]
    _1_lr = [5e-5]  # 2e-3
    _2_on_policy_collected_frames_per_batch___on_policy_minibatch_size___n_minibatch_iters = [
        # (1000, 1000, 45),
        # (1000, 500, 45),
        # (1000, 250, 45),
        # (500, 500, 45),
        (250, 250, 45),
    ]
    _4_seed = [0, 1, 2]
    _5_sigma = [[1] * 7]
    _6_tax_bracket = [[0.0, 10.0, 20.0]]
    _7_tax_rate = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.75],
        [0.0, 0.25, 0.5],
        [0.0, 0.4, 0.7],
        [0.25, 0.5, 0.75],
        [0.1, 0.3, 0.5],
        [0.5, 0.6, 0.7],
        [0.3, 0.4, 0.5],
        [0.0, 0.1, 0.2],
        [0.3, 0.6, 0.8],
    ]  # 10 diff
    _8_entropy_coef = [0.001]


if __name__ == "__main__":
    hparamclass = HyperParameters()
    hparams = HyperParameters.get_product()

    parser = argparse.ArgumentParser(
        description="Run the experiment with dynamic configuration."
    )
    parser.add_argument("--job", type=int, required=True, help="Which job am I?")
    parser.add_argument(
        "--total",
        type=int,
        required=True,
        help="How many total jobs are there running?",
    )

    args = parser.parse_args()

    start_index, end_index = calculate_split(args.total, len(hparams), args.job)
    print(
        "\n\n\n",
        "TOTAL HPARAMS:",
        len(hparams),
        "\n",
        "JOB HPARAMS:",
        end_index - start_index,
        "\n\n\n",
        file=sys.stderr,
    )
    for hparam in hparams[start_index:end_index]:
        (
            gamma,
            lr,
            (collected_frames, minibatch_size, n_minibatch_iters),
            seed,
            tax_bracket,
            tax_rate,
            entropy_coef,
            sigma,
        ) = hparam

        cmd = f"python {abspath()}/<file>.py {hparam}"
        print("Running", cmd, file=sys.stderr)
        os.system(cmd)
