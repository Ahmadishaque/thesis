import os

# Deterministic
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags

import jax
jax.config.update("jax_default_matmul_precision", "highest")

import train
import eval


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/pirate.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    # if FLAGS.config.mode == "train":
    #         # 1. Generate the Mask once at the very start
    #         # import ntk_sap
    #         import pi_sap as ntk_sap
    #         from jax import random
            
    #         # Setup dummy domain for pruning (usually window 1 range)
    #         prune_rng = random.PRNGKey(FLAGS.config.seed)
    #         saliency = ntk_sap.compute_ntk_saliency(
    #             prune_rng, 
    #             FLAGS.config, 
    #             time_window_domain=[0.0, 1.0] 
    #         )
            
    #         # Ensure get_ntk_sap_mask also uses the new config path
    #         mask = ntk_sap.get_ntk_sap_mask(
    #             saliency, 
    #             prune_ratio=FLAGS.config.pruning_ratio
    #         )

    #         # 2. Pass the mask to the training function
    #         train.train_and_evaluate(FLAGS.config, FLAGS.workdir, mask=mask)

    if FLAGS.config.mode == "train":
        # Don't generate the mask here anymore
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval":
        eval.evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
