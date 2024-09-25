import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os

sys.path.append(os.getcwd())
import src.generative as generative
import src.detection as detection
from src.utils import run_exps


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    output_dir = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/results.csv"
    )
    run_exps(
        p=getattr(generative, cfg.dataset.name)(**cfg.dataset.params),
        N_train=cfg.dataset.N_train,
        N_test=cfg.dataset.N_test,
        q=getattr(generative, cfg.model.name)(),
        model_params=dict(cfg.model.params),
        dcd_params=dict(cfg.detector.DataCopyDetector.params),
        tsd_params=dict(cfg.detector.ThreeSampleDetector.params),
        n_trials=cfg.n_trials,
        verbose=True,
        save=output_dir,
    )


if __name__ == "__main__":
    main()
