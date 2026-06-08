"""Training CLI entrypoint for funfam contrastive learning."""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Main training entrypoint."""
    L.seed_everything(cfg.seed, workers=True)
    
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    
    logger = instantiate(cfg.logging.logger) if cfg.logging.logger else None
    callbacks = [instantiate(cb) for cb in cfg.logging.callbacks] if cfg.logging.callbacks else []
    
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = L.Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)

    ckpt_path = cfg.run.get("ckpt_path", None)

    if cfg.run.get("eval_only", False):
        if ckpt_path is None:
            raise ValueError("run.eval_only=true requires run.ckpt_path to be set")
        print(f"📊 Evaluating checkpoint: {ckpt_path}")
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
    else:
        if ckpt_path:
            print(f"🔄 Resuming training from: {ckpt_path}")
        else:
            print("✅ Starting trainer.fit()")
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        if cfg.run.do_test_after_fit:
            trainer.test(model, datamodule=datamodule, ckpt_path="best", weights_only=False)


if __name__ == "__main__":
    main()
