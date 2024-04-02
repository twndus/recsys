import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from dataset.ml_dataset import ML1MDataset
from dataloader.ml_dataloader import ML1MDataLoader
from trainer import Trainer

@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:

    # load dataset
    logger.info('load dataset...')
    dataset = ML1MDataset(cfg)
    train_data, valid_data, test_data = dataset.split_and_get_data()

    # dataloader
    logger.info('datasetloader...')
    train_dataloader = ML1MDataLoader(cfg, train_data)
    valid_dataloader = ML1MDataLoader(cfg, valid_data)
    test_dataloader = ML1MDataLoader(cfg, test_data, train=False)

    # train
    trainer = Trainer(cfg)
    trainer.train(train_dataloader, valid_dataloader)

    # evaluate 
    trainer.evaluate(test_dataloader)

if __name__ == "__main__":
    main()
