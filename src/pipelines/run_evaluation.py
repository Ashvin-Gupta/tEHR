import argparse
import yaml
import torch
import logging
from src.training.classification_trainer import EHRClassificationTrainer

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    trainer = EHRClassificationTrainer(config)
    trainer.load_pretrained_model()
    trainer.create_classifier()
    trainer.classifier_model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")
    )
    _, val_dataset, test_dataset = trainer.prepare_datasets()
    collator = trainer.create_collator()
    trainer.create_trainer(val_dataset, val_dataset, collator, run_name="eval")
    trainer.evaluate_and_visualize(val_dataset, test_dataset)

if __name__ == "__main__":
    main()