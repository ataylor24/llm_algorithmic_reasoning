import argparse
import utils
from Model import GenerativeModel
from Dataset import AlgReasoningDataset
from torch.utils.data import DataLoader

def _load_data():
    training_dataloader = DataLoader(AlgReasoningDataset(config, "training"), shuffle=True)
    validation_dataloader = DataLoader(AlgReasoningDataset(config, "validation"), shuffle=True)
    testing_dataloader = DataLoader(AlgReasoningDataset(config, "testing"), shuffle=True)

    return training_dataloader, validation_dataloader, testing_dataloader

def main():
    training_dataloader, validation_dataloader, testing_dataloader = _load_data()
    model = GenerativeModel(config)
    
    #training loop
    for i, inst in enumerate(training_dataloader):
        # Process your batch
        results = model(inst, num_return_sequences=1, eos_token_id=model.tokenizer.eos_token_id, max_length=512)[0]["generated_text"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read configuration JSON file')
    parser.add_argument('config', type=str, help='Path to the configuration JSON file')
    args = parser.parse_args()

    config = utils.load_json(args.config)
    config.update(args.__dict__)
    global config
    config = argparse.Namespace(**config)
    main()