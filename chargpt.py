"""
Trains a character-level language model.

chargpt trains a character-level language model.

We support three settings: 1 convenience setting and 2 "benchmark" settings that have acedemic literature results:

- a user specified `input.txt` file that we train an LM on (e.g. get tiny-shakespear (1.1MB of data) [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt))
- TODO [text8](http://mattmahoney.net/dc/textdata.html): also derived from Wikipedia text but all XML is removed and is lowercased to only 26 characters of
- TODO [enwik8](http://prize.hutter1.net) benchmark ("Hutter Prize"), first 100M bytes of a Wikipedia XML dump, with 205 unique tokensEnglish plus spaces

"""

import os
import sys

import torch
from torch.utils.data import Dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN


# -----------------------------------------------------------------------------

def get_config() -> CN:

    C = CN()
    C.saveModel = False


    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CN()
    #C.data.block_size = 128
    C.data.block_size = 64

    # model
    C.model = GPT.get_default_config()

    #C.model.model_type = 'gpt-mini'
    #C.model.model_type = 'gpt-micro'
    C.model.model_type = 'gpt-nano'
    #C.model.model_type = 'gpt-pico'

    # trainer
    C.trainer = Trainer.get_default_config()

    #C.trainer.learning_rate = 1e-4
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    #C.trainer.learning_rate = 1e-3 # even faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """


    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------


prompt = "("

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    #setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open('/home/me/sumo/Mid-level-ontology.kif', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 1000 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                x = torch.tensor([train_dataset.stoi[s] for s in prompt], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)

            if config.saveModel:
                # save the latest model
                print("saving model")
                torch.save(model.state_dict(), os.path.join(config.system.work_dir, "model.pt"))

            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
