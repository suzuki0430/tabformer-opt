import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import os
import json
import random
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import mean_squared_error
from transformers import AdamW
# import wandb
from tqdm.notebook import tqdm
from dataset.vocab import AttrDict
from dataset.action_history import FineTuningActionHistoryDataset
from dataset.datacollator import FineTuningDataCollatorForLanguageModeling
from misc.utils import random_split_dataset
from models.common import CommonModel
from args import define_fine_tuning_parser
import tarfile
from os import path

device = torch.device("cuda")
scaler = torch.cuda.amp.GradScaler()

SEED = 0
N_FOLDS = 5
MAX_LEN = 320

LR = 2e-5
WEIGHT_DECAY = 1e-6
# N_EPOCHS = 8
N_EPOCHS = 1
WARM_UP_RATIO = 0.1

BS = 32
ACCUMULATE = 1
MIXED_PRECISION = False

EXP_NAME = 'baseline'

def set_seed(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validation_loop(valid_loader, model):
    model.eval()
    preds = []

    true = []

    for d in valid_loader:
        with torch.no_grad():
            logits = model(
                d["input_ids"].to(device),
                attention_mask=None,
                token_type_ids=None
            )

        # preds.append(logits[:, 0])
        preds.append(logits)
        true.append(d["label"].float().to(device))
        
    # print("preds",a)
    # print("true",len(true))
    y_pred = torch.hstack(preds).cpu().numpy() # tensor連結してndarrayに変換
    y_true = torch.hstack(true).cpu().numpy()
    
    mse_loss = mean_squared_error(y_true, y_pred, squared=False)

    return mse_loss


def main(args):
    key = os.getenv('SM_MODEL_DIR')
    
    if key :
        with tarfile.open(name=path.join(args.model_path, f'model.tar.gz'), mode="r:gz") as mytar:
            mytar.extractall(path.join(args.model_path, f'model'))
            
            token2id_file = path.join(args.model_path, f"model/vocab_token2id.bin")
            vocab_file = path.join(args.model_path, f"model/vocab.nb")
            pretrained_model = path.join(args.model_path, f"model/checkpoint-500/pytorch_model.bin")
            pretrained_config = path.join(args.model_path, f"model/checkpoint-500/config.json")
            encoder_fname = path.join(args.model_path, f"model/{args.data_fname}.encoder_fit.pkl")
    else :
            vocab_file = path.join(args.model_path, f"vocab.nb")
            token2id_file = path.join(args.model_path, f"vocab_token2id.bin")
            pretrained_model = path.join(args.model_path, f"checkpoint-500/pytorch_model.bin")
            pretrained_config = path.join(args.model_path, f"checkpoint-500/config.json")
            encoder_fname = path.join(args.model_path, f"{args.data_fname}.encoder_fit.pkl")
    
    # Datasets
    dataset = FineTuningActionHistoryDataset(
            root=args.data_root,
            fname=args.data_fname, # pretraingのファイルとは別
            vocab_dir="./output_pretraining/action_history/",
            fextension="",
            nrows=None,
            user_ids=None,
            mlm=True,                
            stride=10,
            flatten=True,
            return_labels=True,
            skip_user=False,
            token2id_file=token2id_file,
            encoder_fname=encoder_fname)

    totalN = len(dataset)
    trainN = int(0.80 * totalN)
    valN = totalN - trainN

    assert totalN == trainN + valN

    lengths = [trainN, valN, 0]
    train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)

    # DataCollator
    keys = ["unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
    special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[BOS]", "[EOS]"]
    special_field_tag = "SPECIAL"
    special_tokens_map = {}

    for key, token in zip(keys, special_tokens):
        token = "%s_%s" % (special_field_tag, token)
        special_tokens_map[key] = token

    tok = BertTokenizer(
        vocab_file=vocab_file, 
        do_lower_case=False,
        **AttrDict(special_tokens_map))

    data_collator = FineTuningDataCollatorForLanguageModeling(tokenizer=tok)

    # DataLoader
    train_loader = DataLoader(
                        train_dataset,
                        collate_fn=data_collator,
                        batch_size=BS,
                        pin_memory=True, 
                        shuffle=True, 
                        drop_last=True, 
                        num_workers=0)

    valid_loader = DataLoader(
                        eval_dataset, 
                        collate_fn=data_collator,
                        batch_size=BS,
                        pin_memory=True, 
                        shuffle=False, 
                        # drop_last=False,
                        drop_last=True, 
                        num_workers=0)
    # set models
    model = CommonModel(pretrained_config, pretrained_model)
    model.to(device)

    # freeze parameters in all network
    for name, param in model.named_parameters():
        param.requires_grad = False

    # activate parameters in only lstm network
    for name, param in model.lstm.named_parameters():
        param.requires_grad = True

    # activate parameters in only linear network
    for name, param in model.regressor.named_parameters():
        param.requires_grad = True

    # set optimizer
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    max_train_steps = N_EPOCHS * len(train_loader)
    warmup_steps = int(max_train_steps * WARM_UP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps
    )

    # wandb.init(project='CommonLit', entity='trtd56', name=EXP_NAME)
    # wandb.watch(model)

    bar = tqdm(total=max_train_steps)

    set_seed()
    optimizer.zero_grad()
    train_iter_loss, valid_best_loss, all_step = 0, 999, 0
    for epoch in range(N_EPOCHS):
        for d in train_loader:
            all_step += 1
            model.train()

            if MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    logits = model(
                        d["input_ids"].to(device),
                        attention_mask=None,
                        token_type_ids=None
                    )
                    loss = model.loss_fn(logits, d["label"].float().to(device))
                    loss = loss / ACCUMULATE
            else:
                logits = model(
                    d["input_ids"].to(device),
                    attention_mask=None,
                    token_type_ids=None
                )

                loss = model.loss_fn(logits, d["label"].float().to(device))
                loss = loss / ACCUMULATE

            train_iter_loss += loss.item()

            if MIXED_PRECISION:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if all_step % ACCUMULATE == 0:
                if MIXED_PRECISION:
                    scaler.step(optimizer) 
                    scaler.update() 
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                valid_loss = validation_loop(valid_loader, model)
                if valid_best_loss > valid_loss:  
                    valid_best_loss = valid_loss

                # wandb.log({
                #     "train_loss": train_iter_loss,
                #     "valid_loss": valid_loss,
                #     "valid_best_loss": valid_best_loss,
                # })
                train_iter_loss = 0
            bar.update(1)
    # wandb.finish()

    sample_input_ids = torch.tensor([train_dataset.__getitem__(0)[0].tolist()], dtype=torch.long)
    traced_model = torch.jit.trace(model, [sample_input_ids.to(device)])
    torch.jit.save(traced_model, path.join(args.output_dir, f'fine_tuning_model.pt'))
    
    # torch.save(model.state_dict(), path.join(args.output_dir, f'fine_tuning_model.pth'))
if __name__ == "__main__":

    parser = define_fine_tuning_parser()
    opts = parser.parse_args()
    
    if "SM_HPS" in os.environ.keys():
        hps = json.loads(os.environ["SM_HPS"])
        for key, value in hps.items():
            if opts.__contains__(key):
                opts.__setattr__(key, value)

    main(opts)