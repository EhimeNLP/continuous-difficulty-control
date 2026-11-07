from models import BartForTargetSimplification
from preprocess import MyDataModuleForGradePrediction
import argparse
from transformers import AutoTokenizer
import torch
import numpy as np
from utils import argsPrint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", required=True)
    parser.add_argument("-model_ckpt", required=True)
    parser.add_argument("-lr", type=float, required=True)
    parser.add_argument("-val_batch_size", type=int, required=True)
    parser.add_argument("-test_data", required=True)
    parser.add_argument("-max_length", default=64)
    parser.add_argument("-out_path", default="data/out/out.txt")
    parser.add_argument("-norm_rate", type=float)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # コマンドライン引数をprint
    argsPrint(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # add special tokens to tokenizer
    special_tokens_dict = {"cls_token": "<cls>"}
    tokenizer.add_special_tokens(special_tokens_dict)

    datamodule = MyDataModuleForGradePrediction(args, tokenizer)
    datamodule.setup()
    model = BartForTargetSimplification.load_from_checkpoint(args.model_ckpt, args=args, len_tokenizer=len(tokenizer))

    test_dataset = datamodule.test_dataloader()

    grades = []
    with torch.no_grad():
        for batch in test_dataset:
            norm: torch.Tensor = model(batch.to('cuda:0'), task="norm")
            norm = norm.view(-1).to('cpu').tolist()
            grades += norm
    with open(args.out_path, "w") as f:
        for grade in grades:
            f.write(f"{grade}\n")


if __name__ == "__main__":
    main()
