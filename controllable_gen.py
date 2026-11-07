from models import BartForTargetSimplification, BartForSimplification
from preprocess import MyDataModuleForTargetSimplificationControllableGen, MyDataModuleForSimplification
import argparse
from transformers import AutoTokenizer
import torch
import numpy as np


def argsPrint(p, bar=30):
    """
    argparseの parse_args() で生成されたオブジェクトを入力すると、
    integersとaccumulateを自動で取得して表示する
    [in] p: parse_args()で生成されたオブジェクト
    [in] bar: 区切りのハイフンの数
    """

    print('-' * bar)
    args = [(i, getattr(p, i)) for i in dir(p) if not '_' in i[0]]
    for i, j in args:
        if isinstance(j, list):
            print('{0}[{1}]:'.format(i, len(j)))
            [print('\t{}'.format(k)) for k in j]
        else:
            print('{0}:\t{1}'.format(i, j))

    print('-' * bar)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", required=True)
    parser.add_argument("-model_ckpt", required=True)
    parser.add_argument("-lr", type=float, required=True)
    parser.add_argument("-val_batch_size", type=int, required=True)
    parser.add_argument("-test_data", required=True)
    parser.add_argument("-max_length", default=64)
    parser.add_argument("-out_path", default="data/out/out.txt")
    parser.add_argument("-norm_rate", required=True, type=float)
    parser.add_argument("-task", default="gen")

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

    datamodule = MyDataModuleForTargetSimplificationControllableGen(args, tokenizer)
    model = BartForTargetSimplification.load_from_checkpoint(args.model_ckpt, args=args, len_tokenizer=len(tokenizer))
    datamodule.setup()

    test_dataset = datamodule.test_dataloader()

    with torch.no_grad():
        for batch in test_dataset:
            ids = model(batch.to('cuda:0'), task=args.task)
            texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
            with open(args.out_path, "a+") as f:
                for text in texts:
                    text = text.replace('\n', '')
                    f.write(f"{text}\n")


if __name__ == "__main__":
    main()
