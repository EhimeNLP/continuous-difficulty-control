from transformers import AutoTokenizer
import argparse
from models import (
    BartForGradeTraining
)
from pytorch_lightning.loggers import TensorBoardLogger
from preprocess import MyDataModule
import pytorch_lightning as pl


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
    parser.add_argument("-lr", type=float, required=True)
    parser.add_argument("-train_batch_size", type=int, required=True)
    parser.add_argument("-val_batch_size", type=int, required=True)
    parser.add_argument("-train_data", required=True)
    parser.add_argument("-val_data", required=True)
    parser.add_argument("-test_data", required=True)
    parser.add_argument("-model_path", required=True)
    parser.add_argument("-weight_gen_loss", required=True, type=float)
    parser.add_argument("-norm_rate", required=True, type=float)

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

    datamodule = MyDataModule(args, tokenizer)
    model = BartForGradeTraining(args, len(tokenizer))

    logger = TensorBoardLogger("tb_logs")

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=args.model_path,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True
    )
    earlystopping = pl.callbacks.EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=3
    )
    trainer = pl.Trainer(
        max_epochs=100,
        devices=[0],
        callbacks=[checkpoint, earlystopping],
        enable_progress_bar=False,
        logger=logger
    )

    trainer.fit(model, datamodule=datamodule)
    print("ベストモデル: ", checkpoint.best_model_path)


if __name__ == "__main__":
    main()
