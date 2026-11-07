import torch
import pytorch_lightning as pl
from transformers import (
    BartModel,
    T5Model,
)
import torch.nn as nn
from transformers.models.bart.modeling_bart import shift_tokens_right
from utils import Args
import numpy as np
from transformers.generation import GenerationMixin


class BartForGradeTraining(pl.LightningModule):
    def __init__(self, args: Args, len_tokenizer: int):
        super().__init__()
        self.args = args
        model = BartModel.from_pretrained(args.model_name)
        model.resize_token_embeddings(len_tokenizer)
        self.model = model
        self.lm_head = nn.Linear(self.model.config.d_model, self.model.shared.num_embeddings, bias=False)
        self.lm_head.weight = self.model.decoder.get_input_embeddings().weight

    def training_step(self, batch):
        grade = batch["grade"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # generation loss
        logits: torch.Tensor = self.lm_head(outputs[0])
        loss_fn = nn.CrossEntropyLoss()
        gen_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), input_ids.view(-1))

        # grade loss
        cls_embs = outputs.encoder_last_hidden_state[:, 0, :]
        grade_loss = self.calculate_grade_loss(cls_embs, grade)

        loss = grade_loss + self.args.weight_gen_loss * gen_loss
        self.log_dict({"train/loss": loss, "train/grade_loss": grade_loss, "train/gen_loss": gen_loss})
        return loss

    def validation_step(self, batch):
        grade = batch["grade"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # generation loss
        logits = self.lm_head(outputs[0])
        loss_fn = nn.CrossEntropyLoss()
        gen_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), input_ids.view(-1))

        # grade loss
        cls_embs = outputs.encoder_last_hidden_state[:, 0, :]
        grade_loss = self.calculate_grade_loss(cls_embs, grade)

        loss = grade_loss + self.args.weight_gen_loss * gen_loss
        self.log_dict({"val/loss": loss, "val/grade_loss": grade_loss, "val/gen_loss": gen_loss})
        return loss

    def test_step(self, batch):
        return batch

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def calculate_grade_loss(self, cls_emb, grade):
        norm = torch.norm(cls_emb, dim=1)
        grade = grade.float()

        criterion = nn.MSELoss()
        grade_loss = criterion(norm, grade)
        return grade_loss


class BartForSimplification(pl.LightningModule):
    def __init__(self, args: Args, len_tokenizer: int):
        super().__init__()
        self.args = args
        model = BartModel.from_pretrained(args.model_name)
        model.resize_token_embeddings(len_tokenizer)
        self.model = model
        self.lm_head = nn.Linear(self.model.config.d_model, self.model.shared.num_embeddings, bias=False)
        self.lm_head.weight = self.model.decoder.get_input_embeddings().weight

    def training_step(self, batch):
        grade = batch["grade"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        half_batch_size = input_ids.size()[0] // 2

        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # grade loss
        cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
        src_cls_embs, trg_cls_embs = torch.chunk(cls_embs.contiguous(), 2, dim=0)
        src_grade, _ = torch.chunk(grade, 2, dim=0)
        grade_loss = self.calculate_grade_loss(src_cls_embs, src_grade)

        # shrink norm
        src_norm = torch.norm(src_cls_embs, dim=1)
        trg_norm = torch.norm(trg_cls_embs, dim=1)
        shrink_rate = (trg_norm / src_norm).unsqueeze(1).unsqueeze(2)
        last_hidden_state = encoder_outputs.last_hidden_state[:half_batch_size, 0:1, :].detach().clone()
        last_hidden_state = last_hidden_state * shrink_rate
        encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:half_batch_size, 1:, :]), dim=1)
        decoder_input_ids = batch['input_ids']
        decoder_input_ids = decoder_input_ids[half_batch_size:, :]
        labels = batch['input_ids'][half_batch_size:, :]
        decoder_input_ids = shift_tokens_right(
            input_ids=decoder_input_ids,
            pad_token_id=self.model.config.pad_token_id,
            decoder_start_token_id=self.model.config.decoder_start_token_id
        )

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=attention_mask[:half_batch_size, :]
        )

        # generation loss
        logits: torch.Tensor = self.lm_head(decoder_outputs[0])
        loss_fn = nn.CrossEntropyLoss()
        gen_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        loss = grade_loss + self.args.weight_gen_loss * gen_loss
        self.log_dict({"train/loss": loss, "train/grade_loss": grade_loss, "train/gen_loss": gen_loss})
        return loss

    def validation_step(self, batch):
        grade = batch["grade"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        half_batch_size = input_ids.size()[0] // 2

        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # grade loss
        cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
        src_cls_embs, trg_cls_embs = torch.chunk(cls_embs.contiguous(), 2, dim=0)
        src_grade, _ = torch.chunk(grade, 2, dim=0)
        grade_loss = self.calculate_grade_loss(src_cls_embs, src_grade)

        # shrink norm
        src_norm = torch.norm(src_cls_embs, dim=1)
        trg_norm = torch.norm(trg_cls_embs, dim=1)
        shrink_rate = (trg_norm / src_norm).unsqueeze(1).unsqueeze(2)
        last_hidden_state = encoder_outputs.last_hidden_state[:half_batch_size, 0:1, :].detach().clone()
        last_hidden_state = last_hidden_state * shrink_rate
        encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:half_batch_size, 1:, :]), dim=1)
        decoder_input_ids = batch['input_ids']
        decoder_input_ids = decoder_input_ids[half_batch_size:, :]
        labels = batch['input_ids'][half_batch_size:, :]
        decoder_input_ids = shift_tokens_right(
            input_ids=decoder_input_ids,
            pad_token_id=self.model.config.pad_token_id,
            decoder_start_token_id=self.model.config.decoder_start_token_id
        )

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=attention_mask[:half_batch_size, :]
        )

        # generation loss
        logits: torch.Tensor = self.lm_head(decoder_outputs[0])
        loss_fn = nn.CrossEntropyLoss()
        gen_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        loss = grade_loss + self.args.weight_gen_loss * gen_loss
        self.log_dict({"val/loss": loss, "val/grade_loss": grade_loss, "val/gen_loss": gen_loss})
        return loss

    def forward(self, batch, task="gen"):
        if task == "gen" or task == "trg_norm":
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            half_batch_size = input_ids.size()[0] // 2

            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

            cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
            src_cls_embs, trg_cls_embs = torch.chunk(cls_embs.contiguous(), 2, dim=0)
            src_norm = torch.norm(src_cls_embs, dim=1)
            trg_norm = torch.norm(trg_cls_embs, dim=1)
            shrink_rate = (trg_norm / src_norm).unsqueeze(1).unsqueeze(2)
            last_hidden_state = encoder_outputs.last_hidden_state[:half_batch_size, 0:1, :]
            last_hidden_state = last_hidden_state * shrink_rate
            encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:half_batch_size, 1:, :]), dim=1)

            decoder_input_ids = torch.tensor(np.full((half_batch_size, 1), self.model.config.decoder_start_token_id)).to('cuda:0')

            for _ in range(self.args.max_length):
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_state,
                    encoder_attention_mask=attention_mask[:half_batch_size, :]
                )

                logits: torch.Tensor = self.lm_head(decoder_outputs[0])
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to('cuda:0')

                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to('cuda:0')

            if task == "gen":
                return decoder_input_ids
            return decoder_input_ids, trg_norm

        if task == "grade_norm":
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            half_batch_size = input_ids.size()[0] // 2

            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

            cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
            src_cls_embs, trg_cls_embs = torch.chunk(cls_embs.contiguous(), 2, dim=0)
            src_norm = torch.norm(src_cls_embs, dim=1)
            trg_norm = torch.norm(trg_cls_embs, dim=1)

            return input_ids, src_norm, trg_norm

        if task == "continuous_gen":
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            grade = batch["grade"]
            half_batch_size = input_ids.size()[0] // 2

            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

            cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
            src_cls_embs, trg_cls_embs = torch.chunk(cls_embs.contiguous(), 2, dim=0)
            src_norm = torch.norm(src_cls_embs, dim=1)
            trg_norm = torch.norm(trg_cls_embs, dim=1)
            shrink_rate = ((grade - 0.5) / src_norm).unsqueeze(1).unsqueeze(2)
            last_hidden_state = encoder_outputs.last_hidden_state[:half_batch_size, 0:1, :]
            last_hidden_state = last_hidden_state * shrink_rate
            encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:half_batch_size, 1:, :]), dim=1)

            decoder_input_ids = torch.tensor(np.full((half_batch_size, 1), self.model.config.decoder_start_token_id)).to('cuda:0')

            for _ in range(self.args.max_length):
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_state,
                    encoder_attention_mask=attention_mask[:half_batch_size, :]
                )

                logits: torch.Tensor = self.lm_head(decoder_outputs[0])
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to('cuda:0')

                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to('cuda:0')
            return decoder_input_ids

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def calculate_grade_loss(self, cls_emb, grade):
        norm = torch.norm(cls_emb, dim=1)
        grade = grade.float()

        criterion = nn.MSELoss()
        grade_loss = criterion(norm, grade)
        return grade_loss


class BartForTargetSimplification(pl.LightningModule):
    def __init__(self, args: Args, len_tokenizer: int):
        super().__init__()
        self.args = args
        model = BartModel.from_pretrained(args.model_name)
        model.resize_token_embeddings(len_tokenizer)
        self.model = model
        self.lm_head = nn.Linear(self.model.config.d_model, self.model.shared.num_embeddings, bias=False)
        self.lm_head.weight = self.model.decoder.get_input_embeddings().weight

    def training_step(self, batch):
        src_grade = batch["src_grade"]
        trg_grade = batch["trg_grade"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # grade loss
        cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
        grade_loss = self.calculate_grade_loss(cls_embs, src_grade)

        # shrink norm
        norm = torch.norm(cls_embs, dim=1)
        shrink_rate = (trg_grade / norm).unsqueeze(1).unsqueeze(2)
        last_hidden_state = encoder_outputs.last_hidden_state[:, 0:1, :]
        last_hidden_state = last_hidden_state * shrink_rate
        encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:, 1:, :]), dim=1)
        decoder_input_ids = batch['labels']
        labels = batch['labels']
        decoder_input_ids = shift_tokens_right(
            input_ids=decoder_input_ids,
            pad_token_id=self.model.config.pad_token_id,
            decoder_start_token_id=self.model.config.decoder_start_token_id
        )

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=attention_mask
        )

        # generation loss
        logits: torch.Tensor = self.lm_head(decoder_outputs[0])
        loss_fn = nn.CrossEntropyLoss()
        gen_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        loss = grade_loss + self.args.weight_gen_loss * gen_loss
        self.log_dict({"train/loss": loss, "train/grade_loss": grade_loss, "train/gen_loss": gen_loss})
        return loss

    def validation_step(self, batch):
        src_grade = batch["src_grade"]
        trg_grade = batch["trg_grade"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # grade loss
        cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
        grade_loss = self.calculate_grade_loss(cls_embs, src_grade)

        # shrink norm
        norm = torch.norm(cls_embs, dim=1)
        shrink_rate = (trg_grade / norm).unsqueeze(1).unsqueeze(2)
        last_hidden_state = encoder_outputs.last_hidden_state[:, 0:1, :]
        last_hidden_state = last_hidden_state * shrink_rate
        encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:, 1:, :]), dim=1)
        decoder_input_ids = batch['labels']
        labels = batch['labels']
        decoder_input_ids = shift_tokens_right(
            input_ids=decoder_input_ids,
            pad_token_id=self.model.config.pad_token_id,
            decoder_start_token_id=self.model.config.decoder_start_token_id
        )

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=attention_mask
        )

        # generation loss
        logits: torch.Tensor = self.lm_head(decoder_outputs[0])
        loss_fn = nn.CrossEntropyLoss()
        gen_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        loss = grade_loss + self.args.weight_gen_loss * gen_loss
        self.log_dict({"val/loss": loss, "val/grade_loss": grade_loss, "val/gen_loss": gen_loss})
        return loss

    def forward(self, batch, task="gen"):
        if task == "gen" or task == "trg_norm":
            trg_grade = batch["trg_grade"]
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            batch_size = input_ids.size()[0]

            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

            cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
            norm = torch.norm(cls_embs, dim=1)
            shrink_rate = (trg_grade / norm).unsqueeze(1).unsqueeze(2)
            last_hidden_state = encoder_outputs.last_hidden_state[:, 0:1, :]
            last_hidden_state = last_hidden_state * shrink_rate
            encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:, 1:, :]), dim=1)

            decoder_input_ids = torch.tensor(np.full((batch_size, 1), self.model.config.decoder_start_token_id)).to('cuda:0')

            for _ in range(self.args.max_length):
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_state,
                    encoder_attention_mask=attention_mask
                )

                logits: torch.Tensor = self.lm_head(decoder_outputs[0])
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to('cuda:0')

                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to('cuda:0')

            if task == "gen":
                return decoder_input_ids
            return decoder_input_ids

        if task == "norm":
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
            norm = torch.norm(cls_embs, dim=1)
            return norm

        if task == "continuous_gen":
            trg_grade = batch["trg_grade"]
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            batch_size = input_ids.size()[0]

            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

            cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
            norm = torch.norm(cls_embs, dim=1)
            outputs = []
            for add_grade in [0, 0.5]:
                shrink_rate = ((trg_grade + add_grade) / norm).unsqueeze(1).unsqueeze(2)
                last_hidden_state = encoder_outputs.last_hidden_state[:, 0:1, :]
                last_hidden_state = last_hidden_state * shrink_rate
                encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:, 1:, :]), dim=1)

                decoder_input_ids = torch.tensor(np.full((batch_size, 1), self.model.config.decoder_start_token_id)).to('cuda:0')

                for _ in range(self.args.max_length):
                    decoder_outputs = self.model.decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_hidden_state,
                        encoder_attention_mask=attention_mask
                    )

                    logits: torch.Tensor = self.lm_head(decoder_outputs[0])
                    next_token_logits = logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to('cuda:0')

                    decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to('cuda:0')
                outputs.append(decoder_input_ids)

            return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def calculate_grade_loss(self, cls_emb, grade):
        norm = torch.norm(cls_emb, dim=1)
        grade = grade.float()

        criterion = nn.MSELoss()
        grade_loss = criterion(norm, grade)
        return grade_loss


class T5ForGradeTraining(pl.LightningModule):
    def __init__(self, args: Args, len_tokenizer: int):
        super().__init__()
        self.args = args
        model = T5Model.from_pretrained(args.model_name)
        model.resize_token_embeddings(len_tokenizer)
        self.model = model
        self.lm_head = nn.Linear(self.model.config.d_model, self.model.shared.num_embeddings, bias=False)
        self.lm_head.weight = self.model.decoder.get_input_embeddings().weight

    def training_step(self, batch):
        grade = batch["grade"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        decoder_input_ids = self.model._shift_right(input_ids)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)

        # generation loss
        logits: torch.Tensor = self.lm_head(outputs[0])
        loss_fn = nn.CrossEntropyLoss()
        gen_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), input_ids.view(-1))

        # grade loss
        cls_embs = outputs.encoder_last_hidden_state[:, 0, :]
        grade_loss = self.calculate_grade_loss(cls_embs, grade)

        loss = grade_loss + self.args.weight_gen_loss * gen_loss
        self.log_dict({"train/loss": loss, "train/grade_loss": grade_loss, "train/gen_loss": gen_loss})
        return loss

    def validation_step(self, batch):
        grade = batch["grade"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        decoder_input_ids = self.model._shift_right(input_ids)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)

        # generation loss
        logits: torch.Tensor = self.lm_head(outputs[0])
        loss_fn = nn.CrossEntropyLoss()
        gen_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), input_ids.view(-1))

        # grade loss
        cls_embs = outputs.encoder_last_hidden_state[:, 0, :]
        grade_loss = self.calculate_grade_loss(cls_embs, grade)

        loss = grade_loss + self.args.weight_gen_loss * gen_loss
        self.log_dict({"val/loss": loss, "val/grade_loss": grade_loss, "val/gen_loss": gen_loss})
        return loss

    def test_step(self, batch):
        return batch

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def calculate_grade_loss(self, cls_emb, grade):
        norm = torch.norm(cls_emb, dim=1)
        grade = grade.float()

        criterion = nn.MSELoss()
        grade_loss = criterion(norm, grade)
        return grade_loss


class T5ForTargetSimplification(pl.LightningModule):
    def __init__(self, args: Args, len_tokenizer: int):
        super().__init__()
        self.args = args
        model = T5Model.from_pretrained(args.model_name)
        model.resize_token_embeddings(len_tokenizer)
        self.model = model
        self.lm_head = nn.Linear(self.model.config.d_model, self.model.shared.num_embeddings, bias=False)
        self.lm_head.weight = self.model.decoder.get_input_embeddings().weight

    def training_step(self, batch):
        src_grade = batch["src_grade"]
        trg_grade = batch["trg_grade"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # grade loss
        cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
        grade_loss = self.calculate_grade_loss(cls_embs, src_grade)

        # shrink norm
        norm = torch.norm(cls_embs, dim=1)
        shrink_rate = (trg_grade / norm).unsqueeze(1).unsqueeze(2)
        last_hidden_state = encoder_outputs.last_hidden_state[:, 0:1, :]
        last_hidden_state = last_hidden_state * shrink_rate
        encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:, 1:, :]), dim=1)
        decoder_input_ids = batch['labels']
        labels = batch['labels']
        decoder_input_ids = self.model._shift_right(decoder_input_ids)

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=attention_mask
        )

        # generation loss
        logits: torch.Tensor = self.lm_head(decoder_outputs[0])
        loss_fn = nn.CrossEntropyLoss()
        gen_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        loss = grade_loss + self.args.weight_gen_loss * gen_loss
        self.log_dict({"train/loss": loss, "train/grade_loss": grade_loss, "train/gen_loss": gen_loss})
        return loss

    def validation_step(self, batch):
        src_grade = batch["src_grade"]
        trg_grade = batch["trg_grade"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # grade loss
        cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
        grade_loss = self.calculate_grade_loss(cls_embs, src_grade)

        # shrink norm
        norm = torch.norm(cls_embs, dim=1)
        shrink_rate = (trg_grade / norm).unsqueeze(1).unsqueeze(2)
        last_hidden_state = encoder_outputs.last_hidden_state[:, 0:1, :]
        last_hidden_state = last_hidden_state * shrink_rate
        encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:, 1:, :]), dim=1)
        decoder_input_ids = batch['labels']
        labels = batch['labels']
        decoder_input_ids = self.model._shift_right(decoder_input_ids)

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=attention_mask
        )

        # generation loss
        logits: torch.Tensor = self.lm_head(decoder_outputs[0])
        loss_fn = nn.CrossEntropyLoss()
        gen_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        loss = grade_loss + self.args.weight_gen_loss * gen_loss
        self.log_dict({"val/loss": loss, "val/grade_loss": grade_loss, "val/gen_loss": gen_loss})
        return loss

    def forward(self, batch, task="gen"):
        if task == "gen" or task == "trg_norm":
            trg_grade = batch["trg_grade"]
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            batch_size = input_ids.size()[0]

            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

            cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
            norm = torch.norm(cls_embs, dim=1)
            shrink_rate = (trg_grade / norm).unsqueeze(1).unsqueeze(2)
            last_hidden_state = encoder_outputs.last_hidden_state[:, 0:1, :]
            last_hidden_state = last_hidden_state * shrink_rate
            encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:, 1:, :]), dim=1)

            decoder_input_ids = torch.tensor(np.full((batch_size, 1), self.model.config.decoder_start_token_id)).to('cuda:0')

            for _ in range(self.args.max_length):
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_state,
                    encoder_attention_mask=attention_mask
                )

                logits: torch.Tensor = self.lm_head(decoder_outputs[0])
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to('cuda:0')

                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to('cuda:0')

            if task == "gen":
                return decoder_input_ids
            return decoder_input_ids

        if task == "grade_norm":
            # ここ直してない
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            half_batch_size = input_ids.size()[0] // 2

            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

            cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
            src_cls_embs, trg_cls_embs = torch.chunk(cls_embs.contiguous(), 2, dim=0)
            src_norm = torch.norm(src_cls_embs, dim=1)
            trg_norm = torch.norm(trg_cls_embs, dim=1)

            return input_ids, src_norm, trg_norm

        if task == "continuous_gen":
            trg_grade = batch["trg_grade"]
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            batch_size = input_ids.size()[0]

            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

            cls_embs: torch.Tensor = encoder_outputs.last_hidden_state[:, 0, :]
            norm = torch.norm(cls_embs, dim=1)
            shrink_rate = ((trg_grade + 0.5) / norm).unsqueeze(1).unsqueeze(2)
            last_hidden_state = encoder_outputs.last_hidden_state[:, 0:1, :]
            last_hidden_state = last_hidden_state * shrink_rate
            encoder_hidden_state = torch.cat((last_hidden_state, encoder_outputs.last_hidden_state[:, 1:, :]), dim=1)

            decoder_input_ids = torch.tensor(np.full((batch_size, 1), self.model.config.decoder_start_token_id)).to('cuda:0')

            for _ in range(self.args.max_length):
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_state,
                    encoder_attention_mask=attention_mask
                )

                logits: torch.Tensor = self.lm_head(decoder_outputs[0])
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to('cuda:0')

                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to('cuda:0')

            return decoder_input_ids

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def calculate_grade_loss(self, cls_emb, grade):
        norm = torch.norm(cls_emb, dim=1)
        grade = grade.float()

        criterion = nn.MSELoss()
        grade_loss = criterion(norm, grade)
        return grade_loss
