from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer
from dataset import prepare_dataset_for_training
from lora.config import get_lora
import uuid

def train():
    out_dir = f"./lora/model/lora-{uuid.uuid4()}"

    train, val = prepare_dataset_for_training()
    model = get_lora()

    training_args = Seq2SeqTrainingArguments(
        learning_rate=3e-3,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        auto_find_batch_size=True,
        gradient_accumulation_steps=8,
        output_dir="./lora/train",
        optim="adamw_torch",
        logging_steps=40,
        logging_strategy="steps",
        logging_first_step=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        evaluation_strategy="epoch",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
    )
    print("Training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(out_dir)
    trainer.save_state()


if __name__ == '__main__':
    train()