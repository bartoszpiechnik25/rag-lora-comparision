from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer
from dataset import prepare_dataset_for_training
from lora.config import get_lora
import uuid, os, json

def train():
    out_dir = f"./lora/model/lora-{uuid.uuid4()}"

    train, val = prepare_dataset_for_training()
    model = get_lora()

    training_args = Seq2SeqTrainingArguments(
        learning_rate=3e-3,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        output_dir="./lora/train",
        optim="adamw_torch",
        logging_steps=10,
        logging_strategy="steps",
        logging_first_step=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_strategy="epoch",
        bf16=True
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
    )
    print("Training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("siema")
        pass

    print("Saving model...")
    trainer.save_model(out_dir)
    trainer.save_state()

    print("Saving training metrics...")

    # Save metrics to a JSON file in output_dir
    metrics = trainer.state.log_history  # This contains all logged metrics
    metrics_path = os.path.join(training_args.output_dir, "all_metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == '__main__':
    train()