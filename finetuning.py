import matplotlib.pyplot as plt
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import OnlineContrastiveLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from transformers import TrainerCallback

from firebase_dataset import FirebaseDataset

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="all-mpnet-base-v2 fine-tuned on contrastive pairs",
    )
)

# 2. Load a dataset to finetune on
data = FirebaseDataset("qa.json", train_ratio=0.7, test_ratio=0.2, eval_ratio=0.1)
print(data.size)

train_data = data["train"]
eval_data = data["eval"]
test_data = data["test"]

train_dataset = Dataset.from_dict({
    "anchor": train_data.anchors, 
    "positive/negative": train_data.pos_neg,
    "label": train_data.labels,
})

test_dataset = Dataset.from_dict({
    "anchor": test_data.anchors, 
    "positive/negative": test_data.pos_neg,
    "label": test_data.labels,
})

eval_dataset = Dataset.from_dict({
    "anchor": eval_data.anchors, 
    "positive/negative": eval_data.pos_neg,
    "label": eval_data.labels,
})

# 3. Define a loss function
loss = OnlineContrastiveLoss(model)

# 4. (Optional) Specify training arguments:
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/all-mpnet-base-v2-qa",
    # Optional training parameters:
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # Not sure exactly what is best to use with our loss function
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    run_name="all-mpnet-base-v2-qa",  # Will be used in W&B if `wandb` is installed
)

# 5. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = BinaryClassificationEvaluator(
    sentences1=list(eval_dataset["anchor"]),
    sentences2=list(eval_dataset["positive/negative"]), 
    labels=list(eval_dataset["label"]),
)
before_trainng_results = dev_evaluator(model)

# Custom logging callback
train_losses = []
eval_losses = []
epochs = []
class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            epoch = logs.get('epoch')
            if epoch not in epochs:
                epochs.append(epoch)
            # print(f"Step: {state.global_step}, Logs: {logs}")  # Print all logs
            loss = logs.get('loss')
            if loss is not None:
                train_losses.append(loss)
            
            eval_loss = logs.get('eval_loss')
            if eval_loss is not None:
                eval_losses.append(eval_loss)
            # print(f"Training Loss: {loss}, Evaluation Loss: {eval_loss}")

# 6. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
    callbacks=[CustomLoggingCallback],
)
trainer.train()

# Initialize the evaluator
test_evaluator = BinaryClassificationEvaluator(
    sentences1=list(test_dataset["anchor"]),
    sentences2=list(test_dataset["positive/negative"]),
    labels=list(test_dataset["label"]),
)
test_results = test_evaluator(model)
after_training_results = dev_evaluator(model)

print(f"Before training: {before_trainng_results} \n\n After training: {after_training_results} \n\n Test results: {test_results}")

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Training Loss', marker='o')
plt.plot(epochs, eval_losses, label='Evaluation Loss', marker='o')

plt.title('Training and Evaluation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)  # Set x-ticks to be the epochs
plt.legend()
plt.grid()

# Show the plot
plt.show()

# 7. Save the trained model
model.save_pretrained("models/all-mpnet-base-v2-qa/final")