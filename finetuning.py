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

# 3. Load a dataset to finetune on
data = FirebaseDataset("qa.json", train_ratio=0.7, test_ratio=0.2, eval_ratio=0.1)

train_data = data["train"]
eval_data = data["eval"]
test_data = data["test"]

train_dataset = Dataset.from_dict({
    "anchor": list(map(str, train_data.anchors)), 
    "positive/negative": list(map(str, train_data.pos_neg)),
    "label": list(map(int, train_data.labels)),
})

test_dataset = Dataset.from_dict({
    "anchor": list(map(str, test_data.anchors)), 
    "positive/negative": list(map(str, test_data.pos_neg)),
    "label": list(map(int, test_data.labels)),
})

eval_dataset = Dataset.from_dict({
    "anchor": list(map(str, eval_data.anchors)), 
    "positive/negative": list(map(str, eval_data.pos_neg)),
    "label": list(map(int, eval_data.labels)),
})

if len(train_dataset) == 0 or len(eval_dataset) == 0 or len(test_dataset) == 0:
    raise ValueError("One of the datasets is empty. Check your data loading and splitting logic.")

# 4. Define a loss function
loss = OnlineContrastiveLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/all-mpnet-base-v2-qa",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="all-mpnet-base-v2-qa",  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = BinaryClassificationEvaluator(
    sentences1=list(eval_dataset["anchor"]),
    sentences2=list(eval_dataset["positive/negative"]), 
    labels=list(eval_dataset["label"]),
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# Initialize the evaluator
test_evaluator = BinaryClassificationEvaluator(
    sentences1=list(test_dataset["anchor"]),
    sentences2=list(test_dataset["positive/negative"]), 
    labels=list(test_dataset["label"]),
)
test_evaluator(model)

# 8. Save the trained model
model.save_pretrained("models/all-mpnet-base-v2-qa/final")

# # # 9. (Optional) Push it to the Hugging Face Hub
# # model.push_to_hub("mpnet-base-all-nli-triplet")