from datasets import load_dataset
from transformers import AutoImageProcessor, DefaultDataCollator
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import evaluate
import numpy as np
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer


data = load_dataset("TrieuNguyen/chest-xray")

# Labels function
labels = data["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    
# Preprocess
checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

# Apply it with data
data = data.with_transform(transforms)

# Create batch
data_collator = DefaultDataCollator()

# Evaluate
accuracy = evaluate.load("accuuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Train
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="chest_xray_pneumonia",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()
