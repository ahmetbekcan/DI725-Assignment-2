import json
from datasets import load_dataset
from transformers import AutoImageProcessor
from PIL import Image
import torch
import numpy as np
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from functools import partial
from transformers.image_transforms import center_to_corners_format
from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
import wandb
import os
from transformers import Trainer

def process_images(examples):
    images = []
    annotations = []

    for image_id, anns in zip(examples["image_id"], examples["annotations"]):
        image_path = id_to_path[image_id]
        image = Image.open(image_path).convert("RGB")
        images.append(image)

        annotations.append({
            "image_id": image_id,
            "annotations": anns
        })
    
    processed = image_processor(images=images, annotations=annotations, return_tensors="pt")
    processed.pop("pixel_mask", None)
    return processed


def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]

    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels

    return batch

#-------------------------------------------------------------------------------------------------------------
with open("data/coco_compatible_annotations.json", 'r') as f:
    data = json.load(f)

id2label = {item['id']: item['name'] for item in data['categories']}
label2id = {v: k for k, v in id2label.items()}

dataset = load_dataset('json', data_files='data/coco_compatible_annotations.json', field="annotations")

train_test_split = dataset['train'].train_test_split(test_size=0.4, shuffle=True, seed=2299436)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

val_test_split = test_dataset.train_test_split(test_size=0.25)
test_dataset = val_test_split['train']
val_dataset = val_test_split['test']

MAX_SIZE = 500
checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(
    checkpoint,do_resize=True,
    size={"max_height": MAX_SIZE, "max_width": MAX_SIZE},
    do_pad=True,
    pad_size={"height": MAX_SIZE, "width": MAX_SIZE},)

id_to_path = {entry['id']: "data/images/"+entry['file_name'] for entry in data["images"]}

train_dataset_processed = train_dataset.with_transform(process_images)
val_dataset_processed = val_dataset.with_transform(process_images)

eval_compute_metrics_fn = partial(
    compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
)

model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)


output_dir = "output/detr-finetuned-auair"
training_args = TrainingArguments(
    output_dir="output_dir",
    num_train_epochs=1,
    fp16=True,
    per_device_train_batch_size=32,
    dataloader_num_workers=0,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    max_grad_norm=0.01,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    eval_steps=0.1,
    save_steps=0.5,
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    push_to_hub=True,
    push_to_hub_model_id="detr-resnet-50-finetuned-auair",
    run_name="detr-resnet-50-auair-finetuned"
)


wandb.login(key=os.getenv("WANDB_API"))

wandb.init(
    project="Assignment-2",
    name="detr-resnet-50-auair-finetuned",
    config=training_args
)

import pkg_resources
installed_packages = [pkg.key for pkg in pkg_resources.working_set]
assert('pycocotools' in installed_packages)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_processed,
    eval_dataset=val_dataset_processed,
    processing_class=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn
)

trainer.train()