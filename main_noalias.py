import os
import ast
import math
import shutil
import argparse
import logging
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed
import transformers
from transformers import SchedulerType, get_scheduler
from transformers.utils import send_example_telemetry
from transformers import get_cosine_schedule_with_warmup

from data_loader import DATASETS
from models import MODEL_FACTORY
from metric_noalias import score, reformat_eval_df
from augmentations import various_augs


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RSNA 2024 Lumbar Spine Degenerative Classification")
    # data
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--df_path", type=str, default=None)
    parser.add_argument("--eval_fold", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--dataset_process", type=str, default=None)
    parser.add_argument("--aug_prob", type=float, default=0.75)
    parser.add_argument("--aug_type", type=int, default=0)
    parser.add_argument("--flip_prob", type=float, default=0)
    # model
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--in_channels", type=int, default=30)
    parser.add_argument("--n_labels", type=int, default=25)
    parser.add_argument("--label_weights", type=str, default="[1, 2, 4]")
    # train
    parser.add_argument("--num_train_epochs", type=int, default=25)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--report_to", 
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log_step", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--best_metric", type=str, default="metric_loss")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(args)
    send_example_telemetry("run_no_trainer", args)
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    if args.fp16:
        accelerator = Accelerator(
            mixed_precision='fp16',
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            **accelerator_log_kwargs,
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            **accelerator_log_kwargs,
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Dataset
    all_df = pd.read_csv(args.df_path)
    all_df = all_df.fillna(-100)

    train_df = all_df[all_df["fold"] != args.eval_fold]
    eval_df = all_df[all_df["fold"] == args.eval_fold]
    train_df.reset_index(drop=True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    if args.max_train_samples is not None:
        train_df = train_df[:args.max_train_samples]
    if args.max_eval_samples is not None:
        eval_df = eval_df[:args.max_eval_samples]

    ## Reformat eval_df for calculating metric
    solution_df = reformat_eval_df(eval_df)
    submission_df = solution_df.copy(deep=True).drop(columns=["sample_weight"]) # placeholder

    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    train_df = train_df.replace(label2id)
    eval_df = eval_df.replace(label2id)

    dataset = DATASETS[args.dataset_process]
    transform_train, transform_eval = various_augs(
        args.aug_type, 
        aug_prob=args.aug_prob, 
        img_size=args.img_size
    )
    train_ds = dataset(
        train_df, 
        args.img_dir, 
        transform=transform_train,
        image_size=args.img_size, 
        in_channels=args.in_channels,
        phase="train",
        flip_prob=args.flip_prob
    )
    eval_ds = dataset(
        eval_df, 
        args.img_dir, 
        transform=transform_eval,
        image_size=args.img_size, 
        in_channels=args.in_channels,
        phase="test"
    )
    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers
    )
    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.workers
    )

    # Model
    model = MODEL_FACTORY[args.model_name](
        args.model_name, 
        in_c=args.in_channels, 
        n_classes=args.n_labels * 3,
        pretrained=True)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optimizer and LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    num_warmup_steps = int(args.warmup_ratio * args.max_train_steps)
    """
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )
    """
    if args.lr_scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps
                if overrode_max_train_steps
                else args.max_train_steps * accelerator.num_processes,
            num_cycles = 0.475,
        )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("image_classification_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_df)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Numb trainable parameters = {params}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    weights = torch.tensor(ast.literal_eval(args.label_weights), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.cuda())
    best_eval_metric_loss = 100
    best_eval_metric_epoch = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                img, target = batch[0], batch[1] # TODO: check
                output = model(img)
                """
                loss = 0
                for col in range(args.n_labels):
                    pred = y[:,col*3:col*3+3]
                    gt = label[:,col]
                    loss = loss + criterion(pred, gt) / args.n_labels
                """
                output = output.view(-1, args.n_labels, 3).reshape(-1, 3)
                target = target.view(-1)
                loss = criterion(output, target)

                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if args.log_step is not None:
                if completed_steps % args.log_step == 0:
                    logger.info(f" epoch {epoch}, step {completed_steps} || loss: {loss.item():.4f}, lr: {lr_scheduler.get_lr()[0]:.8f}")

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # evaluation
        model.eval()
        losses = []
        preds = []
        refs = []
        outputs = []
        for step, batch in enumerate(eval_dataloader):
            img, target = batch[0], batch[1]
            with torch.no_grad():
                output = model(img)
            """
            loss = 0
            for col in range(args.n_labels):
                pred = y[:,col*3:col*3+3]
                gt = label[:,col]
                loss = loss + criterion(pred, gt) / args.n_labels
            """
            output = output.view(-1, args.n_labels, 3).reshape(-1, 3)
            target = target.view(-1)
            loss = criterion(output, target)
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

            preds.append(output.argmax(dim=-1))
            refs.append(target)
            outputs.append(output.cpu())

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)

        preds = torch.cat(preds)
        refs = torch.cat(refs)
        mask = refs != -100
        preds = preds[mask]
        refs = refs[mask]
        correct = torch.eq(preds, refs).sum().item() 
        total = refs.size(0)
        eval_acc = correct / total

        outputs = torch.cat(outputs)
        assert len(outputs) == len(submission_df)
        normed_outputs = torch.nn.functional.softmax(outputs, dim=-1)

        submission_df_metric = submission_df.copy(deep=True)
        solution_df_metric = solution_df.copy(deep=True)
        submission_df_metric.iloc[:, 1:] = normed_outputs.numpy()

        eval_metric_loss = score(
                solution_df_metric[solution_df_metric["sample_weight"]!=0].copy(deep=True),
                submission_df_metric[solution_df_metric["sample_weight"]!=0].copy(deep=True),
                "row_id", 
                1.
            )
        
        if args.best_metric == "metric_loss":
            current_loss = eval_metric_loss
        elif args.best_metric == "eval_loss":
            current_loss = eval_loss

        if best_eval_metric_loss > current_loss:
            best_eval_metric_loss = current_loss
            best_eval_metric_epoch = epoch

            output_dir = f"fold{args.eval_fold}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            logger.info(f"Save model to {output_dir}")
            accelerator.save_state(output_dir)

        logger.info(
            f" epoch {epoch} || metric_loss: {eval_metric_loss:.4f}, eval_loss: {eval_loss:.4f}, "
            f"eval_acc: {eval_acc:.4f}, best_loss: {best_eval_metric_loss:.4f}, best_epoch: {best_eval_metric_epoch}"
        )

        if args.with_tracking:
            accelerator.log(
                {
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

    if args.with_tracking:
        accelerator.end_training()

    src_file = f"{output_dir}/model.safetensors"
    tgt_file = f"{args.output_dir}/fold{args.eval_fold}_model.safetensors"
    shutil.copyfile(src_file, tgt_file)

if __name__ == "__main__":
    main()