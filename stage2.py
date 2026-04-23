"""
Stage2 training and reliability flow.

When neither flag is set: train on full training set, validate/test on valid_set and test_set.

BLEU/ROUGE/Meteor and (if --report_go_wang_on_test) GO Wang/IA/Jaccard are computed on the *test* set
when caption_eval_epoch triggers, not on the validation set.

1) --inference_on_training_data: load checkpoint, run inference on 2000 train samples (GO>=2),
   extract GO terms, compute Wang similarity, write a new file in train_set format (r = wang score)
   to all_checkpoints/<filename>/reliability_finetune.json for the next step.

2) --train_reliability_head_only: load checkpoint, freeze all except reliability head,
   train reliability head on the file from step 1 (required: --reliability_finetune_data <path>).

3) --report_go_wang_on_val (default off): on the *last* epoch only, run GO extraction on validation
   set predictions (process_texts_with_api), then compute Wang similarity vs valid_set.json and log.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from data_provider.stage2_dm import Stage2DM
from model.blip2_stage2 import Blip2Stage2
from model.dist_funs import MyDeepSpeedStrategy
from pathlib import Path
from model.convert import ConvertAfterSaveCallback

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision('medium')


def main(args):
    pl.seed_everything(args.seed)
    dm = Stage2DM(args.root, args)
    if args.init_checkpoint:
        # Load with shape-mismatch tolerance: skip keys (e.g. reliability_head) whose shapes differ from current model
        model = Blip2Stage2(args)
        ckpt = torch.load(args.init_checkpoint, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_sd and model_sd[k].shape == v.shape}
        skipped = [k for k in state_dict if k not in filtered]
        if skipped:
            print(f"Skipped {len(skipped)} keys (shape mismatch or unexpected): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
        model.load_state_dict(filtered, strict=False)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage1_path:
        model = Blip2Stage2(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2Stage2(args)

    dm.init_tokenizer(model.blip2.llm_tokenizer, model.blip2.plm_tokenizer)
    print('total params:', sum(p.numel() for p in model.parameters()))

    if args.inference_on_training_data:
        if not args.init_checkpoint:
            raise ValueError("--init_checkpoint is required for --inference_on_training_data")
        os.makedirs(f"all_checkpoints/{args.filename}", exist_ok=True)
        out_train = f"all_checkpoints/{args.filename}/reliability_finetune.json"
        out_valid = f"all_checkpoints/{args.filename}/reliability_finetune_valid.json"
        print("Starting reliability inference (train subset + validation)...", flush=True)
        model.run_inference_on_training_subset(
            dm,
            out_train,
            min_go=getattr(args, 'inference_min_go', 2),
            sample_size=getattr(args, 'inference_sample_size', 0),
            seed=args.seed,
            reliability_label_zero=getattr(args, 'reliability_label_zero', False),
        )
        model.run_inference_on_validation_set(dm, out_valid, reliability_label_zero=getattr(args, 'reliability_label_zero', False))
        print(f"Inference done. Train subset: {out_train}; Validation: {out_valid}")
        print("Next: --train_reliability_head_only --reliability_finetune_data " + out_train + " --reliability_finetune_valid_data " + out_valid)
        return

    if args.train_reliability_head_only:
        if not args.init_checkpoint:
            raise ValueError("--init_checkpoint is required for --train_reliability_head_only")
        if not getattr(args, 'reliability_finetune_data', None) or not args.reliability_finetune_data:
            raise ValueError("--reliability_finetune_data is required for --train_reliability_head_only (run --inference_on_training_data first)")
        model.freeze_for_reliability_finetune()

    callbacks = []
    ckpt_dirpath = "all_checkpoints/" + args.filename + "/"
    if args.train_reliability_head_only:
        # Reliability head training: save best by val class-1 (r=1.0) F1 (max)
        callbacks.append(plc.ModelCheckpoint(dirpath=ckpt_dirpath,
                                             filename='best_val_reliability_class1_f1',
                                             monitor='val/reliability_class1_f1',
                                             mode='max',
                                             save_top_k=1,
                                             save_on_train_epoch_end=False))
    else:
        # Normal Stage2: save best by val/bleu2 (max)
        callbacks.append(plc.ModelCheckpoint(dirpath=ckpt_dirpath,
                                             filename='best_val_bleu2',
                                             monitor='val/bleu2',
                                             mode='max',
                                             save_top_k=1,
                                             save_on_train_epoch_end=False))
    if len(args.devices.split(',')) > 1 and args.strategy == 'deepspeed':
        callbacks.append(ConvertAfterSaveCallback(ckpt_dirpath, args.save_every_n_epochs, run_after_train_epoch=False))

    if len(args.devices.split(',')) > 1:
        if args.strategy == 'ddp':
            find_unused_parameters = (not args.ptm) or (not args.lm)
            strategy = strategies.DDPStrategy(start_method='spawn', find_unused_parameters=find_unused_parameters)
        elif args.strategy == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            NotImplementedError()
    else:
        strategy = 'ddp'
        args.devices = 1
    if args.use_wandb_logger:
        Path(f'./all_checkpoints/{args.filename}/wandb').mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(project=args.filename, save_dir=f'./all_checkpoints/{args.filename}/')
    else:
        logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices if isinstance(args.devices, int) else len(args.devices.split(',')),
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
    )
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'eval':
        if not args.init_checkpoint:
            raise ValueError("--init_checkpoint is required for eval mode")
        print(f"Running evaluation mode with checkpoint: {args.init_checkpoint}")
        print(f"Caption eval epoch trigger: {args.caption_eval_epoch}")
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError(f"Unknown mode: {args.mode}. Use 'train' or 'eval'.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy', type=str, default='deepspeed')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--use_wandb_logger', action='store_true', default=False)
    parser.add_argument('--inference_on_training_data', action='store_true', default=False, help='Run inference on train samples (GO>=min_go), extract GO, compute Wang, write train_set-format file for reliability head training')
    parser.add_argument('--inference_min_go', type=int, default=2, help='Min GO terms for inference training subset (GO>1 means min_go=2). Default 2.')
    parser.add_argument('--inference_sample_size', type=int, default=0, help='Max samples for inference training subset. 0 or negative = all samples with GO>=min_go. Default 0.')
    parser.add_argument('--reliability_label_zero', action='store_true', default=False, help='When used with --inference_on_training_data: skip GO extraction, write r=0 for all rows (faster, no API calls)')
    parser.add_argument('--train_reliability_head_only', action='store_true', default=False, help='Freeze all except reliability head; train on file from --inference_on_training_data step')
    parser.add_argument('--reliability_finetune_data', type=str, default='', help='Required for --train_reliability_head_only: path to reliability_finetune.json from step 3')
    parser.add_argument('--reliability_finetune_valid_data', type=str, default='', help='Optional for --train_reliability_head_only: path to reliability_finetune_valid.json from step 3')
    parser = Blip2Stage2.add_model_specific_args(parser)
    parser = Stage2DM.add_model_specific_args(parser)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

if __name__ == '__main__':
    main(get_args())

