import os
import argparse
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from model.blip2_stage1 import Blip2Stage1
from data_provider.stage1_dm import Stage1DM
from model.dist_funs import MyDeepSpeedStrategy
from pathlib import Path
from model.convert import ConvertAfterSaveCallback


os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision('medium')


def main(args):
    pl.seed_everything(args.seed)

    if args.init_checkpoint:
        print(f"loading model from {args.init_checkpoint}")
        model = Blip2Stage1.load_from_checkpoint(args.init_checkpoint, device=args.devices, strict=False)
    else:
        model = Blip2Stage1(args)
    
    print('total params:', sum(p.numel() for p in model.parameters()))

    dm = Stage1DM(args.num_workers, args.batch_size, args.root, args)
    dm.init_tokenizer(model.blip2qformer.tokenizer, model.blip2qformer.plm_tokenizer)
    model.val_match_loader, model.test_match_loader = dm.match_dataloader()

    callbacks = []
    ckpt_dirpath = "all_checkpoints/" + args.filename + "/"
    callbacks.append(plc.ModelCheckpoint(dirpath=ckpt_dirpath,
                                         filename='{epoch:02d}',
                                         every_n_epochs=args.save_every_n_epochs,
                                         save_top_k=1,
                                         save_on_train_epoch_end=True))
    if len(args.devices.split(',')) > 1 and args.strategy == 'deepspeed':
        callbacks.append(ConvertAfterSaveCallback(ckpt_dirpath, args.save_every_n_epochs, run_after_train_epoch=True))

    if len(args.devices.split(',')) > 1:
        if args.strategy == 'ddp':
            find_unused_parameters = (not args.ptm) or (not args.lm)
            strategy = strategies.DDPStrategy(start_method='spawn', find_unused_parameters=find_unused_parameters)
        elif args.strategy == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            raise NotImplementedError()
    else:
        strategy = None
        args.devices = eval(args.devices)
        print(args.devices)
    if args.use_wandb_logger:
        Path(f'./all_checkpoints/{args.filename}/wandb').mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(project=args.filename, save_dir=f'./all_checkpoints/{args.filename}/')
    else:
        logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    trainer = Trainer(accelerator=args.accelerator,
                     devices=args.devices if isinstance(args.devices, int) else len(args.devices.split(',')),
                     precision=args.precision,
                     max_epochs=args.max_epochs,
                     check_val_every_n_epoch=args.check_val_every_n_epoch, 
                     callbacks=callbacks, 
                     strategy=strategy, 
                     logger=logger,
                     )
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = 49
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage1_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy', type=str, default='deepspeed')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--use_wandb_logger', action='store_true', default=False)
    parser = Blip2Stage1.add_model_specific_args(parser)
    parser = Stage1DM.add_model_specific_args(parser)

    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

