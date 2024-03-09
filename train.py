import argparse
import random
import torch
from torch import nn, optim
import numpy as np
import editdistance
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed
import torch.distributed as dist
import time
import glob
import wandb
import editdistance
from datetime import datetime
import sacrebleu, json
import soundfile as sf
import shutil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import contextlib
import math

# model
from src.models.TMT_ENDE_BERT import TMT
from src.lr_scheduler import LinearWarmup, CosineAnnealingLRWarmup
# data
from src.data.dataset_train_TMT import AVTDataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--coco_path', default="path_to/COCO_2014")
    parser.add_argument('--flickr_path', default="path_to/Flickr8k/Images")
    parser.add_argument('--spcoco_path', default="path_to/SpokenCOCO/Hubert_units")
    parser.add_argument('--spflickr_path', default="path_to/flickr_audio/Hubert_units")
    parser.add_argument('--spcoco_split_path', default="path_to/SpokenCOCO")
    parser.add_argument('--karpathy_split_path', default="path_to/Karpathy_split")

    parser.add_argument("--max_sp_len", type=int, default=384)
    parser.add_argument("--max_txt_len", type=int, default=128)
    
    # Im token size
    parser.add_argument("--im_size", type=int, default=16)

    # Tasks
    parser.add_argument("--num_task", type=int, default=6)

    parser.add_argument("--im_sp", default=False, action='store_true')
    parser.add_argument("--im_txt", default=False, action='store_true')

    parser.add_argument("--txt_sp", default=False, action='store_true')
    parser.add_argument("--txt_im", default=False, action='store_true')

    parser.add_argument("--sp_txt", default=False, action='store_true')
    parser.add_argument("--sp_im", default=False, action='store_true')

    parser.add_argument("--pad", type=int, default=0, help='will be set by dataset')
    parser.add_argument("--bos", type=int, default=0, help='will be set by dataset')
    parser.add_argument("--eos", type=int, default=0, help='will be set by dataset')

    parser.add_argument("--txt", type=int, default=0)
    parser.add_argument("--sp", type=int, default=1)
    parser.add_argument("--im", type=int, default=2)
    
    parser.add_argument("--train_data", type=str, default='coco')

    parser.add_argument("--temp_dir", type=str, default='')
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/TMT')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--update_frequency", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--tot_iters", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--warmup", default=False, action='store_true')
    parser.add_argument("--warmup_iteration", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.000001)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_step", type=int, default=5000)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--mode", type=str, default='train', help='train, test, valid')
    parser.add_argument("--reset_optimizer", default=False, action='store_true')

    parser.add_argument("--fp16", default=False, action='store_true')

    parser.add_argument("--architecture", default='bert')
    parser.add_argument("--image_tokenizer", default='VQGAN', help='VQGAN, SEED')

    parser.add_argument("--generate_im", default=False, action='store_true')
    parser.add_argument("--generation_step", type=int, default=1000)
    parser.add_argument("--num_gen_im", type=int, default=3)
    parser.add_argument("--gen_max_len", type=int, default=256, help='sp/txt generation length if test mode')

    parser.add_argument("--distributed", default=False, action='store_true')
    parser.add_argument("--masterport", type=str, default='1234')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    return args


def train_net(args):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.local_rank)
    torch.cuda.manual_seed_all(args.local_rank)
    random.seed(args.local_rank)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_PORT'] = args.masterport

    args.temp_dir = './tmp_eval/' + args.checkpoint_dir.split('/')[-1]

    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    train_data = AVTDataset(
        coco_path=args.coco_path, 
        flickr_path=args.flickr_path, 
        spcoco_path=args.spcoco_path, 
        spflickr_path=args.spflickr_path, 
        spcoco_split_path=args.spcoco_split_path, 
        karpathy_split_path=args.karpathy_split_path, 
        mode='train', 
        num_im_unit=8192, 
        max_sp_len=args.max_sp_len,
        max_txt_len=args.max_txt_len,
        tokenizer='bert-base-uncased',
        train_data=args.train_data,
        image_tokenizer=args.image_tokenizer,
        architecture=args.architecture,
    )

    args.bos, args.eos, args.pad = train_data.bos, train_data.eos, train_data.pad
    
    if args.architecture == 'bert':
        model = TMT(args, train_data.num_sp_unit, train_data.num_txt, train_data.num_im_unit)
    else:
        raise NotImplementedError
    num_model = sum(p.numel() for p in model.parameters())

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layernorm", "LayerNorm", "embeddings.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    
    num_train = []
    for param in params:
        for p in param['params']:
            num_train.append(p.numel())

    num_train = sum(num_train)
    if args.local_rank == 0:
        print(f'Train # of params: {num_train} / {num_model}')

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    if args.warmup:
        if args.tot_iters is not None:
            scheduler = CosineAnnealingLRWarmup(optimizer, T_max=args.tot_iters, T_warmup=args.warmup_iteration)
        else:
            scheduler = LinearWarmup(optimizer, T_warmup=args.warmup_iteration)
    else:
        scheduler = None

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    if args.local_rank == 0:
        num_in_optimizer = []
        for param in optimizer.param_groups:
            for p in param['params']:
                num_in_optimizer.append(p.numel())
        print(f"Params in optimizer: {sum(num_in_optimizer)}")

    if args.distributed:
        model = DDP(model, 
                    device_ids=[args.local_rank], 
                    output_device=args.local_rank, 
                    find_unused_parameters=False \
                    if args.im_txt and args.im_sp and args.sp_txt and args.sp_im and args.txt_im and args.txt_sp \
                    else True,
                    )

    if args.image_tokenizer == 'SEED':
        from SEED.models.seed_llama_tokenizer import ImageTokenizer
        vq_model = ImageTokenizer(model_path='./pretrained/seed_quantizer.pt', 
                                  diffusion_model_path='stabilityai/stable-diffusion-2-1-unclip', 
                                  fp16=False, 
                                  load_diffusion=True if args.generate_im else False)
        if args.generate_im:
            vq_model.diffusion_model.enable_xformers_memory_efficient_attention()
    else:
        assert NotImplementedError
    vq_model.requires_grad_(False)
    vq_model.cuda()

    # _ = validate(model, vq_model, fast_validate=True)
    train(model, vq_model, train_data, args.epochs, optimizer=optimizer, scheduler=scheduler, args=args, scaler=scaler)

def train(model, vq_model, train_data, epochs, optimizer, scheduler, args, scaler):
    best_val_bleu = 0.0
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")
    if args.local_rank == 0:
        writer = SummaryWriter(comment=os.path.split(args.checkpoint_dir)[-1])
        if args.project is not None:
            wandbrun = wandb.init(project="TMT", name=args.project + f'_{dt_string}')
            wandbrun.config.epochs = args.epochs
            wandbrun.config.batch_size = args.batch_size
            wandbrun.config.learning_rate = args.lr
            wandbrun.config.architecture = args.architecture
            wandbrun.config.eval_step = args.eval_step
            wandbrun.config.update_frequency = args.update_frequency
            wandbrun.config.warmup = args.warmup
            wandbrun.config.warmup_iteration = args.warmup_iteration
            wandbrun.config.im_txt = args.im_txt
            wandbrun.config.im_sp = args.im_sp
            wandbrun.config.sp_txt = args.sp_txt
            wandbrun.config.sp_im = args.sp_im
            wandbrun.config.txt_sp = args.txt_sp
            wandbrun.config.txt_im = args.txt_im
            wandbrun.config.pad = args.pad
            wandbrun.config.bos = args.bos
            wandbrun.config.eos = args.eos
            wandbrun.config.txt = args.txt
            wandbrun.config.sp = args.sp
            wandbrun.config.im = args.im
            wandbrun.config.fp16 = args.fp16
            wandbrun.config.tot_iters = args.tot_iters
            wandbrun.config.image_tokenizer = args.image_tokenizer
            wandbrun.config.generate_im = args.generate_im
        else:
            wandbrun = None
    else:
        writer = None
        wandbrun = None

    model.train()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    dataloader = DataLoader(
        train_data,
        shuffle=False if args.distributed else True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=lambda x: train_data.collate_fn(x),
    )

    samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    step = args.start_step

    optimizer.zero_grad()
    for epoch in range(args.start_epoch, epochs):
        loss_list = []
        it_uer_list, is_uer_list, ts_uer_list, st_uer_list = [], [], [], []
        it_acc_list, is_acc_list, ts_acc_list, ti_acc_list, st_acc_list, si_acc_list = [], [], [], [], [], []
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print(f"Epoch [{epoch}/{epochs}]")
            prev_time = time.time()
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 100 == 0:
                iter_time = (time.time() - prev_time) / 100
                prev_time = time.time()
                print("******** Training [%d / %d] : %d / %d, Iter Time : %.3f sec, Learning Rate of %f ********" % (
                    epoch, epochs, (i + 1) * batch_size, samples, iter_time, optimizer.param_groups[0]['lr']))
            image_input, sp_unit, txt_unit, sp_unit_len, txt_unit_len, _ = batch

            gt_im_unit = im_unit_preparation(args, image_input.cuda(), vq_model)

            im_unit = train_data.add_special_room(gt_im_unit.cpu())
            im_unit = train_data.append_eos(im_unit)
            im_unit = train_data.append_bos(im_unit)

            # im_unit, sp_unit, txt_unit, sp_unit_len, txt_unit_len
            with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                output_is, output_it, output_st, output_si, output_ts, output_ti, is_loss, it_loss, st_loss, si_loss, ts_loss, ti_loss \
                    = model(im_unit.cuda(), sp_unit.cuda(), txt_unit.cuda(), sp_unit_len, txt_unit_len)
            
            loss = (is_loss + it_loss + st_loss + si_loss + ts_loss + ti_loss) / (args.num_task * args.update_frequency)

            if args.fp16:
                scaler.scale(loss).backward()
                loss = loss.float()
            else:
                loss.backward()

            if ((i + 1) % args.update_frequency == 0) or (i + 1 == len(dataloader)):
                step += 1
                if args.fp16:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    if not torch.isfinite(grad_norm):
                        print(f"The grad norm is {grad_norm}. Skipping updating the model.")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    if not torch.isfinite(grad_norm):
                        print(f"The grad norm is {grad_norm}. Skipping updating the model.")
                    else:
                        optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            else:
                continue

            loss = loss.cpu().item()

            ## image -> text
            if args.im_txt: 
                output_it.logits = output_it.logits.float()
                it_loss = it_loss.float().cpu().item()

                it_results, it_gt_txt, it_acc_list, it_uer_list = process_text(args, output_it, train_data, txt_unit, txt_unit_len, it_acc_list, it_uer_list)
                save_log(args, step=step, results=it_results, gt=it_gt_txt, acc_list=it_acc_list, uer_list=it_uer_list, loss=it_loss, writer=writer, wandbrun=wandbrun, text='Image -> Text', task='it')
                                        
            ## image -> speech
            if args.im_sp: 
                output_is.logits = output_is.logits.float()
                is_loss = is_loss.float().cpu().item()

                is_results, is_gt_sp, is_acc_list, is_uer_list = process_speech(args, output_is, train_data, sp_unit, sp_unit_len, is_acc_list, is_uer_list)
                save_log(args, step=step, results=is_results, gt=is_gt_sp, acc_list=is_acc_list, uer_list=is_uer_list, loss=is_loss, writer=writer, wandbrun=wandbrun, text='Image -> Speech', task='is')

            ## speech -> text
            if args.sp_txt: 
                output_st.logits = output_st.logits.float()
                st_loss = st_loss.float().cpu().item()

                st_results, st_gt_txt, st_acc_list, st_uer_list = process_text(args, output_st, train_data, txt_unit, txt_unit_len, st_acc_list, st_uer_list)
                save_log(args, step=step, results=st_results, gt=st_gt_txt, acc_list=st_acc_list, uer_list=st_uer_list, loss=st_loss, writer=writer, wandbrun=wandbrun, text='Speech -> Text', task='st')

            ## speech -> image
            if args.sp_im: 
                output_si.logits = output_si.logits.float()
                si_loss = si_loss.float().cpu().item()

                si_results, si_gt_im, si_pred_im = process_image(output_si, train_data, vq_model, gt_im_unit, si_acc_list, args, generate=True if (step % args.generation_step == 0 and args.generate_im) else False)
                save_log(args, step=step, results=si_results, gt=si_gt_im, acc_list=si_acc_list, uer_list=None, loss=si_loss, writer=writer, wandbrun=wandbrun, text='Speech -> Image', task='si', pred_im=si_pred_im)

                if args.distributed:
                    dist.barrier()

            ## text -> speech
            if args.txt_sp: 
                output_ts.logits = output_ts.logits.float()
                ts_loss = ts_loss.float().cpu().item()

                ts_results, ts_gt_sp, ts_acc_list, ts_uer_list = process_speech(args, output_ts, train_data, sp_unit, sp_unit_len, ts_acc_list, ts_uer_list)
                save_log(args, step=step, results=ts_results, gt=ts_gt_sp, acc_list=ts_acc_list, uer_list=ts_uer_list, loss=ts_loss, writer=writer, wandbrun=wandbrun, text='Text -> Speech', task='ts')

            ## text -> image
            if args.txt_im: 
                output_ti.logits = output_ti.logits.float()
                ti_loss = ti_loss.float().cpu().item()

                ti_results, ti_gt_im, ti_pred_im = process_image(output_ti, train_data, vq_model, gt_im_unit, ti_acc_list, args, generate=True if (step % args.generation_step == 0 and args.generate_im) else False)
                save_log(args, step=step, results=ti_results, gt=ti_gt_im, acc_list=ti_acc_list, uer_list=None, loss=ti_loss, writer=writer, wandbrun=wandbrun, text='Text -> Image', task='ti', pred_im=ti_pred_im)

                if args.distributed:
                    dist.barrier()

            loss_list.append(loss)
            
            if args.local_rank == 0 and writer is not None:
                writer.add_scalar('train/loss', loss, step)
                writer.add_scalar('lr/learning_rate', optimizer.param_groups[0]['lr'], step)
                if step % 100 == 0:
                    if args.fp16:
                        print(f'######## Step(Epoch): {step}({epoch}), Loss: {loss}, Scale: {scaler.get_scale()} #########')
                    else:
                        print(f'######## Step(Epoch): {step}({epoch}), Loss: {loss} #########')
                        
                    if wandbrun is not None:
                        wandbrun.log({'train/loss': loss}, step)
                        wandbrun.log({'train/learning_rate': optimizer.param_groups[0]['lr']}, step)
            
            if (step - 1) % args.eval_step == 0:
                logs = validate(model, vq_model, epoch=epoch, writer=writer, fast_validate=True, wandbrun=wandbrun, step=step)
                model.train()

                if args.distributed:
                    dist.barrier()

                if args.local_rank == 0:
                    print('VAL_UER: ', logs[0])
                    print('VAL_BLEU: ', logs[1])
                    print('Saving checkpoint: %d' % epoch)
                    if args.distributed:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)
                    torch.save({'state_dict': state_dict},
                               os.path.join(args.checkpoint_dir, 'Epoch_%04d_%05d_%.2f.ckpt' % (epoch, step, logs[1])))

                    if logs[1] >= best_val_bleu:
                        best_val_bleu = logs[1]
                        bests = glob.glob(os.path.join(args.checkpoint_dir, 'Best_*.ckpt'))
                        for prev in bests:
                            os.remove(prev)
                        torch.save({'state_dict': state_dict},
                                   os.path.join(args.checkpoint_dir, 'Best_%04d_%05d_%.2f.ckpt' % (epoch, step, logs[1])))

            if (step - 1) == args.tot_iters:
                if args.distributed:
                    dist.barrier()
                assert 1 == 0, 'Total Iteration Reached'

    if args.local_rank == 0:
        print('Finishing training')

def validate(model, vq_model, fast_validate=False, epoch=0, writer=None, wandbrun=None, step=0):
    with torch.no_grad():
        model.eval()

        val_data = AVTDataset(
            coco_path=args.coco_path, 
            flickr_path=args.flickr_path, 
            spcoco_path=args.spcoco_path, 
            spflickr_path=args.spflickr_path, 
            spcoco_split_path=args.spcoco_split_path, 
            karpathy_split_path=args.karpathy_split_path, 
            mode='val', 
            num_im_unit=8192, 
            max_sp_len=args.max_sp_len,
            max_txt_len=args.max_txt_len, 
            test_data='coco',
            tokenizer='bert-base-uncased',
            image_tokenizer=args.image_tokenizer,
            architecture=args.architecture,
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=False,
            collate_fn=lambda x: val_data.collate_fn(x),
        )

        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(5 * batch_size, int(len(dataloader.dataset)))
            max_batches = 5
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        it_uer_list, st_uer_list, si_acc_list, ti_acc_list = [], [], [], []
        it_gts = []
        it_preds = []

        gt_im = None

        if args.local_rank == 0:
            if os.path.exists(os.path.join(args.temp_dir, 'unit')) \
            or os.path.exists(os.path.join(args.temp_dir, 'si_gt_images')) \
            or os.path.exists(os.path.join(args.temp_dir, 'ti_gt_images')):
                shutil.rmtree(args.temp_dir)

        description = 'Validation on subset of the Val dataset' if fast_validate else 'Validation'
        if args.local_rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 10 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            image_input, sp_unit, txt_unit, sp_unit_len, txt_unit_len, f_names = batch

            gt_im_unit = im_unit_preparation(args, image_input.cuda(), vq_model)

            im_unit = val_data.add_special_room(gt_im_unit.cpu())
            im_unit = val_data.append_eos(im_unit)
            im_unit = val_data.append_bos(im_unit)

            ## im -> text
            if args.im_txt:
                with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                    if hasattr(model, "module"):
                        output_it = model.module.forward_task(im_unit.cuda(), None, None, None, input_modal='image', output_modal='text', inference=True)
                    else:
                        output_it = model.forward_task(im_unit.cuda(), None, None, None, input_modal='image', output_modal='text', inference=True)

                it_gt_txt = val_data.tokenizer.batch_decode(txt_unit, skip_special_tokens=True)
                it_results = val_data.tokenizer.batch_decode(output_it.cpu().detach(), skip_special_tokens=True)

                it_uer = uer_calc(it_results, it_gt_txt)

                it_gts.extend(it_gt_txt)
                it_preds.extend(it_results)
                it_uer_list.extend(it_uer)

                if args.local_rank == 0:
                    if (i % 10) == 0:
                        print("*" * 10, "Image -> Text", "*" * 10)
                        for j in range(image_input.size(0)):
                            print(f"GT: {it_gt_txt[j]}\nPR : {it_results[j]}\n")

                if args.distributed:
                    dist.barrier()

            ## im -> speech
            if args.im_sp:
                with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                    if hasattr(model, "module"):
                        output_is = model.module.forward_task(im_unit.cuda(), None, None, None, input_modal='image', output_modal='speech', inference=True)
                    else:
                        output_is = model.forward_task(im_unit.cuda(), None, None, None, input_modal='image', output_modal='speech', inference=True)
                
                # is_results 
                is_results = output_is[:, 1:].cpu().detach().numpy()
                
                gt_sp_unit = sp_unit[:, 1:].clone()
                gt_sp_unit = val_data.del_special_room(gt_sp_unit)
                is_results = val_data.del_special_room(is_results)

                gt_sp_unit = decode(gt_sp_unit)
                is_results = decode(is_results)

                pred_sp_unit = [[int(u) for u in unit.split()] for unit in is_results]

                if args.local_rank == 0:
                    for pred_sp, f_name in zip(pred_sp_unit, f_names):
                        save_name = os.path.join(args.temp_dir, 'is_unit', f_name + '.unit')
                        if not os.path.exists(os.path.dirname(save_name)):
                            os.makedirs(os.path.dirname(save_name))
                        torch.save(pred_sp, save_name)

                if args.distributed:
                    dist.barrier()

            ## speech -> text
            if args.sp_txt:
                with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                    if hasattr(model, "module"):
                        output_st = model.module.forward_task(sp_unit.cuda(), sp_unit_len, None, None, input_modal='speech', output_modal='text', inference=True)
                    else:
                        output_st = model.forward_task(sp_unit.cuda(), sp_unit_len, None, None, input_modal='speech', output_modal='text', inference=True)
                
                st_gt_txt = val_data.tokenizer.batch_decode(txt_unit, skip_special_tokens=True)
                st_results = val_data.tokenizer.batch_decode(output_st.cpu().detach(), skip_special_tokens=True)

                st_uer = uer_calc(st_results, st_gt_txt)
                st_uer_list.extend(st_uer)

                if args.local_rank == 0:
                    if (i % 10) == 0:
                        print("*" * 10, "Speech -> Text", "*" * 10)
                        for j in range(image_input.size(0)):
                            print(f"GT: {st_gt_txt[j]}\nPR : {st_results[j]}\n")

                if args.distributed:
                    dist.barrier()

            ## text -> speech
            if args.txt_sp:
                with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                    if hasattr(model, "module"):
                        output_ts = model.module.forward_task(txt_unit.cuda(), txt_unit_len, None, None, input_modal='text', output_modal='speech', inference=True)
                    else:
                        output_ts = model.forward_task(txt_unit.cuda(), txt_unit_len, None, None, input_modal='text', output_modal='speech', inference=True)                   
                
                # ts_results 
                ts_results = output_ts[:, 1:].cpu().detach().numpy()
                
                gt_sp_unit = sp_unit[:, 1:].clone()
                gt_sp_unit = val_data.del_special_room(gt_sp_unit)
                ts_results = val_data.del_special_room(ts_results)

                gt_sp_unit = decode(gt_sp_unit)
                ts_results = decode(ts_results)

                pred_sp_unit = [[int(u) for u in unit.split()] for unit in ts_results]

                if args.local_rank == 0:
                    for pred_sp, f_name in zip(pred_sp_unit, f_names):
                        save_name = os.path.join(args.temp_dir, 'ts_unit', f_name + '.unit')
                        if not os.path.exists(os.path.dirname(save_name)):
                            os.makedirs(os.path.dirname(save_name))
                        torch.save(pred_sp, save_name)

                if args.distributed:
                    dist.barrier()

            if i >= max_batches:
                break

        ## speech -> im
        if args.sp_im:
            with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                if hasattr(model, "module"):
                    output_si = model.module.forward_task(sp_unit.cuda(), sp_unit_len, None, None, input_modal='speech', output_modal='image', inference=True)
                else:
                    output_si = model.forward_task(sp_unit.cuda(), sp_unit_len, None, None, input_modal='speech', output_modal='image', inference=True)
            
            # si_results 
            si_results = output_si[:, 1:33].cpu().detach().numpy()
            
            si_results = val_data.del_special_room(si_results)
            si_results[si_results < 0] = 0

            if si_results.shape[1] < 32:
                si_results = np.concatenate([si_results, np.ones([si_results.shape[0], 32 - si_results.shape[1]])], 1)

            for b_size in range(si_results.shape[0]):
                acc = (si_results[b_size] == gt_im_unit.cpu().numpy()[b_size]).mean()
                si_acc_list.append(acc)

            if args.generate_im:
                gt_si_im = im_decode(args, gt_im_unit[:args.num_gen_im], vq_model)
                si_pred_im = im_decode(args, si_results[:args.num_gen_im], vq_model)
                
                if args.distributed:
                    dist.barrier()

                if args.local_rank == 0:
                    for gt_im, pred_im, f_name in zip(gt_si_im, si_pred_im, f_names):
                        gt_save_name = os.path.join(args.temp_dir, 'si_gt_images', f_name + '.jpg')
                        save_name = os.path.join(args.temp_dir, 'si_images', f_name + '.jpg')
                        if not os.path.exists(os.path.dirname(save_name)):
                            os.makedirs(os.path.dirname(save_name))
                        if not os.path.exists(os.path.dirname(gt_save_name)):
                            os.makedirs(os.path.dirname(gt_save_name))
                        gt_im.save(gt_save_name)
                        pred_im.save(save_name)

            if args.distributed:
                dist.barrier()

        ## text -> im
        if args.txt_im:
            with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                if hasattr(model, "module"):
                    output_ti = model.module.forward_task(txt_unit.cuda(), txt_unit_len, None, None, input_modal='text', output_modal='image', inference=True)
                else:
                    output_ti = model.forward_task(txt_unit.cuda(), txt_unit_len, None, None, input_modal='text', output_modal='image', inference=True)
            
            # ti_results 
            ti_results = output_ti[:, 1:33].cpu().detach().numpy()
        
            ti_results = val_data.del_special_room(ti_results)
            ti_results[ti_results < 0] = 0

            if ti_results.shape[1] < 32:
                ti_results = np.concatenate([ti_results, np.ones([ti_results.shape[0], 32 - ti_results.shape[1]])], 1)

            for b_size in range(ti_results.shape[0]):
                acc = (ti_results[b_size] == gt_im_unit.cpu().numpy()[b_size]).mean()
                ti_acc_list.append(acc)

            if args.generate_im:
                gt_ti_im = im_decode(args, gt_im_unit[:args.num_gen_im], vq_model)
                ti_pred_im = im_decode(args, ti_results[:args.num_gen_im], vq_model)

                if args.distributed:
                    dist.barrier()

                if args.local_rank == 0:
                    for gt_im, pred_im, f_name in zip(gt_ti_im, ti_pred_im, f_names):
                        gt_save_name = os.path.join(args.temp_dir, 'ti_gt_images', f_name + '.jpg')
                        save_name = os.path.join(args.temp_dir, 'ti_images', f_name + '.jpg')
                        if not os.path.exists(os.path.dirname(save_name)):
                            os.makedirs(os.path.dirname(save_name))
                        if not os.path.exists(os.path.dirname(gt_save_name)):
                            os.makedirs(os.path.dirname(gt_save_name))
                        gt_im.save(gt_save_name)
                        pred_im.save(save_name)

            if args.distributed:
                dist.barrier()

        if args.im_txt:
            it_gts = [[it_gt] for it_gt in it_gts]
            it_BLEU_score = sacrebleu.corpus_bleu(it_preds, it_gts).format()
            if args.local_rank == 0:
                print("## IM -> Text BLEU ## \n", it_BLEU_score)
            it_BLEU_score = float(it_BLEU_score.split()[2])          

            if args.local_rank == 0 and writer is not None:
                writer.add_scalar('val/it_uer', np.mean(it_uer_list), step)
                writer.add_scalar('val/it_bleu', it_BLEU_score, step)
                if wandbrun is not None:
                    wandbrun.log({'val/it_uer': np.mean(it_uer_list)}, step)
                    wandbrun.log({'val/it_bleu': it_BLEU_score}, step)

        if args.sp_txt:  
            if args.local_rank == 0 and writer is not None:
                writer.add_scalar('val/st_uer', np.mean(st_uer_list), step)
                if wandbrun is not None:
                    wandbrun.log({'val/st_uer': np.mean(st_uer_list)}, step)

        if args.sp_im:
            if args.local_rank == 0 and writer is not None:
                writer.add_scalar('val/si_acc', np.mean(si_acc_list), step)
                if wandbrun is not None:
                    wandbrun.log({'val/si_acc': np.mean(si_acc_list)}, step)
                if args.generate_im:
                    for kk, (predict, truth) in enumerate(list(zip(si_pred_im, gt_si_im))[:args.num_gen_im]):
                        truth = truth.resize((224, 224))
                        predict = predict.resize((224, 224))
                        writer.add_image(f'val/GT_si/{kk}', np.array(truth), global_step=step, dataformats='HWC')
                        writer.add_image(f'val/Pred_si/{kk}', np.array(predict), global_step=step, dataformats='HWC')
                        if wandbrun is not None:
                            wandbrun.log({f'val/GT_si/{kk}': wandb.Image(truth)}, step)
                            wandbrun.log({f'val/Pred_si/{kk}': wandb.Image(predict)}, step)

        if args.txt_im:
            if args.local_rank == 0 and writer is not None:
                writer.add_scalar('val/ti_acc', np.mean(ti_acc_list), step)
                if wandbrun is not None:
                    wandbrun.log({'val/ti_acc': np.mean(ti_acc_list)}, step)                
                if args.generate_im:
                    for kk, (predict, truth) in enumerate(list(zip(ti_pred_im, gt_ti_im))[:args.num_gen_im]):
                        truth = truth.resize((224, 224))
                        predict = predict.resize((224, 224))
                        writer.add_image(f'val/GT_ti/{kk}', np.array(truth), global_step=step, dataformats='HWC')
                        writer.add_image(f'val/Pred_ti/{kk}', np.array(predict), global_step=step, dataformats='HWC')
                        if wandbrun is not None:
                            wandbrun.log({f'val/GT_ti/{kk}': wandb.Image(truth)}, step)
                            wandbrun.log({f'val/Pred_ti/{kk}': wandb.Image(predict)}, step)

        if args.txt_sp:
            speech_generation(args, task='ts')

        if args.im_sp:
            speech_generation(args, task='is')
            is_BLEU_score = speech_bleu(args, task='is')
            if args.local_rank == 0 and writer is not None:
                writer.add_scalar('val/is_bleu', is_BLEU_score, step)
                if wandbrun is not None:
                    wandbrun.log({'val/is_bleu': is_BLEU_score}, step)

        if args.im_txt and args.im_sp:
            return np.mean(it_uer_list), is_BLEU_score
        elif args.im_txt:
            return np.mean(it_uer_list), it_BLEU_score
        elif args.im_sp:
            return 1.0, is_BLEU_score
        else:
            return 1.0, 0.0

def generate_key_mask(length, sz):
    masks = []
    for i in range(length.size(0)):
        mask = [1] * length[i]
        mask += [0] * (sz - length[i])
        masks += [torch.tensor(mask)]
    masks = torch.stack(masks, dim=0)
    return masks

def gen_to_pil(x):
    x = torch.clamp(x, 0., 1.)
    x = x.permute(1,2,0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def uer_calc(predict, truth):
    uer = []
    for pred, truth in zip(predict, truth):
        uer.append(1.0 * editdistance.eval(pred.split(' '), truth.split(' ')) / len(truth.split(' ')))
    return uer

def decode(units):
    out = list()
    if not isinstance(units, np.ndarray):
        units = units.numpy()
    for unit in units:
        valid_unit = list()
        for u in unit:
            if u >= 0:
                valid_unit.append(str(u))
            elif u <= -100: # EOS
                break
            else:
                continue
        out.append(' '.join(valid_unit))
    return out

def process_text(args, output, dataset, txt_unit, txt_unit_len, acc_list, uer_list):
    txt_unit_len -= 1
    results = F.softmax(output.logits, dim=2).cpu()
    _, results = results.topk(1, dim=2)
    results = results.squeeze(dim=2).detach().numpy()

    gt_txt = txt_unit[:, 1:].clone()
    results = results[:, :-1]

    for b_size in range(results.shape[0]):
        acc = (results[b_size, :txt_unit_len[b_size]] == gt_txt.numpy()[b_size, :txt_unit_len[b_size]]).mean()
        if not np.isnan(acc):
            acc_list.append(acc)
    
    gt_txt = dataset.tokenizer.batch_decode(gt_txt, skip_special_tokens=True)
    results = dataset.tokenizer.batch_decode(results, skip_special_tokens=True)

    uer = uer_calc(results, gt_txt)
    uer_list.extend(uer)
    return results, gt_txt, acc_list, uer_list

def process_speech(args, output, dataset, sp_unit, sp_unit_len, acc_list, uer_list):
    sp_unit_len -= 1
    results = F.softmax(output.logits, dim=2).cpu()
    _, results = results.topk(1, dim=2)
    results = results.squeeze(dim=2).detach().numpy()

    gt_sp = sp_unit[:, 1:].clone()
    results = results[:, :-1]

    for b_size in range(results.shape[0]):
        acc = (results[b_size, :sp_unit_len[b_size]] == gt_sp.numpy()[b_size, :sp_unit_len[b_size]]).mean()
        acc_list.append(acc)
    
    gt_sp = dataset.del_special_room(gt_sp)
    results = dataset.del_special_room(results)

    gt_sp = decode(gt_sp)
    results = decode(results)

    uer = uer_calc(results, gt_sp)
    uer_list.extend(uer)
    return results, gt_sp, acc_list, uer_list

def process_image(output, dataset, vq_model, gt_im_unit, acc_list, args, generate=False):
    results = F.softmax(output.logits, dim=2).cpu()
    _, results = results.topk(1, dim=2)
    results = results.squeeze(dim=2).detach().numpy()

    results = results[:, :32]
    results = dataset.del_special_room(results)

    for b_size in range(results.shape[0]):
        acc = (results[b_size] == gt_im_unit.cpu().numpy()[b_size]).mean()
        acc_list.append(acc)
    
    results[results < 0] = 0
    if generate:
        gt_images = im_decode(args, gt_im_unit[:args.num_gen_im], vq_model)
        pred_images = im_decode(args, results[:args.num_gen_im], vq_model)
    else:
        gt_images = None
        pred_images = None

    return results, gt_images, pred_images

def save_log(args, step, results, gt, acc_list, uer_list, loss, writer, wandbrun, text='Image -> Text', task='it', pred_im=None):
    if args.local_rank == 0:
        if step % 100 == 0 and task not in ['si', 'ti']:
            for (predict, truth) in list(zip(results, gt))[:3]:
                print('*' * 5, f' {text} ', '*' * 5)
                print(f'GT: {truth}')
                print(f'PR: {predict}\n')

        if writer is not None:
            writer.add_scalar(f'train/{task}_acc', np.array(acc_list).mean(), step)
            writer.add_scalar(f'train/{task}_loss', loss, step)
            if uer_list is not None:
                writer.add_scalar(f'train/{task}_uer', np.array(uer_list).mean(), step)
            if pred_im is not None:
                for kk, (predict, truth) in enumerate(list(zip(pred_im, gt))):
                    truth = truth.resize((224, 224))
                    predict = predict.resize((224, 224))
                    writer.add_image(f'train/GT_{task}/{kk}', np.array(truth), global_step=step, dataformats='HWC')
                    writer.add_image(f'train/Pred_{task}/{kk}', np.array(predict), global_step=step, dataformats='HWC')
                    if wandbrun is not None:
                        wandbrun.log({f'train/GT_{task}/{kk}': wandb.Image(truth)}, step)
                        wandbrun.log({f'train/Pred_{task}/{kk}': wandb.Image(predict)}, step)

            if wandbrun is not None:
                wandbrun.log({f'train/{task}_acc': np.array(acc_list).mean()}, step)
                wandbrun.log({f'train/{task}_loss': loss}, step)
                if uer_list is not None:
                    wandbrun.log({f'train/{task}_uer': np.array(uer_list).mean()}, step)

@torch.no_grad()
def speech_generation(args, task='is'):
    ##### Wav Gen #####
    print('Generating WAV from Unit')
    from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
    with open('./Vocoder/config.json') as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder('./Vocoder/g_00950000', vocoder_cfg).cuda()
    
    def load_code(in_file):
        unit_paths = glob.glob(f"{in_file}/*.unit")
        for unit_path in unit_paths:
            unit = torch.load(unit_path)
            if len(unit) < 5:
                unit += [0] * (5 - len(unit))
            yield unit_path, unit

    data = load_code(os.path.join(args.temp_dir, f'{task}_unit'))
    for d_path, d in tqdm(data):
        f_name = os.path.splitext(os.path.basename(d_path))[0]
        x = {
            "code": torch.LongTensor(d).view(1, -1).cuda(),
            }
        with torch.no_grad():
            wav = vocoder(x, True)
        wav_array = wav.detach().cpu().numpy()
        if args.local_rank == 0:
            save_name = os.path.join(args.temp_dir, f'{task}_wav', f_name + '.wav')
            if not os.path.exists(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            sf.write(save_name, wav_array, 16000)
    
    if args.distributed:
        dist.barrier()
    del vocoder, x, wav, wav_array
    return

@torch.no_grad()
def speech_bleu(args, task='is'):
    ##### Txt Gen #####
    print('Generating Transcription from WAV')
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    asrmodel = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    asrmodel = asrmodel.cuda()

    def load_wav(in_file):
        wav_paths = glob.glob(f"{in_file}/*.wav")
        for wav_path in wav_paths:
            wav, sample_rate = sf.read(wav_path)
            if len(wav) < 16000:
                wav = np.concatenate([wav, np.zeros([16000 - len(wav)])], axis=0)
            assert sample_rate == 16_000
            yield wav_path, wav, sample_rate

    data = load_wav(os.path.join(args.temp_dir, f'{task}_wav'))
    for d_path, d, sr in tqdm(data):
        f_name = os.path.splitext(os.path.basename(d_path))[0]
        inputs = processor(d, sampling_rate=sr, return_tensors="pt", padding="longest")

        with torch.no_grad():
            logits = asrmodel(inputs.input_values.cuda(), attention_mask=inputs.attention_mask.cuda()).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        assert len(transcription) == 1
        transcription = transcription[0]
        if args.local_rank == 0:
            save_name = os.path.join(args.temp_dir, f'{task}_transcription', f_name + '.txt')
            if not os.path.exists(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            with open(save_name, 'w') as f:
                f.write(transcription)

    if args.distributed:
        dist.barrier()
    del asrmodel, logits, inputs, processor, predicted_ids

    ##### Score Gen #####
    if args.local_rank == 0:
        print('Generating BLEU score')
    gt_lists = {}
    data = json.load(open(os.path.join(args.karpathy_split_path, 'dataset_coco.json')))
    for d in data['images']:
        if d['split'] == 'val':
            im_name = d['filename'][:-4]
            captions = []
            for c in d['sentences']:
                captions.append(c['raw'].lower())
            gt_lists[im_name] = captions

    refs = []
    preds = []
    pred_files = glob.glob(os.path.join(args.temp_dir, f'{task}_transcription', '*.txt'))
    for p in pred_files:
        refs.append(gt_lists[os.path.basename(p)[:-4]])
        with open(p, 'r') as txt:
            try:
                preds.append(txt.readlines()[0].strip().lower())               
            except:
                preds.append(' ')             
    
    refs = list(zip(*refs))
    BLEU_score = sacrebleu.corpus_bleu(preds, refs).format()
    if args.local_rank == 0:
        print(f'{task} BLEU:\n', BLEU_score)
    BLEU_score = float(BLEU_score.split()[2])
    return BLEU_score

@torch.no_grad()
def im_unit_preparation(args, image_input, vq_model):
    if args.image_tokenizer == 'SEED':
        image_tokens = vq_model.encode(image_torch=image_input) # B,T
    return image_tokens

@torch.no_grad()
def im_decode(args, im_unit, vq_model):
    if args.image_tokenizer == 'SEED':
        if not torch.is_tensor(im_unit):
            im_unit = torch.tensor(im_unit)
        images = vq_model.decode(im_unit.cuda())
    else:
        assert NotImplementedError
    return images #list of PIL Image

if __name__ == "__main__":
    args = parse_args()
    train_net(args)