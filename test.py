import argparse
import random
import torch
import numpy as np
import editdistance
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed
import torch.distributed as dist
import glob
import wandb
import editdistance
import sacrebleu, json
import soundfile as sf
import shutil
from tqdm import tqdm
from PIL import Image

# model
from src.models.TMT_ENDE_BERT import TMT
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
    parser.add_argument("--test_data", type=str, default='coco')

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
    parser.add_argument("--mode", type=str, default='test', help='train, test, valid')
    parser.add_argument("--reset_optimizer", default=False, action='store_true')

    parser.add_argument("--fp16", default=False, action='store_true')

    parser.add_argument("--save_name", default='TMT_ENDE_coco')
    parser.add_argument("--architecture", default='bert')
    parser.add_argument("--image_tokenizer", default='SEED', help='SEED')

    parser.add_argument("--generate_im", default=False, action='store_true')
    parser.add_argument("--generation_step", type=int, default=1000)
    parser.add_argument("--num_gen_im", type=int, default=3)
    parser.add_argument("--gen_max_len", type=int, default=256, help='sp/txt generation length if test mode')
    parser.add_argument("--beam_size", type=int, default=5)

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

    args.temp_dir = './test_results/' + args.save_name

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
    
    model = TMT(args, train_data.num_sp_unit, train_data.num_txt, train_data.num_im_unit)

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

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

    _ = test(model, vq_model)

def test(model, vq_model):
    with torch.no_grad():
        model.eval()

        args.num_gen_im = args.batch_size

        val_data = AVTDataset(
            coco_path=args.coco_path, 
            flickr_path=args.flickr_path, 
            spcoco_path=args.spcoco_path, 
            spflickr_path=args.spflickr_path, 
            spcoco_split_path=args.spcoco_split_path, 
            karpathy_split_path=args.karpathy_split_path, 
            mode = args.mode, 
            num_im_unit=8192, 
            max_sp_len=args.max_sp_len,
            max_txt_len=args.max_txt_len, 
            test_data=args.test_data,
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

        description = 'Test'
        if args.local_rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 10 == 0:
                print("******** Test : %d / %d ********" % ((i + 1) * batch_size, samples))
            image_input, sp_unit, txt_unit, sp_unit_len, txt_unit_len, f_names = batch

            gt_im_unit = im_unit_preparation(args, image_input.cuda(), vq_model)

            im_unit = val_data.add_special_room(gt_im_unit.cpu())
            im_unit = val_data.append_bos(im_unit)
            im_unit = val_data.append_eos(im_unit)

            if i == 10:
                break

            ## im -> text
            if args.im_txt:
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
                    for it_result, f_name in zip(it_results, f_names):
                        save_name = os.path.join(args.temp_dir, 'it_transcription', f_name + '.txt')
                        if not os.path.exists(os.path.dirname(save_name)):
                            os.makedirs(os.path.dirname(save_name))
                        with open(save_name, 'w') as txt:
                            txt.write(it_result)

                if args.local_rank == 0:
                    if (i % 10) == 0:
                        print("*" * 10, "Image -> Text", "*" * 10)
                        for j in range(image_input.size(0)):
                            print(f"GT: {it_gt_txt[j]}\nPR : {it_results[j]}\n")

                if args.distributed:
                    dist.barrier()

            ## im -> speech
            if args.im_sp:
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
            ## Only for first caption :: Use test_ASR.py
            if args.sp_txt:
                if hasattr(model, "module"):
                    output_st = model.module.forward_task(sp_unit.cuda(), sp_unit_len, None, None, input_modal='speech', output_modal='text', inference=True)
                else:
                    output_st = model.forward_task(sp_unit.cuda(), sp_unit_len, None, None, input_modal='speech', output_modal='text', inference=True)
                
                st_gt_txt = val_data.tokenizer.batch_decode(txt_unit, skip_special_tokens=True)
                st_results = val_data.tokenizer.batch_decode(output_st.cpu().detach(), skip_special_tokens=True)

                st_uer = uer_calc(st_results, st_gt_txt)
                st_uer_list.extend(st_uer)

                if args.local_rank == 0:
                    for st_result, f_name in zip(st_results, f_names):
                        save_name = os.path.join(args.temp_dir, 'st_transcription', f_name + '.txt')
                        if not os.path.exists(os.path.dirname(save_name)):
                            os.makedirs(os.path.dirname(save_name))
                        with open(save_name, 'w') as txt:
                            txt.write(st_result)

                if args.local_rank == 0:
                    if (i % 10) == 0:
                        print("*" * 10, "Speech -> Text", "*" * 10)
                        for j in range(image_input.size(0)):
                            print(f"GT: {st_gt_txt[j]}\nPR : {st_results[j]}\n")

                if args.distributed:
                    dist.barrier()

            ## text -> speech
            ## Only for first caption :: Use test_ASR.py
            if args.txt_sp:
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

            ## speech -> im
            if args.sp_im:
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
                    acc = np.mean(si_results[b_size] == gt_im_unit.cpu().numpy()[b_size])
                    si_acc_list.append(acc)

                if args.generate_im:
                    si_pred_im = im_decode(args, si_results[:args.num_gen_im], vq_model)
                    
                    if args.distributed:
                        dist.barrier()

                    if args.local_rank == 0:
                        for pred_im, f_name in zip(si_pred_im, f_names):
                            save_name = os.path.join(args.temp_dir, 'si_images', f_name + '.jpg')
                            if not os.path.exists(os.path.dirname(save_name)):
                                os.makedirs(os.path.dirname(save_name))
                            pred_im.save(save_name)

                if args.distributed:
                    dist.barrier()

            ## text -> im
            if args.txt_im:
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
                    acc = np.mean(ti_results[b_size] == gt_im_unit.cpu().numpy()[b_size])
                    ti_acc_list.append(acc)

                if args.generate_im:
                    ti_pred_im = im_decode(args, ti_results[:args.num_gen_im], vq_model)

                    if args.distributed:
                        dist.barrier()

                    if args.local_rank == 0:
                        for pred_im, f_name in zip(ti_pred_im, f_names):
                            save_name = os.path.join(args.temp_dir, 'ti_images', f_name + '.jpg')
                            if not os.path.exists(os.path.dirname(save_name)):
                                os.makedirs(os.path.dirname(save_name))
                            pred_im.save(save_name)

                if args.distributed:
                    dist.barrier()

        if args.im_txt:
            it_bleu = score_generation(args, task='it')
        else:
            it_bleu = None

        if args.sp_txt:  
            st_wer = np.mean(st_uer_list)
        else:
            st_wer = None

        if args.txt_sp:
            speech_generation(args, task='ts')

        if args.im_sp:
            speech_generation(args, task='is')
            text_generation(args, task='is')
            is_bleu = score_generation(args, task='is')
        else:
            is_bleu = None
        
        if it_bleu is not None:
            print("I => T BLEU: ", it_bleu)
        if st_wer is not None:
            print("S => T WER: ", st_wer)
        if is_bleu is not None:
            print("I => S BLEU: ", is_bleu)

        return

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

def process_text(output, dataset, txt_unit, txt_unit_len, acc_list, uer_list):
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

def process_speech(output, dataset, sp_unit, sp_unit_len, acc_list, uer_list):
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
def text_generation(args, task='is'):
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
    return

@torch.no_grad()
def score_generation(args, task='is'):
    ##### Score Gen #####
    if args.local_rank == 0:
        print('Generating BLEU score')
    gt_lists = {}
    if args.test_data == 'coco':
        data = json.load(open(os.path.join(args.karpathy_split_path, 'dataset_coco.json')))
    else:
        data = json.load(open(os.path.join(args.karpathy_split_path, 'dataset_flickr8k.json')))
    for d in data['images']:
        if d['split'] == args.mode:
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