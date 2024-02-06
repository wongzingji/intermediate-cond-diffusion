import argparse
import os
import sys
import functools
from tqdm import tqdm
import numpy as np

import torch
from torchvision.transforms import functional as TF
from torchvision import utils
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../guided-diffusion'))
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../mem_nets'))
from assessor.memnet import memnet
sys.path.append(os.path.join(os.path.dirname(__file__), '../dataset'))
from lamem import load_data


def main():
    args = create_argparser()

    print("creating model and diffusion...")
    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '250',  # Modify this value to decrease the number of
                                    # timesteps.
        'image_size': args.image_size,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': False,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model, diffusion = create_model_and_diffusion(
        **model_config
    )
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if model_config['use_fp16']:
        model.convert_to_fp16()
    model.eval()

    print("creating MemNet...")
    mem, in_transform, out_transform = memnet(device, False)
    mem.eval()

    # base image
    if args.base_image:
        assert args.batch_size == 1
        with open(args.base_image, 'rb') as f:
            base_image = Image.open(f).convert('RGB')
            base_image = base_image.resize((args.image_size, args.image_size), Image.LANCZOS) # crop
            base_image = TF.to_tensor(base_image).to(device).unsqueeze(0).mul(2).sub(1) # [-1, 1]
    elif args.base_image_dir:
        test_data = load_data(
            data_dir=args.base_image_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )

    # seed
    torch.manual_seed(60)


    ## -----------------------------------------------
    # lpips_model = lpips.LPIPS(net='vgg').to(device)
    # l1_w, l2_w, l3_w = [1000, 100, 300]
    
    # mem_scores = None
    def cond_fn(operation, x, t, out, y=None):
        '''
        x: x_t
        out: pred res
        '''
        # fac = diffusion.sqrt_one_minus_alphas_cumprod[t] # cur_t
        # x_in = out['pred_xstart'] * fac + x * (1 - fac) # pred x_t, [-1, 1]
        
        # # losses
        # tv_losses = losses.tv_loss(x_in)
        # range_losses = losses.range_loss(out['pred_xstart'])
        pred_scores = out_transform(mem(in_transform(out['pred_xstart'])).cpu())
        # pred_scores = pred_scores.view(-1)
        
        # mem_scores.append(pred_scores.item()) # extract the values

        if operation == 'max':
            loss1 = -pred_scores.sum() * args.mem_scale # maximize score
        else:
            loss1 = pred_scores.sum() * args.mem_scale # minimize score
        # loss2 = tv_losses.sum() * l2_w
        # loss3 = range_losses.sum() * weights[2]

        # init_losses = lpips_model(x_in, base_img)
        # loss3 = init_losses.sum() * l3_w

        # # check if the losses are balanced
        # print(loss1)
        # print(loss2)
        # print(loss3)
        grad = -torch.autograd.grad(loss1, x)[0]
        # print(f'mem loss: {grad}')
        # print(f'tv loss: {-torch.autograd.grad(loss2, x)[0]}')
        return grad # +loss2+loss3


    def create_samples(cond_fn, batch, names, skip_timesteps, operation, mem_scores):
        # TODO: mem_scores
        samples = diffusion.p_sample_loop_progressive(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=batch,
            cond_fn_with_grad=True,
            device=args.device,
        )

        cur_t = diffusion.num_timesteps - skip_timesteps - 1
        for sample in samples: # sample: [3, 256, 256]
            cur_t -= 1
            if cur_t == -1:
                pred_xstart = sample['pred_xstart']
        
        # save the final imag
        for i in range(pred_xstart.size(0)):
            saved_name = f"skip{skip_timesteps}_{operation}_{int(args.mem_scale)}_memnet.png"
            utils.save_image(
                        pred_xstart[i, :, :, :],
                        os.path.join(args.save_dir, names[i], saved_name),
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
            print(f'Image {names[i]}, {saved_name} saved!')
    
    ## -----------------------------------------------
    print("creating samples...")
    
    if not args.decrease:
        operation = 'max'
    else:
        operation = 'min'
    
    stopiter = 5000
    mem_scores = np.zeros([stopiter, len(args.skip_timesteps)])
    cond_fn_ = functools.partial(cond_fn, operation)

    if args.base_image:
        os.makedirs(args.save_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(args.base_image))[0]
        for skip_timesteps in args.skip_timesteps:
            create_samples(cond_fn_, base_image, name, skip_timesteps, operation, mem_scores)
    elif args.base_image_dir:
        for i, sample in enumerate(tqdm(test_data)):
            base_images, target = sample
            if i >= stopiter:
                break
            base_images = base_images.to(device)
            names = []
            for name in target["name"]:
                name = os.path.splitext("/".join((name.split("/")[-3:])))[0]
                names.append(name)
                os.makedirs(os.path.join(args.save_dir, name), exist_ok=True)
            for skip_timesteps in args.skip_timesteps:
                create_samples(cond_fn_, base_images, names, skip_timesteps, operation, mem_scores)


def create_argparser():
    parser = argparse.ArgumentParser()
    
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        '--base_image',
    )
    input_group.add_argument(
        '--base_image_dir'
    )
    parser.add_argument(
        '--batch_size', type=int,
        default=1
    )
    parser.add_argument(
        '--image_size', type=int,
        default=256
    )
    parser.add_argument(
        '--model_path', required=True,
    )
    parser.add_argument(
        '--save_dir', required=True,
    )
    parser.add_argument(
        '--mem_scale', type=int, # TODO: list
        required=True,
    )
    parser.add_argument(
        '--skip_timesteps', type=int, nargs="+",
        default=[30,50,70,90,110,130,150,170,190,210],
    )
    parser.add_argument(
        '--timestep_respacing', default='250',
    )
    parser.add_argument(
        '--decrease', action='store_true'
    )
    parser.add_argument(
        '--device', default='cuda:0'
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()