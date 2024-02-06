import argparse
import os
import sys
import functools
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision import utils
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../../mem_nets'))
from resmem.resmem.model import ResMem
sys.path.append(os.path.join(os.path.dirname(__file__), '../guided-diffusion'))
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../../dataset'))
from LaMemDataset import LaMemEvalDataset


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

    print("creating ResMem...")

    # base image
    transform_resmem = transforms.Compose([
        transforms.CenterCrop(size=(227, 227)),
        lambda x: x.add(1).div(2), # [0, 1]
    ])
    data_transform = transforms.Compose([
        transforms.Resize(((args.image_size, args.image_size)), Image.LANCZOS),
        transforms.ToTensor(),  # [0, 1]
        lambda x: x.mul(2).sub(1), # [-1, 1]
    ])
    if args.base_image: # TODO: mem_score
        assert args.batch_size == 1
        with open(args.base_image, 'rb') as f:
            base_image = Image.open(f).convert('RGB') # HWC
            base_image = data_transform(base_image)
    elif args.base_image_dir:
        test_dataset = LaMemEvalDataset(args.base_image_dir, args.csv_file, data_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    resmem = ResMem(pretrained=True).to(device)
    resmem.eval()

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
        image_in = transform_resmem(out['pred_xstart'])
        pred_scores = resmem.forward(image_in).view(args.batch_size, -1).mean(1) # TODO: batch_size is imposed to be 1
        # pred_scores = pred_scores.view(-1)
        
        # mem_scores.append(pred_scores.item()) # extract the values

        if operation == '+':
            loss1 = -pred_scores.sum() * args.mem_scale # maximize score
        elif operation == '-':
            loss1 = pred_scores.sum() * args.mem_scale # minimize score
        else:
            loss1 = pred_scores.sum() * 0
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


    def create_samples(cond_fn, batch, names, skip_timesteps, operation, seed, mem_scores):
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
        
        # print(mem_scores)
        # save the final image
        for i in range(pred_xstart.size(0)):
            saved_name = f"skip{skip_timesteps}_{operation}_scale{args.mem_scale}_seed{seed}_resmem.png"
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
    
    stopiter = 5000

    if args.base_image:
        name = os.path.splitext(os.path.basename(args.base_image))[0]
        os.makedirs(os.path.join(args.save_dir, name), exist_ok=True)
        base_image = base_image.to(device).unsqueeze(0)

        for skip_timesteps in args.skip_timesteps:
            for seed in args.seed:
                torch.manual_seed(seed)
                for op in ['+', '0', '-']:
                    mem_scores = []
                    cond_fn_ = functools.partial(cond_fn, op)
                    create_samples(cond_fn_, base_image, [name], skip_timesteps, op, seed, mem_scores)
    # TODO
    elif args.base_image_dir:
        for i, sample in enumerate(tqdm(test_loader)):
            base_images, targets = sample
            if i >= stopiter:
                break
            base_images = base_images.to(device)
            names = []
            for name in targets["name"]:
                name = os.path.splitext("/".join((name.split("/")[-3:])))[0]
                names.append(name)
                os.makedirs(os.path.join(args.save_dir, name), exist_ok=True)
            for skip_timesteps in args.skip_timesteps:
                for seed in args.seed:
                    mem_scores = []
                    for op in ['+', '0', '-']:
                        cond_fn_ = functools.partial(cond_fn, op)
                        create_samples(cond_fn_, base_images, names, skip_timesteps, op, seed, mem_scores)


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
        '--csv_file', type=str,
        default=None,
        help="split file"
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
        '--seed', type=int, nargs="+",
        default=[60],
    )
    parser.add_argument(
        '--timestep_respacing', default='250',
    )
    parser.add_argument(
        '--device', default='cuda:0'
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()