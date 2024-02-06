import argparse
from argparse import Namespace
import os
import sys
import functools
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision import utils
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../guided-diffusion'))
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../mem_nets'))
from amnet.amnet import AMNet  # noqa
from amnet.config import get_amnet_config
sys.path.append(os.path.join(os.path.dirname(__file__), '../dataset'))
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
        'timestep_respacing': args.timestep_respacing,
        'image_size': 256,
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

    print("creating AMNet...")
    hps = get_amnet_config(args)
    amnet = AMNet()
    amnet.init(hps)
    if hps.use_cuda:
        amnet.model.cuda()  # TODO

    # base image
    transform_amnet = transforms.Compose([
        transforms.Resize([224, 224]),
        lambda x: x.add(1)/2, # [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_transform = transforms.Compose([
        transforms.Resize(((args.image_size, args.image_size)), Image.LANCZOS),
        transforms.ToTensor(),  # [0, 1]
        lambda x: x.mul(2).sub(1), # [-1, 1]
    ])
    if args.base_image:
        assert args.batch_size == 1
        with open(args.base_image, 'rb') as f:
            base_image = Image.open(f).convert('RGB')
            base_image = data_transform(base_image)
    elif args.base_image_dir:
        test_dataset = LaMemEvalDataset(args.base_image_dir, args.csv_file, data_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.save_dir, exist_ok=True)
    
    def cond_fn(
        operation: str, 
        # pr: Callable,
        x, t, out):
        '''
        operation: +, -, 0
        x: x_t
        out: pred res
        '''
        fac = diffusion.sqrt_one_minus_alphas_cumprod[t] # cur_t
        x_in = out['pred_xstart'] * fac + x * (1 - fac) # pred x_t, [-1, 1], [BCHW]
        
        # losses
        # result = amnet.predict_memorability_image_batch(x_in.squeeze().cpu().detach().numpy())

        x_in = transform_amnet(x_in) # out['pred_xstart'], x_in
        output_, outputs_, alphas_ = amnet.model(x_in) # output: None; outputs: tensor, [B, num_step], on cuda
                                                        # alphas: tensor, [B, num_step, 196], on cuda

        # outputs_ = outputs_.cpu().data.numpy()
        # output_ = None if output_ is None else output_.cpu().data.numpy()
        # alphas_ = alphas_.cpu().data.numpy()
        # pr.attention_masks = alphas_.cpu().data.numpy() ##########

        memity, outputs_ = amnet.postprocess(output_, outputs_)  # TODO
        # mem_scores.append(memity.item())
        # pr.outputs = outputs_.cpu().data.numpy() ########

        # att_maps = result.attention_masks
        # num_images = len(att_maps)
        # seq_len = result.outputs.shape[1]
        # ares = int(np.sqrt(att_maps.shape[2]))

        # amaps_imgs = []
        # for b in range(num_images):
        #     for s in range(seq_len):
        #         img_alpha = att_maps[b,s]
        #         img_alpha = img_alpha.reshape((ares, ares))
        #         # normalize, *255
        #         heat_map_img = cv2.resize(img_alpha, (256, 256), interpolation=cv2.INTER_CUBIC)
        #         amaps_imgs.append(heat_map_img)
        # amaps_imgs = torch.Tensor(amaps_imgs) # batch
        
        if operation == '+':
            loss1 = -memity.sum() * args.mem_scale # maximize score
        elif operation == '-':
            loss1 = memity.sum() * args.mem_scale # minimize score
        else:
            loss1 = memity.sum() * 0
        grad = -torch.autograd.grad(loss1, x)[0]
        
        return grad


    def create_samples(cond_fn, batch, names, skip_timesteps, operation, seed, mem_scores):

        samples = diffusion.p_sample_loop_progressive(
            model=model,
            shape=(args.batch_size, 3, args.image_size, args.image_size),
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
            saved_name = f"skip{skip_timesteps}_{operation}_scale{args.mem_scale}_seed{seed}_amnet.png"
            utils.save_image(
                        pred_xstart[i, :, :, :],
                        os.path.join(args.save_dir, names[i], saved_name),
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
            print(f'Image {names[i]}, {saved_name} saved!')

    # def mem_score_plot(mem_scores):
    #     # mem trend plot
    #     xs = range(len(mem_scores[0]))
    #     plt.plot(xs, mem_scores[0], label='minimum')
    #     plt.plot(xs, mem_scores[1], label='maximum')
    #     plt.title('Change of Memorability Score predicted by MemNet in the sampling process')
    #     plt.xlabel('iteration')
    #     plt.ylabel('mem score')
    #     plt.legend()
    #     # plt.savefig(os.path.join(args.save_dir, f"sampled_{os.path.basename(args.base_image).split('.')[0]}_skip{args.skip_timesteps}_{operation}_{int(mem_scale)}_mem_plot.png"))
    #     plt.savefig('out_vis/plot.png')

    print("creating samples...")

    stopiter = 5000
    # TODO
    # pr = PredictionResult() #########
    # pr.image_names = [args.base_image]

    if args.base_image: # single image
        # save_name = f"sampled_{os.path.basename(args.base_image).split('.')[0]}_skip{args.skip_timesteps}_{operation}_{int(args.mem_scale)}_amnet.png"
        # create_samples(cond_fn_, base_img, [save_name])

        # pr.write_attention_maps('/home/jingyihuang/clip-diffusion/out_vis') # TODO: parameterize
        # print(mem_scores)
        # mem_score_plot(mem_scores)
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
    else: # dataloader
        for i, sample in enumerate(tqdm(test_loader)):
            base_images, target = sample
            if i >= stopiter:  # TODO
                break
            base_images = base_images.to(device)
            names = []
            for name in target["name"]:
                name = os.path.splitext("/".join((name.split("/")[-3:])))[0]
                names.append(name)
                os.makedirs(os.path.join(args.save_dir, name), exist_ok=True)
            
            for skip_timesteps in args.skip_timesteps:
                for seed in args.seed:
                    mem_scores = []
                    torch.manual_seed(seed)
                    for op in ['+']:#, '0', '-']:
                        cond_fn_ = functools.partial(cond_fn, op)
                        create_samples(cond_fn_, base_images, names, skip_timesteps, op, seed, mem_scores)
    

def create_argparser():
    parser = argparse.ArgumentParser()
    img = parser.add_mutually_exclusive_group() # required=True

    '''For diffusion model'''
    img.add_argument(
        '--base_image',
    )
    img.add_argument(
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
        '--csv_file', type=str,
        help="split file"
    )
    parser.add_argument(
        '--model_path', #required=True, TODO
    )
    parser.add_argument(
        '--save_dir', #required=True,
    )
    parser.add_argument(
        '--mem_scale', #required=True,
        type=int
    )
    parser.add_argument(
        '--skip_timesteps', type=int, nargs="+",
        default=[30,50,70,90,110,130,150,170,190,210],
    )
    parser.add_argument(
        '--seed', nargs='+', default=[60]
    )
    parser.add_argument(
        '--timestep_respacing', default='250',
    )
    
    parser.add_argument(
        '--num_workers', type=int,
        default=4
    )

    parser.add_argument(
        '--device', default='cuda:0'
    )
    args, unknown = parser.parse_known_args()
    
    '''For AMNet'''
    amnet_args = vars(create_amnet_argparser(args, unknown, parser))
    args = vars(args)
    amnet_args.update(args)
    args = Namespace(**amnet_args)
    
    return args


def create_amnet_argparser(args, unknown, parser):
    parser.add_argument('--gpu', default=int(args.device.split(':')[1]), type=int, help='GPU ID. If -1 the application will run on CPU')
    parser.add_argument('--model-weights', default='', type=str, help='pkl file with the model weights')

    parser.add_argument('--cnn', default='ResNet50FC', type=str, help='Name of CNN model for features extraction [ResNet18FC, ResNet50FC, ResNet101FC, VGG16FC, ResNet50FT]')
    parser.add_argument('--att-off', action="store_true", help='Runs training/testing without the visual attention')
    parser.add_argument('--lstm-steps', default=3, type=int,
                        help='Number of LSTM steps. Default 3. To disable LSTM set to zero')

    parser.add_argument('--last-step-prediction', action="store_true",
                        help='Predicts memorability only at the last LSTM step')

    parser.add_argument('--test', action='store_true', help='Run evaluation')

    parser.add_argument('--eval-images', default=args.base_image, type=str, help='Directory or a csv file with images to predict memorability for')
    parser.add_argument('--csv-out', default='', type=str, help='File where to save prediced memorabilities in csv format')
    parser.add_argument('--att-maps-out', default='', type=str, help='Directory where to store attention maps')

    # Training
    parser.add_argument('--epoch-max', default=-1, type=int,
                        help='If not specified, number of epochs will be set according to selected dataset')
    parser.add_argument('--epoch-start', default=0, type=int,
                        help='Allows to resume training from a specific epoch')

    parser.add_argument('--train-batch-size', default=-1, type=int,
                        help='If not specified a default size will be set according to selected dataset')
    parser.add_argument('--test-batch-size', default=-1, type=int,
                        help='If not specified a default size will be set according to selected dataset')

    # Dataset configuration
    parser.add_argument('--dataset', default='lamem', type=str, help='Dataset name [lamem, sun]')
    parser.add_argument('--experiment', default='', type=str, help='Experiment name. Usually no need to set' )
    parser.add_argument('--dataset-root', default='', type=str, help='Dataset root directory')
    parser.add_argument('--images-dir', default='images', type=str, help='Relative path to the test/train images')
    parser.add_argument('--splits-dir', default='splits', type=str, help='Relative path to directory with split files')
    parser.add_argument('--train-split', default='', type=str, help='Train split filename e.g. train_2')
    parser.add_argument('--val-split', default='', type=str, help='Validation split filename e.g. val_2')
    parser.add_argument('--test-split', default='', type=str, help='Test split filename e.g. test_2')

    amnet_args = parser.parse_args(unknown)
    
    return amnet_args


if __name__ == '__main__':
    main()