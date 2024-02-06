import cv2
import os
import argparse
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../metrics'))
from metrics import psnr


ap = argparse.ArgumentParser()
ap.add_argument("-o","--original", type=str, help="Root directory of the original dataset", 
                default="/scratch/jingyihuang/lamem/test_data")
ap.add_argument("-s","--contrast", type=str, help="Directory of the generated image",
                default="/scratch/jingyihuang/gen/test_data")
ap.add_argument("-c","--out_csv_path", required=True, type=str)
ap.add_argument("-i","--interpolation", type=str, default=cv2.INTER_AREA)
ap.add_argument("-v","--video", action='store_true')
args = ap.parse_args()


def main():
    if args.video:
        dir_len = len(os.walk(args.original).__next__()[2])
        print("total file number:{}".format(dir_len))
        total = 0

        for i in tqdm(range(1, dir_len)):
            try:
                o_image = "%05d" % i +".png"
                c_image = "%05d" % i +".png"

                original = cv2.imread(args.original + o_image)
                contrast = cv2.imread(args.contrast + c_image)

                o_height, o_width, o_channel = original.shape
                contrast = cv2.resize(contrast, dsize=(o_width,o_height), interpolation=cv2.INTER_AREA)

                total += psnr(original, contrast)

            except Exception as e:
                    print(str(e) + ": Total count mismatch!!!!")

            #if(i%100 == 0):
            #     print("PSNR: {}".format(psnr(original, contrast)))

        video_psnr_mean = total / dir_len
        print("Video PSNR Mean : {}".format(video_psnr_mean))

    else:
        # original = cv2.imread(args.original)
        # contrast = cv2.imread(args.contrast)

        # o_height, o_width, o_channel = original.shape
        # contrast = cv2.resize(contrast, dsize=(o_width, o_height), interpolation=args.interpolation)

        out = []
        scale = 500
        eval_net_name = 'amnet'
        skip_timesteps = range(30, 220, 20)
        seeds = range(30, 160, 30)
        
        for image_name in tqdm(os.listdir(args.contrast)):
            if not os.path.isfile(
                os.path.join(args.contrast, image_name, f"skip{skip_timesteps[-1]}_+_scale{scale}_seed{seeds[-1]}_{eval_net_name}.png")
            ):
                print(f'The sampling is not completed for {image_name}!')
                continue
            # try:
            original = cv2.imread(os.path.join(args.original, image_name+".jpg"))
            original = cv2.resize(original, (256, 256))  # TODO: hard-coded
            # except Exception:
            #     # print(f"No original image for {d}, {image_name}")
            #     print(f"No original image for {image_name}")
            #     continue
            
            tmp = np.zeros((len(skip_timesteps), len(seeds)))
            for i, step in enumerate(skip_timesteps):
                for j, seed in enumerate(seeds):
                    filename = f"skip{step}_+_scale{scale}_seed{seed}_{eval_net_name}.png"
                
                    # try:
                    contrast = cv2.imread(os.path.join(args.contrast, image_name, filename))

                    # except Exception:
                    #     print(f"No contrast image for {image_name}, {filename}")
                    #     tmp.append(np.nan)
                    #     continue
                    res = psnr(original, contrast)
                    tmp[i, j] = res
            out.append(tmp)

        out = np.array(out)
        avg_lst = []
        for i, step in enumerate(skip_timesteps):
            dat = out[:, i, :]
            avg = np.mean(dat)
            avg_lst.append(avg)
        print(avg_lst)

        # if os.path.dirname(args.out_csv_path):
        #     os.makedirs(os.path.dirname(args.out_csv_path), exist_ok=True)

        # with open(args.out_csv_path, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',')
        #     for item in out:
        #         writer.writerow(item)
        # print('csv file saved!')

if __name__ == '__main__':
    main()
