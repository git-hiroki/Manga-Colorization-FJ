import os
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from colorizator import MangaColorizator
from PIL import Image, ImageStat
import cv2
def is_grayscale(path):
    thumb_size=40
    MSE_cutoff=22
    adjust_color_bias=True
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    thumb = im.resize((thumb_size,thumb_size))
    SSE, bias = 0, [0,0,0]
    if adjust_color_bias:
        bias = ImageStat.Stat(thumb).mean[:3]
        bias = [b - sum(bias)/3 for b in bias ]
    for pixel in thumb.getdata():
        mu = sum(pixel)/3
        SSE += sum((pixel[i] - mu - bias[i])*(pixel[i] - mu - bias[i]) for i in [0,1,2])
    MSE = float(SSE)/(thumb_size*thumb_size)
    if MSE <= MSE_cutoff:
        return True
    else:
        return False
		
def process_image(image, colorizator, args, iscolor):
    colorizator.set_image(image, args.size, args.denoiser, args.denoiser_sigma)
        
    return colorizator.colorize(iscolor=iscolor)
    
def colorize_single_image(image_path, save_path, colorizator, args, iscolor):
    
        #image = plt.imread(image_path)

        image=cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),-1)
        if image.ndim == 3 and iscolor == 0:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        #image = cv2.imread(image_path)
        # cv2.imshow("lena", image )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        colorization = process_image(image, colorizator, args, iscolor)
        # cv2.imshow("lena", colorization )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()        
        if args.superr:
            cv2.imwrite(save_path, colorization,[int(cv2.IMWRITE_WEBP_QUALITY),75])
            # cv2.imencode('.webp', colorization,[int(cv2.IMWRITE_WEBP_QUALITY),75])[1].tofile(save_path)
        else:
            plt.imsave(save_path, colorization)
        #
        return True
    

def colorize_images(target_path, colorizator, args):
    images = os.listdir(args.path)
    
    for image_name in images:
        if os.path.splitext(image_name)[1].lower() in ('.jpg', '.png', '.jpeg', '.webp'):
            
            file_path = os.path.join(args.path, image_name)
            save_path = os.path.join(target_path, image_name)
            if os.path.isdir(file_path):
                continue
            if args.onlysr==False:
                if args.colorall==False:
                    if is_grayscale(file_path):
                        iscolor=0
                        #continue
                        print("Gray img!: "+str(file_path))
                        #continue
                    else:
                        iscolor=1
                        if args.superr == False:
                            plt.imsave(save_path,plt.imread(file_path))
                            print("Color img!: "+str(file_path)+" | COPY!")
                            continue
                        else:
                            print("Color img!: "+str(file_path)+" | SR!")
                            pass
                else:
                    iscolor=0
            else:
                iscolor=1
            name, ext = os.path.splitext(image_name)
            image_name = name + '.webp'
            save_path = os.path.join(target_path, image_name)
            #print(file_path)
                    
            colorize_single_image(file_path, save_path, colorizator, args, iscolor)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path",default = './input', help='input dir/file')# required=True
    parser.add_argument("-op", "--outputpath",default = './output', help='output dir')# required=True
    parser.add_argument("-gen", "--generator", default = 'networks/generator.pt')
    parser.add_argument("-sur", "--surperpath", default = 'networks/RealESRGAN_x4plus_anime_6B.pt')
    parser.add_argument("-ext", "--extractor", default = 'networks/extractor.pth')
    parser.add_argument('-g', '--gpu', dest = 'gpu', action = 'store_true', help= "Use gpu")
    parser.add_argument('-nd', '--no_denoise', dest = 'denoiser', action = 'store_false', help= "No denoiser before color")
    parser.add_argument("-ds", "--denoiser_sigma", type = int, default = 25, help="Denoiser_sigma")
    parser.add_argument("-s", "--size", type = int, default = 576, help='Color output size' )
    parser.add_argument('-ct', '--colortile', type=int, default=0, help='Color Tile size, 0 for no tile')
    parser.add_argument('-st', '--srtile', type=int, default=256, help='SR Tile size, 0 for no tile')
    parser.add_argument('--tile_pad', type=int, default=8, help='Tile padding')
    parser.add_argument('-nosr', '--no_superr', dest = 'superr', action = 'store_false', help='SR or not SR by RealESRGAN_x4plus_anime_6B aftercolored')
    parser.add_argument('-ca', '--color_all', dest = 'colorall', action = 'store_true', help= "colorall images, no skip color one")
    parser.add_argument('-onlysr', '--only_sr', dest = 'onlysr', action = 'store_true', help= "only SR all images, no color")
    parser.add_argument('-sub', '--all_subdir', dest = 'subdir', action = 'store_true', help= "handle all input sub folders")
    # https://github.com/xinntao/Real-ESRGAN/
    parser.set_defaults(gpu = False)
    parser.set_defaults(superr = True)
    parser.set_defaults(denoiser = True)
    parser.set_defaults(colorall = False)
    parser.set_defaults(onlysr = False)
    parser.set_defaults(subdir = False)
    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    
    args = parse_args()
    
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'#"cpu"
        
    colorizer = MangaColorizator(device, args.generator, args.extractor, args.surperpath, args.superr, args.colortile, args.srtile, args.tile_pad)
    if args.subdir == False:
        if os.path.isdir(args.path):
            #colorization_path = os.path.join(args.outputpath, 'colorization')
            colorization_path = args.outputpath
            if not os.path.exists(colorization_path):
                os.makedirs(colorization_path)              
            colorize_images(colorization_path, colorizer, args)
            
        elif os.path.isfile(args.path):
            
            split = os.path.splitext(args.path)

            if split[1].lower() in ('.jpg', '.png', '.jpeg', '.webp'):
                new_image_path = args.outputpath+'/'+os.path.basename(split[0]) + '.webp'
                if args.onlysr:
                    iscolor=1
                else:
                    iscolor=0
                colorize_single_image(args.path, new_image_path, colorizer, args, iscolor)
            else:
                print('Wrong format, pass')
        else:
            print('Wrong path')
    else: 
        for root, dirs, files in os.walk(args.path):
            for filename in files:
                if filename.endswith(".jpg") or filename.endswith(".JPG")\
                or filename.endswith(".jepg") or filename.endswith(".JEPG")\
                or filename.endswith(".png") or filename.endswith(".PNG")\
                or filename.endswith(".webp") or filename.endswith(".WEBP"):
                    name = os.path.splitext(os.path.basename(filename))[0]
                    imgpath = root+'/'+filename
                    outpath=imgpath.replace(args.path,args.outputpath)
                    new_image_path = outpath.replace(os.path.splitext(os.path.basename(filename))[1],".webp")
                    if os.path.isdir(os.path.dirname(new_image_path)):
                        pass
                    else:
                        os.mkdir(os.path.dirname(new_image_path))
                    if args.onlysr==False:
                        if args.colorall==False:
                            if is_grayscale(imgpath):
                                iscolor=0
                                print("Gray img!: "+str(imgpath))
                            else:
                                iscolor=1
                                if args.superr == False:
                                    plt.imsave(new_image_path, plt.imread(imgpath))
                                    print("Color img!: "+str(imgpath)+" | COPY!")
                                    continue
                                else:
                                    print("Color img!: "+str(imgpath)+" | SR!")
                                    pass
                        else:
                            iscolor=0
                    else:
                        iscolor=1
                    colorize_single_image(imgpath, new_image_path, colorizer, args, iscolor)

                  
