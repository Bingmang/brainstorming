import argparse
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True, help='folder where your data is in')
parser.add_argument('--out', default='imagecut/', help='folder to output cut images')
parser.add_argument('--size', type=int, default=200, help='the height / width of the output cut image')

opt = parser.parse_args()
print(opt)

def mkoutdir(file_dir):
    try:
        os.makedirs(file_dir)
    except OSError:
        pass

def read_input_folder(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def cut_image(file, size, outdir):
    basename, extension = os.path.splitext(os.path.basename(file))
    img = cv2.imread(file)
    height = len(img)
    width = len(img[0])
    cut_num = 0
    for y in range(0, height, size):
        for x in range(0, width, size):
            if x + size > width or y + size > height:
                print(x, y, width, height)
                continue
            # if you want to change the cut direction, just change the code to img[x : x + size, y : y + size]
            cv2.imwrite('%s/%s-%s%s' % (outdir, basename, cut_num, extension), img[y : y + size, x : x + size]) 
            cut_num += 1
    print('image: [%s] cuts out %s images.' % (file, cut_num))

def main():
    mkoutdir(opt.out)
    file_list = read_input_folder(opt.dir)
    if not file_list:
        print('find nothing in your data folder: %s' % opt.dir)
        return
    for file in file_list:
        cut_image(file, opt.size, opt.out)

if __name__ == '__main__':
    main()
