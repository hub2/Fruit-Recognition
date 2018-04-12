from PIL import Image
import os
import numpy as np
import random
import struct

def convert_one(path, out_dir):
    byt = image_to_labeled_bytearray(path, 0)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, "batch_1.bin"), "wb") as f:
        f.write(byt)


def image_to_labeled_bytearray(path, label):
    i = Image.open(path)
    i.thumbnail((50,50), Image.ANTIALIAS)
    a = np.array(i)
    rs = bytearray(a[:,:,0])
    gs = bytearray(a[:,:,1])
    bs = bytearray(a[:,:,2])
    return bytearray([label]) + rs + gs + bs

def get_classes(path):
    classes = {}
    i = 0
    for dirname in sorted(os.listdir(path)):
        classes[dirname] = i
        i+=1
    return classes

def convert(path):
    files = []
    classes = get_classes(path)
    i = 0

    for dirname in sorted(os.listdir(path)):
        for sample in os.listdir(os.path.join(path, dirname)):
            files.append((path, dirname, sample))


    random.shuffle(files)
    out_file = bytearray()
    for file in files:
        dirname = file[1]
        _class = classes[dirname]

        out_file += image_to_labeled_bytearray(os.path.join(*file), _class)

    with open("%sBatches/batch_1.bin" % path, "wb") as f:
        f.write(out_file)

if __name__ == '__main__':
    convert("Training")
    convert("Validation")








