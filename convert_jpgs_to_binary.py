from PIL import Image
import os
import numpy as np
import random
import struct

def image_to_labeled_bytearray(path, label):
    i = Image.open(path)
    i.thumbnail((25,25), Image.ANTIALIAS)
    a = np.array(i)
    rs = bytearray(a[:,:,0])
    gs = bytearray(a[:,:,1])
    bs = bytearray(a[:,:,2])
    return bytearray([label]) + rs + gs + bs

def convert(path):
    files = []
    classes = {}
    i = 0
    print(os.listdir(path))

    for dirname in sorted(os.listdir(path)):
        classes[dirname] = i
        i+=1
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

convert("Training")
convert("Validation")








