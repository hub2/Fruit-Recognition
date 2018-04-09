from PIL import Image
import os
import numpy as np
import random
import struct
def convert(path):
    files = []
    classes = {}
    i = 0
    for dirname in os.listdir(path):
        classes[dirname] = i
        i+=1
        for sample in os.listdir(os.path.join(path, dirname)):
            files.append((path, dirname, sample))
    #classes = {"Apple Golden 3": 0, "Banana": 1, "Cherry": 2, "Cocos": 3, "Mango": 4, "Orange": 5, "Pineapple": 6}

    random.shuffle(files)
    out_file = bytearray()
    for file in files:
        dirname = file[1]
        _class = classes[dirname]
        i = Image.open(os.path.join(*file))
        i.thumbnail((25,25), Image.ANTIALIAS)
        a = np.array(i)
        rs = bytearray(a[:,:,0])
        gs = bytearray(a[:,:,1])
        bs = bytearray(a[:,:,2])

        out_file += bytearray([_class]) + rs + gs + bs

    with open("%sBatches/batch_1.bin" % path, "wb") as f:
        f.write(out_file)

convert("Training")
convert("Validation")










