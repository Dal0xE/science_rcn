import numpy as np
import os
from multiprocessing import Pool
from functools import partial
from scipy.misc import imresize
from scipy.ndimage import imread

from science_rcn.inference import test_image
from science_rcn.learning import train_image

from cmd import Cmd

model_args = ['data_dir', 'train_size', 'test_size', 'full_test_set', 'pool_shape', 'perturb_factor', 'parallel', 'verbose', 'seed']
data_dir = 'data'
train_size = 20
test_size = 20
full_test_set = False
pool_shape = (25, 25)
perturb_factor = 2.
parallel = True
verbose = False
seed = 5

all_model_factors = None

num_workers = None if parallel else 1
pool = Pool(num_workers)

def grab_training_data(ndir, train_size, test_size, full_test_set=False, seed=5):
    if not os.path.isdir(ndir)
    def _load_data(image_dir, num_per_class, get_filenames=False):
        loaded_data = []
        for category in sorted(os.listdir(image_dir)):
            cat_path = os.path.join(image_dir, category)
            if not os.path.isdir(cat_path) or category.startswith('.'):
                continue
            if num_per_class is None:
                samples = sorted(os.listdir(cat_path))
            else:
                samples = np.random.choice(sorted(os.listdir(cat_path)), num_per_class)

            for fname in samples:
                filepath = os.path.join(cat_path, fname)
                # Resize and pad the images to (200, 200)
                image_arr = imresize(imread(filepath), (112, 112))
                img = np.pad(image_arr,
                             pad_width=tuple([(p, p) for p in (44, 44)]),
                             mode='constant', constant_values=0)
                loaded_data.append((img, category))
        return loaded_data

    np.random.seed(seed)
    train_set = _load_data(os.path.join(ndir, 'training'), num_per_class=train_size // 10)
    return train_set

class LearningPrompt(Cmd):
    def do_train(self, args):
        if len(args) == 0:
            if data_dir == 'data':
                print "Please set a location for training data, or specify it as an argument."
                return
            print "Training on preset data..."
            train_data = grab_training_data(data_dir)
        else:
            if args.startswith('/'):
                data_dir = args
            else:
                data_dir += args
        train_partial = partial(train_image, perturb_factor=perturb_factor)
        train_results = pool.map_async(train_partial, [d[0] for d in train_data]).get(9999999)
        all_model_factors = zip(*train_results)
    def do_set(self, args):
        '''Sets values for model arguments. Can handle one or multiple arguments at once. Use 'argument: value' notation, and be sure to enclose strings in quotation marks. Note: function uses eval, and is thus unsafe for untrusted sources.'''
        args = args.split(' ')
        for x in range(len(args)):
            if args[x].endswith(':'):
                modarg = args[x].substring(0, args[x].length() - 1)
                if modarg in model_args:
                    try:
                        eval(modarg) = eval(args[x + 1])
                        if modarg == 'parallel':
                            if num_workers == None:
                                num_workers = 1
                            elif num_workers == 1:
                                num_workers = None
                            pool = Pool(num_workers)
                    except IndexError:
                        print "No value provided for argument: {}".format(modarg)
                        continue
                else:
                    print "Argument not recognized: {}".format(modarg)
                    continue
            else:
                continue
