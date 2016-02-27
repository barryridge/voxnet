
import logging
import cPickle as pickle
import warnings
import numpy as np

from path import Path
import time

import lasagne

def save_weights(fname, l_out, metadata=None):
    """ assumes all params have unique names.
    """
    params = lasagne.layers.get_all_params(l_out)
    names = [par.name for par in params]
    if len(names) != len(set(names)):
        raise ValueError('need unique param names')
    param_dict = { param.name : param.get_value(borrow=False)
            for param in params }
    if metadata is not None:
        param_dict['metadata'] = pickle.dumps(metadata)
    logging.info('saving {} parameters to {}'.format(len(params), fname))
    # try to avoid half-written files
    fname = Path(fname)
    if fname.exists():
        file_in_use = True
        try_num = 0
        while file_in_use and try_num < 5:
            try:
                _ = np.load(fname)
                _.close()
                file_in_use = False
            except:
                try_num += 1
                time.sleep(try_num)
                logging.warning('file {} in use, waiting...{}'.format(fname,try_num))
                
    np.savez_compressed(str(fname), **param_dict)
    logging.info('weights saved to file {}'.format((fname,)))

def load_weights(fname, l_out):
    params = lasagne.layers.get_all_params(l_out)
    names = [ par.name for par in params ]
    if len(names)!=len(set(names)):
        raise ValueError('need unique param names')

    param_dict = np.load(fname)
    for param in params:
        if param.name in param_dict:
            stored_shape = np.asarray(param_dict[param.name].shape)
            param_shape = np.asarray(param.get_value().shape)
            if not np.all(stored_shape == param_shape):
                warn_msg = 'shape mismatch:'
                warn_msg += '{} stored:{} new:{}'.format(param.name, stored_shape, param_shape)
                warn_msg += ', skipping'
                warnings.warn(warn_msg)
            else:
                param.set_value(param_dict[param.name])
        else:
            logging.warn('unable to load parameter {} from {}'.format(param.name, weights_fname))
    if 'metadata' in param_dict:
        metadata = pickle.loads(str(param_dict['metadata']))
    else:
        metadata = {}
    return metadata
