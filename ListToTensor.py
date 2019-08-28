import numpy as np
import torch as t
def ListToTensor(lister):
    nparr = np.asarray(lister)
    tens = t.from_numpy(nparr)
    return tens
