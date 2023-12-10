import numpy as np
from sklearn import decomposition
from skimage import io
import glob

def pca_compare(dpath, new_path, n_comp = None, dtype = "txt", rtype = "close"):
    """pca_analysis

    Args:
        path (str): path to either folder or the specific data file
        new_path (str): path to "missing" thing we need to compare with other data 
        n_comp (int): number of components to use if exercise specifies
        dtype (str, optional): whether the data is given in txt or images. Defaults to "txt".
        rtype (str, optional): whether to return which data is closest or furthest from the missing

    Raises:
        SyntaxError: provide either txt or image

    Returns:
        the index that is either closest or furthest from the missing data
    """
    
    if dtype == "txt":
        data = np.loadtxt(dpath, comments="%")
        new_data = np.loadtxt(new_path, comments = "%")
    elif dtype == "image":
        images = glob.glob(dpath + "*")
        if len(io.imread(images[0]).shape) == 3:
            w, h, r = io.imread(images[0]).shape
            data = np.zeros((len(images), w * h * r))
        else:
            w, h = io.imread(images[0]).shape
            data = np.zeros((len(images), w * h))
            
        for idx, image in enumerate(images):
            img_data = io.imread(image)
            data[idx] = img_data.flatten()
        
        new_data = io.imread(new_path).flatten().reshape(1, -1)
    else:
        raise SyntaxError("You need to specify how the data should be read")

    pca = decomposition.PCA(n_comp)
    pca.fit(data)
    
    pc_proj = pca.transform(data)
    pc_new = pca.transform(new_data).flatten()
    
    if rtype == "close":
        # NOTE!!!! glob changes the order of the indices, so have a look at the glob indices!!
        # if you need the order of images of any sort
        return np.linalg.norm(pc_proj - pc_new, axis = 1).argmin()
    else:
        return np.linalg.norm(pc_proj - pc_new, axis = 1).argmax()