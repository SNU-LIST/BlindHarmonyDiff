import numpy as np
import pickle
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import scipy.fftpack as fftpack
from PIL import Image


def load_pkls( path,nmax = -1,nmin = 0):
    assert os.path.isfile(path), path
    images = []
    with open(path, "rb") as f:
        images += pickle.load(f)
    assert len(images) > 0, path

    if nmin > 0 or nmax > 0:
        images = images[nmin:nmax]
    else:
        images = images[:]
    return images



def create_all_dirs(path):
    if "." in path.split("/")[-1]:
        dirs = os.path.dirname(path)
    else:
        dirs = path
    os.makedirs(dirs, exist_ok=True)
    
def to_pklv4(obj, path, vebose=False):
    create_all_dirs(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    if vebose:
        print("Wrote {}".format(path))
        
def hist_match(source, t_values,t_quantiles):

    oldshape = source.shape
    source = source.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    _, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]


    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def metric_corr_update(har_image,source_image,alpha,num):
    source_image = source_image - source_image.mean()
    har_image_upd = har_image
    for iter in range(0,num):
        har_image_upd_mean = har_image_upd.mean()
        har_image_upd = har_image_upd- har_image_upd_mean
        grad = -source_image/(1e-6+np.sqrt(np.mean(har_image_upd**2)))/(1e-6+np.sqrt(np.mean(source_image**2))) \
            + np.mean(har_image_upd*source_image)*har_image_upd/(1e-6+np.sqrt(np.mean(har_image_upd**2))**3)/(1e-6+np.sqrt(np.mean(source_image**2)))      

        har_image_upd = har_image_upd - alpha*grad + har_image_upd_mean

    return har_image_upd


pckl_tr = load_pkls('/fast_storage/hwihun/pkl_blindharmony/pkl_image_35177_val.pklv4',100,0)
print(len(pckl_tr))
ref = np.zeros([144, 208, 92, len(pckl_tr)])
for i in range(0,len(pckl_tr)):
    ref[:,:,:,i] = pckl_tr[i]
        
alpha = 0.005
Rval = 6

template = ref.ravel()
t_values, t_counts = np.unique(template, return_counts=True)
t_quantiles = np.cumsum(t_counts).astype(np.float64)
t_quantiles /= t_quantiles[-1]

sources =  ["0" ,  "51010", "21926",  "175614"]
for name in sources:
    pckl_sc = load_pkls(f'/fast_storage/hwihun/pkl_blindharmony/pkl_{name}_registered_{name}_test.pklv4',-1,0)
    pckl_har = load_pkls(f'/home/hwihun/blindharmony_diff/RF_winsor/results_pckl_3d_bfrefine/pkl_{name}_registered_{name}_test.pklv4',-1,0)

    img_har = []
    for ind_sub in tqdm(range(len(pckl_har))):
        har_image = pckl_har[ind_sub]
        img_sc = pckl_sc[ind_sub]
        har_image_update  = metric_corr_update(har_image,img_sc,alpha,Rval)
        img_har.append(har_image_update)
    to_pklv4(img_har, f'/home/hwihun/blindharmony_diff/RF_winsor/results_pckl_3d_registered/BHD_corr_wsc_refine_{name}_alpha_{alpha}_R_{Rval}.pklv4', vebose=True)

    
    pckl_sc = load_pkls(f'/fast_storage/hwihun/pkl_blindharmony/pkl_image_{name}_test.pklv4',-1,0)
    pckl_har = load_pkls(f'/home/hwihun/blindharmony_diff/RF_winsor/results_pckl_3d_bfrefine/pkl_image_{name}_test.pklv4',-1,0)

    img_har = []
    for ind_sub in tqdm(range(len(pckl_har))):
        har_image = pckl_har[ind_sub]
        img_sc = pckl_sc[ind_sub]
        har_image_update  = metric_corr_update(har_image,img_sc,alpha,Rval)


        img_har.append(har_image_update)
    to_pklv4(img_har, f'/home/hwihun/blindharmony_diff/RF_winsor/results_pckl_3d_downstream/BHD_corr_wsc_refine_{name}_alpha_{alpha}_R_{Rval}.pklv4', vebose=True)


