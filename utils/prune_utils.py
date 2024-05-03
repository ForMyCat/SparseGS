import numpy as np
import torch
from scipy import cluster
import torchmetrics.functional as TMF
from scipy.signal import argrelmin, find_peaks
from scipy import ndimage as ndi

## GHT Code from "A Generalization of Otsu’s Method and Minimum Error Thresholding" Jon Barron ECCV 2020,https://arxiv.org/pdf/2007.07350.pdf

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)


def blur(img, size=2):
        return ndi.gaussian_filter(img, sigma=size)

def preliminaries(n, x):
  """Some math that is shared across multiple algorithms."""
  assert np.all(n >= 0)
  x = np.arange(len(n), dtype=n.dtype) if x is None else x
  assert np.all(x[1:] >= x[:-1])
  w0 = clip(csum(n))
  w1 = clip(dsum(n))
  p0 = w0 / (w0 + w1)
  p1 = w1 / (w0 + w1)
  mu0 = csum(n * x) / w0
  mu1 = dsum(n * x) / w1
  d0 = csum(n * x**2) - w0 * mu0**2
  d1 = dsum(n * x**2) - w1 * mu1**2
  return x, w0, w1, p0, p1, mu0, mu1, d0, d1

def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5, prelim=None):
  assert nu >= 0
  assert tau >= 0
  assert kappa >= 0
  assert omega >= 0 and omega <= 1
  x, w0, w1, p0, p1, _, _, d0, d1 = prelim or preliminaries(n, x)
  v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
  v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
  f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
  f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
  return argmax(x, f0 + f1), f0 + f1

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def get_hist(mode_img):
    counts, bins = np.histogram(mode_img, bins=1000)

    return counts, bins

def kl_d(a, b):
    a = torch.from_numpy(a).unsqueeze(0)
    b = torch.from_numpy(b).unsqueeze(0)
    val = TMF.kl_divergence(a, b)
    return val.item()

def ght_test(img):
    counts, bins = get_hist(img)

    default_nu = np.sum(counts)
    default_tau = np.sqrt(1/12)
    default_kappa = np.sum(counts)
    default_omega = 0.5

    prelim = preliminaries(counts, bins[:-1])

    t, score = GHT(counts, bins[:-1], nu=default_nu, tau=default_tau, kappa=default_kappa, omega=default_omega, prelim=prelim)

    return t, score

def recursive_ght(img, levels=5):
    t, score = ght_test(img)

    ts = [t]

    if levels == 0:
        return ts

    img0 = img[img < t]
    img1 = img[img > t]

    ts.extend(recursive_ght(img0, levels-1))
    ts.extend(recursive_ght(img1, levels-1))

    return ts

def dist_matching(gt, mode):
    mode_img = normalize(mode)

    c, b = get_hist(mode_img)

    gc, gb = get_hist(gt)

    kl_min = np.inf
    kl_min_idx = -1

    kl = []

    for k in range(len(c)-1):
        trunc_dist = c[k:]

        if len(trunc_dist) < 20:
            continue

        hist, _ = np.histogram(normalize(trunc_dist), bins=len(gc))

        kld = kl_d(hist, gc)

        kl.append(kld)
        
        if kld < kl_min:
            kl_min = kld
            kl_min_idx = k

        kl.append(kld)

    try:
        kl_first_peak = find_peaks(-1 * np.array(kl))[0][0]
    except:
        kl_first_peak = len(b)

    return kl_first_peak / len(b), kl_min_idx / len(b)

# def identify_floaters_old(depth_img, psuedo_gt_img, thresh_bin=0.13):
#     depth_img = normalize(depth_img.cpu().numpy())
#     psuedo_gt_img = psuedo_gt_img.cpu().numpy()

#     counts, bins = np.histogram(depth_img, bins=1000)

#     first_peak, min_idx = dist_matching(psuedo_gt_img, depth_img)
#     t = np.min(recursive_ght(depth_img))

#     choice_arr = np.array([first_peak, min_idx, t])

#     thresh_bin = (np.where(np.cumsum(counts) / np.sum(counts) >= thresh_bin)[0][0]) / len(bins)

#     valid_mask = choice_arr < thresh_bin

#     cut_off = thresh_bin
#     if np.count_nonzero(valid_mask) > 0:
#         cut_off = np.max(choice_arr[valid_mask])

#     mask = depth_img < cut_off

#     filtered_depth = np.zeros_like(depth_img)
#     filtered_depth[mask] = 1

#     return torch.from_numpy(filtered_depth)

def calc_diff(mode_img, depth_img):
    mean = torch.mean(depth_img)
    std = torch.std(depth_img)
    diff = (depth_img - mode_img) / (depth_img + max(depth_img.min(), mean - 3*std))
    return diff



