''' Utilities '''
import torch

def A_(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    # # for 3-D measurements only
    # return np.sum(x*Phi, axis=2)  # element-wise product
    # for general N-D measurements
    return torch.sum(x*Phi, axis=tuple(range(2,Phi.ndim)))  # element-wise product

def At_(y, Phi):
    '''
    Tanspose of the forward model. 
    '''
    # (nrow, ncol, nmask) = Phi.shape
    # x = np.zeros((nrow, ncol, nmask))
    # for nt in range(nmask):
    #     x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
    # return x
    # # for 3-D measurements only
    # return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)
    # for general N-D measurements (original Phi: H x W (x C) x F, y: H x W)
    # [expected] direct broadcasting (Phi: F x C x H x W, y: H x W)
    # [todo] change the dimension order to follow NumPy convention
    # D = Phi.ndim
    # ax = tuple(range(2,D))
    # return np.multiply(np.repeat(np.expand_dims(y,axis=ax),Phi.shape[2:D],axis=ax), Phi) # inefficient considering the memory layout https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
    return torch.t(torch.t(Phi)*torch.t(y)) #(Phi.transpose()*y.transpose()).transpose() # broadcasted by numpy

def psnr(ref, img):
    '''
    Peak signal-to-noise ratio (PSNR).
    '''
    mse = torch.mean( (ref - img) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def denoise_tv(image, weight, max_num_iter=5):
    device = torch.device('cuda')
    ndim = image.ndim
    p = torch.zeros((image.ndim, ) + image.shape, dtype=image.dtype).to(device)
    g = torch.zeros_like(p)
    d = torch.zeros_like(image)
    i = 0
    while i < max_num_iter:
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            slices_d = [slice(None), ] * ndim
            slices_p = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax+1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] = d[tuple(slices_d)] + p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax+1] = slice(None)
            out = image + d
        else:
            out = image
        # E = (d ** 2).sum()

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        slices_g = [slice(None), ] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax+1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = torch.diff(out, axis=ax)
            slices_g[ax+1] = slice(None)

        # norm = torch.sqrt((g ** 2).sum(axis=0))[np.newaxis, ...]
        norm = torch.sqrt((g ** 2).sum(axis=0)).unsqueeze(0).to(device)
        # E = E + weight * norm.sum()
        tau = 1. / (2.*ndim)
        norm = norm * tau / weight
        norm = norm + 1.
        p = p - tau * g
        p = p / norm
        # E = E / image.numel()
        # if i == 0:
        #     E_init = E
        #     E_previous = E
        # else:
        #     if torch.abs(E_previous - E) < eps * E_init:
        #         break
        #     else:
        #         E_previous = E
        i = i + 1
    return out