# import droid_slam.geom.projective_ops as  pops
import torch

# def coords_grid(ht, wd, **kwargs):
#     y, x = torch.meshgrid(
#         torch.arange(ht).to(**kwargs).float(),
#         torch.arange(wd).to(**kwargs).float())
#
#     return torch.stack([x, y], dim=-1)
#
# coords0 = coords_grid(100, 100)[None,None]
# print(coords0.shape)
# a = torch.zeros([1, 0, 100, 100, 2])
# print(a.shape)
#
# c = torch.meshgrid(torch.arange(100),torch.arange(100))
# for i in c:
#     print(i.shape)
# import numpy as np
# c = np.meshgrid(range(100),range(100))
# c = np.asarray(c)
# print(c.shape)

ii, jj = torch.meshgrid(torch.arange(4),torch.arange(4))
ii = ii.reshape(-1)
jj = jj.reshape(-1)
# print(a)
# print(c)
#
# d = a.cat(c)
# print(d)

keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= 3)
print(keep)

iis = torch.as_tensor([])
print(iis.shape[0]+1)

c = torch.zeros_like(ii)
c[5:] = ~c[5:]
d = torch.argsort(c)
k = torch.sort(c)
print(c)
print(d)
print(k)

a = torch.as_tensor([[1,2,3,4],[1,2,3,4]])
fx,fy,cx,cy = a[...,None,None,:].unbind(dim=-1)
# print(a.shape)
print(fx.shape)

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)


def iproj(disps, intrinsics, jacobian=False):
    """ pinhole camera inverse projection """
    ht, wd = disps.shape[2:]
    # [n,1],[n,1],[n,1],[n,1]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)

    # y = wd个0~wd个ht
    # x = ht个(0~ht)
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[...,-1] = 1.0
        return pts, J

    return pts, None

a = torch.rand(1,10,100,100)
b = torch.rand(1,10,4)
iproj(a, b, jacobian=True)