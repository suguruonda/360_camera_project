import os
import sys
import torch
#import pytorch3d
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    

def project(p,x,m):
    new_c = p.matmul(x) 
    xp = new_c[:,0,:].clone()
    yp = new_c[:,1,:].clone()
    sp = new_c[:,2,:].clone()
    return ((xp/sp)*m, (yp/sp)*m)

def loss(a,b):
    #pd = torch.sqrt(torch.pow((a[0] - b[0]),2) + torch.pow((a[1] - b[1]),2))
    #pd = torch.pow((a[0] - b[0]),2) + torch.pow((a[1] - b[1]),2)
    pd = torch.abs(a[0] - b[0]) + torch.abs(a[1] - b[1])
    return torch.sum(pd)
    

def main(coordinates,mask,p_s,X):
    n_iter = 50000  # fix the number of iterations
    c = torch.from_numpy(coordinates)
    m = torch.from_numpy(mask)
    p = torch.from_numpy(p_s)
    x = torch.from_numpy(X)
    p_param = p.clone().detach()
    x_param = x.clone().detach()

    c.to(device)
    p_param.to(device)
    m.to(device)
    x_param.to(device)

    c.requires_grad = False
    p_param.requires_grad = True
    m.requires_grad = False
    x_param.requires_grad = True

    optimizer = torch.optim.Adam([x_param,p_param], lr=0.01)

    for it in range(n_iter):
        # re-init the optimizer gradients
        optimizer.zero_grad()

        pred = project(p_param,x_param,m)
        res = loss(pred,(c[:,:,0].T,c[:,:,1].T))
        res.backward(retain_graph=True)
        optimizer.step()

        # plot and print status message
        if it % 10==0 or it==n_iter-1:
            status = 'iteration=%3d; loss=%1.3e' % (it, res)
            print(status)

    print('Optimization finished.')
    output = x_param.T[:,0:3].detach() / x_param.T[:,3].detach().view(-1,1)
    return (output.to("cpu")).numpy()

def main2(coordinates,mask,p_s,X):
    n_iter = 10000  # fix the number of iterations
    c = torch.from_numpy(coordinates)
    m = torch.from_numpy(mask)
    p = torch.from_numpy(p_s)
    x = torch.from_numpy(X)
    p_param = p.clone().detach()
    x_param = x.clone().detach()

    c.to(device)
    p_param.to(device)
    m.to(device)
    x_param.to(device)

    c.requires_grad = False
    p_param.requires_grad = True
    m.requires_grad = False
    x_param.requires_grad = False

    optimizer = torch.optim.Adam([p_param], lr=0.01)

    for it in range(n_iter):
        # re-init the optimizer gradients
        optimizer.zero_grad()

        pred = project(p_param,x_param,m)
        res = loss(pred,(c[:,:,0].T,c[:,:,1].T))
        res.backward(retain_graph=True)
        optimizer.step()

        # plot and print status message
        if it % 10==0 or it==n_iter-1:
            status = 'iteration=%3d; loss=%1.3e' % (it, res)
            print(status)

    print('Optimization finished.')
    return (p_param.to("cpu")).numpy()