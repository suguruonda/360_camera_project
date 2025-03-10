import os
import sys
import torch
import pytorch3d.transforms
import torch.nn as nn

device = torch.device("cpu")
"""
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
"""    
class cb(torch.nn.Module):
    def __init__(self,r1,t1,cp1,p3d1,r2,t2,cp2,p3d2,mtx):
        super(cb, self).__init__()
        self.r1_p = nn.Parameter(torch.from_numpy(r1.reshape(8,-1))).to(device)
        self.t1_p = nn.Parameter(torch.from_numpy(t1.reshape(8,-1))).to(device)
        self.r2_p = nn.Parameter(torch.from_numpy(r2.reshape(8,-1))).to(device)
        self.t2_p = nn.Parameter(torch.from_numpy(t2.reshape(8,-1))).to(device)

        self.mtx = torch.from_numpy(mtx).to(torch.float64).to(device)
        self.mtx.requires_grad = False
        self.cp1 = [torch.from_numpy(i).to(torch.float64).to(device) for i in cp1]
        self.cp2 = [torch.from_numpy(i).to(torch.float64).to(device) for i in cp2]
        self.p3d1 = [torch.from_numpy(i).to(torch.float64).to(device) for i in p3d1]
        self.p3d2 = [torch.from_numpy(i).to(torch.float64).to(device) for i in p3d2]
        for i in range(len(self.cp1)):
            self.cp1[i].requires_grad = False
            self.cp2[i].requires_grad = False
            self.p3d1[i].requires_grad = False
            self.p3d2[i].requires_grad = False
        self.P = None

    def cal_relativePose(self):
        r1s = pytorch3d.transforms.axis_angle_to_matrix(self.r1_p)
        r2s = pytorch3d.transforms.axis_angle_to_matrix(self.r2_p)
        t1s = self.t1_p 
        t2s = self.t2_p 
        k = -1
        R8 = r1s[8 + k]
        R7 = r1s[7 + k]
        R5 = r1s[5 + k] @ torch.t(r2s[7 + k]) @ R7
        R4 = r1s[4 + k] @ torch.t(r2s[5 + k]) @ R5
        R2 = r1s[2 + k] @ torch.t(r2s[4 + k]) @ R4
        R1 = r1s[1 + k] @ torch.t(r2s[2 + k]) @ R2
        R3 = r1s[3 + k] @ torch.t(r2s[1 + k]) @ R1
        R6 = r1s[6 + k] @ torch.t(r2s[3 + k]) @ R3

        T8 = t1s[8 + k]
        T7 = t1s[7 + k]
        T5 = t1s[5 + k] + r1s[5 + k] @ torch.t(r2s[7 + k]) @ (T7 - t2s[7 + k])
        T4 = t1s[4 + k] + r1s[4 + k] @ torch.t(r2s[5 + k]) @ (T5 - t2s[5 + k])
        T2 = t1s[2 + k] + r1s[2 + k] @ torch.t(r2s[4 + k]) @ (T4 - t2s[4 + k])
        T1 = t1s[1 + k] + r1s[1 + k] @ torch.t(r2s[2 + k]) @ (T2 - t2s[2 + k])
        T3 = t1s[3 + k] + r1s[3 + k] @ torch.t(r2s[1 + k]) @ (T1 - t2s[1 + k])
        T6 = t1s[6 + k] + r1s[6 + k] @ torch.t(r2s[3 + k]) @ (T3 - t2s[3 + k])

        R_8 = r2s[8 + k]
        R_6 = r2s[6 + k]
        R_3 = r2s[3 + k] @ torch.t(r1s[6 + k]) @ R_6
        R_1 = r2s[1 + k] @ torch.t(r1s[3 + k]) @ R_3
        R_2 = r2s[2 + k] @ torch.t(r1s[1 + k]) @ R_1
        R_4 = r2s[4 + k] @ torch.t(r1s[2 + k]) @ R_2
        R_5 = r2s[5 + k] @ torch.t(r1s[4 + k]) @ R_4
        R_7 = r2s[7 + k] @ torch.t(r1s[5 + k]) @ R_5

        T_8 = t2s[8 + k]
        T_6 = t2s[6 + k]
        T_3 = t2s[3 + k] + r2s[3 + k] @ torch.t(r1s[6 + k]) @ (T_6 - t1s[6 + k])
        T_1 = t2s[1 + k] + r2s[1 + k] @ torch.t(r1s[3 + k]) @ (T_3 - t1s[3 + k])
        T_2 = t2s[2 + k] + r2s[2 + k] @ torch.t(r1s[1 + k]) @ (T_1 - t1s[1 + k])
        T_4 = t2s[4 + k] + r2s[4 + k] @ torch.t(r1s[2 + k]) @ (T_2 - t1s[2 + k])
        T_5 = t2s[5 + k] + r2s[5 + k] @ torch.t(r1s[4 + k]) @ (T_4 - t1s[4 + k])
        T_7 = t2s[7 + k] + r2s[7 + k] @ torch.t(r1s[5 + k]) @ (T_5 - t1s[5 + k])


        Rs = torch.cat((R1.view(1,3,3),R2.view(1,3,3),R3.view(1,3,3),R4.view(1,3,3),R5.view(1,3,3),R6.view(1,3,3),R7.view(1,3,3),R8.view(1,3,3)))
        R_s = torch.cat((R_1.view(1,3,3),R_2.view(1,3,3),R_3.view(1,3,3),R_4.view(1,3,3),R_5.view(1,3,3),R_6.view(1,3,3),R_7.view(1,3,3),R_8.view(1,3,3)))
        Ts = torch.cat((T1.view(1,3),T2.view(1,3),T3.view(1,3),T4.view(1,3),T5.view(1,3),T6.view(1,3),T7.view(1,3),T8.view(1,3)))
        T_s = torch.cat((T_1.view(1,3),T_2.view(1,3),T_3.view(1,3),T_4.view(1,3),T_5.view(1,3),T_6.view(1,3),T_7.view(1,3),T_8.view(1,3)))
        self.P = torch.cat((Rs,Ts.unsqueeze(2)),2)
        #self.P = torch.cat((R_s,T_s.unsqueeze(2)),2)

        return Rs, R_s, Ts, T_s

    def pose_loss(self, Rs, R_s, Ts, T_s):
        R_distance = (1.-pytorch3d.transforms.so3_relative_angle(Rs, R_s, cos_angle=True)).mean()
        T_distance = ((Ts - T_s)**2).sum(1).mean()
        return R_distance + T_distance

    def project(self, p, x, mtx):
        new_c = mtx.matmul(p).matmul(x.T)
        xp = new_c.T[:,0:2].clone()
        sp = new_c.T[:,2].clone()
        return xp/sp.unsqueeze(1)

    def reprojection_loss(self):
        criterion = nn.MSELoss()
        reprojection_loss = 0.0
        r1s = pytorch3d.transforms.axis_angle_to_matrix(self.r1_p)
        for i in range(len(self.cp1)): 
            p_pre = self.project(torch.cat((r1s[i],self.t1_p[i].unsqueeze(1)),1),self.p3d1[i],self.mtx[i])
            #loss = torch.sqrt(criterion(x, y))
            reprojection_loss += criterion(p_pre, self.cp1[i])
        return reprojection_loss/len(self.cp1)

    def total_loss(self):
        Rs, R_s, Ts, T_s = self.cal_relativePose()
        p_loss = self.pose_loss(Rs, R_s, Ts, T_s)
        rp_loss = self.reprojection_loss()
        return (p_loss, rp_loss)

def main(r1,t1,cp1,p3d1,r2,t2,cp2,p3d2,mtx):
    n_iter = 200  # fix the number of iterations
    c = cb(r1,t1,cp1,p3d1,r2,t2,cp2,p3d2,mtx)

    #Rs, R_s, Ts, T_s = c.cal_relativePose()


    optimizer = torch.optim.Adam(c.parameters(), lr=0.01)

    for it in range(n_iter):
        # re-init the optimizer gradients
        optimizer.zero_grad()
        loss = c.total_loss()
        loss_total = loss[0]+loss[1]
        loss_total.backward()
        optimizer.step()

        # plot and print status message
        if it % 10==0 or it==n_iter-1:
            status = 'iteration=%3d; Pose loss=%1.3e; Reprojection loss=%1.3e' % (it, loss[0],loss[1])
            print(status)

    print('Optimization finished.')
    with torch.no_grad():
        Rs, R_s, Ts, T_s = c.cal_relativePose()
        output = ((Rs.detach().to("cpu")).numpy(), (R_s.detach().to("cpu")).numpy(), (Ts.detach().to("cpu")).numpy(), (T_s.detach().to("cpu")).numpy())
    return output

