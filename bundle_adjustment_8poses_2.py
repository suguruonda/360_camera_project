import os
import sys
import torch
import pytorch3d.transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")
"""
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
"""    
class pose8(torch.nn.Module):
    def __init__(self,stores_dict):
        super(pose8, self).__init__()
        self.r1_p = nn.Parameter(torch.from_numpy(np.array(stores_dict["R1"]).reshape(8,-1))).to(device)
        self.t1_p = nn.Parameter(torch.from_numpy(np.array(stores_dict["T1"]).reshape(8,-1))).to(device)
        self.r2_p = nn.Parameter(torch.from_numpy(np.array(stores_dict["R2"]).reshape(8,-1))).to(device)
        self.t2_p = nn.Parameter(torch.from_numpy(np.array(stores_dict["T2"]).reshape(8,-1))).to(device)
        self.r1_p.requires_grad = False
        self.t1_p.requires_grad = False
        self.r2_p.requires_grad = False
        self.t2_p.requires_grad = False
        
        self.mtx = torch.from_numpy(np.array(stores_dict["MTX"])).to(torch.float64).to(device)
        self.mtx.requires_grad = False
        self.cp1 = [torch.from_numpy(i).to(torch.float64).to(device) for i in stores_dict["CP1"]]
        self.cp2 = [torch.from_numpy(i).to(torch.float64).to(device) for i in stores_dict["CP2"]]
        self.p3d1 = [torch.from_numpy(i).to(torch.float64).to(device) for i in stores_dict["P3D1"]]
        self.p3d2 = [torch.from_numpy(i).to(torch.float64).to(device) for i in stores_dict["P3D1"]]
        for i in range(len(self.cp1)):
            self.cp1[i].requires_grad = False
            self.cp2[i].requires_grad = False
            self.p3d1[i].requires_grad = False
            self.p3d2[i].requires_grad = False

        
        r_dif_5,t_dif_5,r_dif_6,t_dif_6,r_dif_4,t_dif_4,r_dif_3,t_dif_3 = self.cal_initial_diffs()
        self.r_dif_5 = nn.Parameter(r_dif_5).to(device)
        self.t_dif_5 = nn.Parameter(t_dif_5).to(device)
        self.r_dif_6 = nn.Parameter(r_dif_6).to(device)
        self.t_dif_6 = nn.Parameter(t_dif_6).to(device)
        self.r_dif_4 = nn.Parameter(r_dif_4).to(device)
        self.t_dif_4 = nn.Parameter(t_dif_4).to(device)
        self.r_dif_3 = nn.Parameter(r_dif_3).to(device)
        self.t_dif_3 = nn.Parameter(t_dif_3).to(device)
        


        self.P = None

    def cal_initial_diffs(self):
        #Rodrigues conversion 1*3 to 3*3 rotation matrix
        r1s = pytorch3d.transforms.axis_angle_to_matrix(self.r1_p)
        r2s = pytorch3d.transforms.axis_angle_to_matrix(self.r2_p)
        t1s = self.t1_p 
        t2s = self.t2_p 
        
        k = -1
        R6 = r2s[6 + k]
        T6 = t2s[6 + k]
        R5 = r2s[5 + k]
        T5 = t2s[5 + k]
        R4 = r1s[4 + k]
        T4 = t1s[4 + k]
        R3 = r1s[3 + k]
        T3 = t1s[3 + k]

        r_dif_6 = torch.t(r1s[6 + k]) @ R6
        t_dif_6 = torch.t(r1s[6 + k]) @ (T6 - t1s[6 + k])
        r_dif_5 = torch.t(r1s[5 + k]) @ R5
        t_dif_5 = torch.t(r1s[5 + k]) @ (T5 - t1s[5 + k])

        R8, T8 = self.composePose(r_dif_5,t_dif_5,r2s[8 + k],t2s[8 + k])
        R7, T7 = self.composePose(r_dif_5,t_dif_5,r1s[7 + k],t1s[7 + k])
        
        r_dif_3 = torch.t(r2s[3 + k]) @ R3
        t_dif_3 = torch.t(r2s[3 + k]) @ (T3 - t2s[3 + k])
        r_dif_4 = torch.t(r2s[4 + k]) @ R4
        t_dif_4 = torch.t(r2s[4 + k]) @ (T4 - t2s[4 + k])


        R2, T2 = self.composePose(r_dif_4,t_dif_4,r1s[2 + k],t1s[2 + k])
        R1, T1 = self.composePose(r_dif_4,t_dif_4,r1s[1 + k],t1s[1 + k])

        #Rodrigues conversion 3*3 to 3*1 rotation matrix we try to make parameter simple(fewer). 
        r_dif_5 = pytorch3d.transforms.matrix_to_axis_angle(r_dif_5)
        r_dif_6 = pytorch3d.transforms.matrix_to_axis_angle(r_dif_6)
        r_dif_4 = pytorch3d.transforms.matrix_to_axis_angle(r_dif_4)
        r_dif_3 = pytorch3d.transforms.matrix_to_axis_angle(r_dif_3)

        return r_dif_5,t_dif_5,r_dif_6,t_dif_6,r_dif_4,t_dif_4,r_dif_3,t_dif_3
        #return r_dif_2, t_dif_2, r_dif_3, t_dif_3

    def update_poses_test(self):
        #Rodrigues conversion 3*1 to 3*3 rotation matrix
        r1s = pytorch3d.transforms.axis_angle_to_matrix(self.r1_p)
        r2s = pytorch3d.transforms.axis_angle_to_matrix(self.r2_p)
        t1s = self.t1_p 
        t2s = self.t2_p 
        
        k = -1

        R6 = r2s[6 + k]
        T6 = t2s[6 + k]
        R5 = r2s[5 + k]
        T5 = t2s[5 + k]
        R4 = r1s[4 + k]
        T4 = t1s[4 + k]
        R3 = r1s[3 + k]
        T3 = t1s[3 + k]

        r_dif_6 = torch.t(r1s[6 + k]) @ R6
        t_dif_6 = torch.t(r1s[6 + k]) @ (T6 - t1s[6 + k])
        r_dif_5 = torch.t(r1s[5 + k]) @ R5
        t_dif_5 = torch.t(r1s[5 + k]) @ (T5 - t1s[5 + k])

        R8, T8 = self.composePose(r_dif_5,t_dif_5,r2s[8 + k],t2s[8 + k])
        R7, T7 = self.composePose(r_dif_5,t_dif_5,r1s[7 + k],t1s[7 + k])

        #R8, T8 = self.composePose(r_dif_6,t_dif_6,r2s[8 + k],t2s[8 + k])
        #R7, T7 = self.composePose(r_dif_6,t_dif_6,r1s[7 + k],t1s[7 + k])
        
        r_dif_3 = torch.t(r2s[3 + k]) @ R3
        t_dif_3 = torch.t(r2s[3 + k]) @ (T3 - t2s[3 + k])
        r_dif_4 = torch.t(r2s[4 + k]) @ R4
        t_dif_4 = torch.t(r2s[4 + k]) @ (T4 - t2s[4 + k])


        R2, T2 = self.composePose(r_dif_4,t_dif_4,r1s[2 + k],t1s[2 + k])
        R1, T1 = self.composePose(r_dif_4,t_dif_4,r1s[1 + k],t1s[1 + k])
        #R2, T2 = self.composePose(r_dif_3,t_dif_3,r1s[2 + k],t1s[2 + k])
        #R1, T1 = self.composePose(r_dif_3,t_dif_3,r1s[1 + k],t1s[1 + k])
        
        Rs = torch.cat((R1.view(1,3,3),R2.view(1,3,3),R3.view(1,3,3),R4.view(1,3,3),R5.view(1,3,3),R6.view(1,3,3),R7.view(1,3,3),R8.view(1,3,3)))
        #R_s = torch.cat((R_1.view(1,3,3),R_2.view(1,3,3),R_3.view(1,3,3),R_4.view(1,3,3),R_5.view(1,3,3),R_6.view(1,3,3),R_7.view(1,3,3),R_8.view(1,3,3)))
        Ts = torch.cat((T1.view(1,3),T2.view(1,3),T3.view(1,3),T4.view(1,3),T5.view(1,3),T6.view(1,3),T7.view(1,3),T8.view(1,3)))
        #T_s = torch.cat((T_1.view(1,3),T_2.view(1,3),T_3.view(1,3),T_4.view(1,3),T_5.view(1,3),T_6.view(1,3),T_7.view(1,3),T_8.view(1,3)))
        self.P = torch.cat((Rs,Ts.unsqueeze(2)),2)
        #self.P = torch.cat((R_s,T_s.unsqueeze(2)),2)

        return Rs, Ts

    def move_to_end(self,test_list):
        first_ele = test_list[0]
        test_list.pop(0)
        test_list.append(first_ele)
        return test_list

    def cal_relativePose(self, R1,T1,R2,T2):
        R_dif = R2 @ torch.t(R1)
        T_dif = T2 - R2 @ torch.t(R1) @ T1
        return R_dif,T_dif

    def composePose(self, R1,T1,R2,T2):
        R_new = R2 @ R1
        T_new = T2 + R2 @ T1
        return R_new,T_new

    def update_poses(self):
        #Rodrigues conversion 3*1 to 3*3 rotation matrix
        r1s = pytorch3d.transforms.axis_angle_to_matrix(self.r1_p)
        r2s = pytorch3d.transforms.axis_angle_to_matrix(self.r2_p)
        t1s = self.t1_p 
        t2s = self.t2_p 
        
        k = -1
        R8 = r1s[8 + k]
        T8 = t1s[8 + k]
        r_dif_8 = torch.t(r2s[8 + k]) @ R8
        t_dif_8 = torch.t(r2s[8 + k]) @ (T8 - t2s[8 + k])

        R7 = r1s[7 + k] @ r_dif_8
        T7 = t1s[7 + k] + r1s[7 + k] @ t_dif_8
        R6 = r1s[6 + k] @ r_dif_8
        T6 = t1s[6 + k] + r1s[6 + k] @ t_dif_8
        R5 = r1s[5 + k] @ r_dif_8
        T5 = t1s[5 + k] + r1s[5 + k] @ t_dif_8
        
        #r_dif_5,t_dif_5,r_dif_6,t_dif_6,r_dif_4,t_dif_4,r_dif_3,t_dif_3 = self.cal_initial_diffs()
        
        r_dif_5 = pytorch3d.transforms.axis_angle_to_matrix(self.r_dif_5) 
        #r_dif_5 = self.r_dif_5 
        t_dif_5 = self.t_dif_5 
        R4 = r1s[4 + k] @ r_dif_5
        T4 = t1s[4 + k] + r1s[4 + k] @ t_dif_5
        R3 = r1s[3 + k] @ r_dif_5
        T3 = t1s[3 + k] + r1s[3 + k] @ t_dif_5

        r_dif_4 = pytorch3d.transforms.axis_angle_to_matrix(self.r_dif_4) 
        #r_dif_4 = self.r_dif_4
        t_dif_4 = self.t_dif_4 
        R2 = r1s[2 + k] @ r_dif_4
        T2 = t1s[2 + k] + r1s[2 + k] @ t_dif_4
        R1 = r1s[1 + k] @ r_dif_4
        T1 = t1s[1 + k] + r1s[1 + k] @ t_dif_4

        r_dif_6 = pytorch3d.transforms.axis_angle_to_matrix(self.r_dif_6) 
        #r_dif_6 = self.r_dif_6 
        t_dif_6 = self.t_dif_6 
        R_4 = r1s[4 + k] @ r_dif_6
        T_4 = t1s[4 + k] + r1s[4 + k] @ t_dif_6
        R_3 = r1s[3 + k] @ r_dif_6
        T_3 = t1s[3 + k] + r1s[3 + k] @ t_dif_6

        r_dif_3 = pytorch3d.transforms.axis_angle_to_matrix(self.r_dif_3) 
        #r_dif_3 = self.r_dif_3
        t_dif_3 = self.t_dif_3 
        R_2 = r1s[2 + k] @ r_dif_3
        T_2 = t1s[2 + k] + r1s[2 + k] @ t_dif_3
        R_1 = r1s[1 + k] @ r_dif_3
        T_1 = t1s[1 + k] + r1s[1 + k] @ t_dif_3

        R_8 = R8
        T_8 = T8
        R_7 = R7
        T_7 = T7
        R_6 = R6
        T_6 = T6
        R_5 = R5
        T_5 = T5
        
        Rs = torch.cat((R1.view(1,3,3),R2.view(1,3,3),R3.view(1,3,3),R4.view(1,3,3),R5.view(1,3,3),R6.view(1,3,3),R7.view(1,3,3),R8.view(1,3,3)))
        R_s = torch.cat((R_1.view(1,3,3),R_2.view(1,3,3),R_3.view(1,3,3),R_4.view(1,3,3),R_5.view(1,3,3),R_6.view(1,3,3),R_7.view(1,3,3),R_8.view(1,3,3)))
        Ts = torch.cat((T1.view(1,3),T2.view(1,3),T3.view(1,3),T4.view(1,3),T5.view(1,3),T6.view(1,3),T7.view(1,3),T8.view(1,3)))
        T_s = torch.cat((T_1.view(1,3),T_2.view(1,3),T_3.view(1,3),T_4.view(1,3),T_5.view(1,3),T_6.view(1,3),T_7.view(1,3),T_8.view(1,3)))
        self.P = torch.cat((Rs,Ts.unsqueeze(2)),2)
        #self.P = torch.cat((R_s,T_s.unsqueeze(2)),2)

        return Rs, R_s, Ts, T_s

    def pose_loss(self, Rs, R_s, Ts, T_s):
        R_distance = torch.abs(1.-pytorch3d.transforms.so3_relative_angle(Rs, R_s, cos_angle=True)).mean()
        T_distance = (torch.abs(Ts - T_s)).sum(1).mean()
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
        Rs, R_s, Ts, T_s = self.update_poses()
        p_loss = self.pose_loss(Rs, R_s, Ts, T_s)
        rp_loss = self.reprojection_loss()
        return (p_loss, rp_loss)

def optimize(stores_dict):
    n_iter = 3000  # fix the number of iterations
    c = pose8(stores_dict)

    #Rs, R_s, Ts, T_s = c.update_poses()


    #optimizer = torch.optim.Adam(c.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(c.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
    # Lists to store losses for plotting
    pose_losses = []
    reprojection_losses = []

    for it in range(n_iter):
        # re-init the optimizer gradients
        optimizer.zero_grad()
        loss = c.total_loss()
        loss_total = loss[0]+loss[1]/10
        #loss_total = loss[0]

        # Store losses for plotting
        pose_losses.append(loss[0].item())
        reprojection_losses.append(loss[1].item())

        loss_total.backward()
        optimizer.step()

        # plot and print status message
        if it % 10==0 or it==n_iter-1:
            status = 'iteration=%3d; Pose loss=%1.3e; Reprojection loss=%1.3e' % (it, loss[0],loss[1])
            scheduler.step(loss_total)
            print(status)
    
    # Plotting the losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(pose_losses, label='Pose Loss')
    plt.title('Pose Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(reprojection_losses, label='Reprojection Loss')
    plt.title('Reprojection Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.tight_layout()

    # save figure
    plt.savefig('losses.png')


    print('Optimization finished.')
    with torch.no_grad():
        Rs, R_s, Ts, T_s = c.update_poses()
        output = ((Rs.detach().to("cpu")).numpy(), (R_s.detach().to("cpu")).numpy(), (Ts.detach().to("cpu")).numpy(), (T_s.detach().to("cpu")).numpy())
    return output

def optimize2(stores_dict):
    n_iter = 3000  # fix the number of iterations
    c = pose8(stores_dict)

    with torch.no_grad():
        Rs, Ts= c.update_poses_test()
        output = ((Rs.detach().to("cpu")).numpy(), (Rs.detach().to("cpu")).numpy(), (Ts.detach().to("cpu")).numpy(), (Ts.detach().to("cpu")).numpy())
    return output