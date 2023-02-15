import torch
from scipy.special import sph_harm, lpmn, lpmv
from scipy.special import factorial
import numpy as np
import math
import time


class SphericalHarm(object):
    def __init__(self, total_deg):
        self.total_deg = total_deg
        self.orderIds, self.lIds, self.mIds, self.num_at_deg, self.m0inorder, self.restinorder, self.orderinorg = self.genalpids(
            self.total_deg)
        self.sh_ordermIds, self.sh_orderlIds, self.sh_orderIds, self.sh_orderinorg = self.genshids(
            self.total_deg)
        self.f2m_1, self.Klm = self.precompff(self.total_deg)
        self.orderKlm = self.Klm[self.orderIds]

    # def sh_all(self, theta, phi):
    #     phi = phi.view(-1, 1)
    #     theta = theta.view(-1, 1)
    def sh_all(self, indirs):
        indirs = indirs.view(-1, 3)
        theta = torch.acos(indirs[:, [2]])
        phi = torch.atan2(indirs[:, [1]], indirs[:, [0]])
        # phi = phi.view(-1, 1)
        # theta = theta.view(-1, 1)
        alp = self.associated_lengedre_poly_all(torch.cos(theta))

        m0 = alp[:, self.m0inorder] * torch.from_numpy(
            self.orderKlm[self.m0inorder]).to(theta.device).type(theta.dtype)
        # print("alp", alp[:, self.m0inorder], self.orderKlm[self.m0inorder])

        ms = torch.from_numpy(self.mIds[self.orderIds][self.restinorder]).to(
            theta.device).type(theta.dtype)

        restKlm = torch.from_numpy(self.orderKlm[self.restinorder]).to(
            theta.device).type(theta.dtype)

        m0p = restKlm * torch.cos(ms * phi) * alp[:, self.restinorder]
        m0n = restKlm * torch.sin(ms * phi) * alp[:, self.restinorder]
        # print(phi.shape, m0p.shape)

        m = torch.cat([m0, m0p, m0n], 1)
        m = m[:, self.sh_orderinorg]
        return m

    def associated_lengedre_poly_all(self, x):
        x = x.view(-1, 1)
        l = self.total_deg
        # alp = torch.ones((x.shape[0], l * (l + 1) // 2), device=x.device)

        ms = self.mIds[self.orderIds[:l]]
        somx2 = torch.sqrt((1 - x) * (1 + x))
        f2m_1s = torch.from_numpy(self.f2m_1[ms]).to(x.device).type(x.dtype)
        pmm = torch.pow(-somx2, torch.from_numpy(ms).to(x.device)) * f2m_1s
        alp = [pmm]
        t = l - 1
        if t > 0:
            ms = self.mIds[self.orderIds[self.total_deg:self.total_deg + t]]
            ms = torch.from_numpy(ms).to(x.device)

            pmp1m = x * (2 * ms + 1) * pmm[:, :t]
            alp.append(pmp1m)
            cur = self.total_deg + t
            for i in range(l - 2):
                t = l - 2 - i
                ms = self.mIds[self.orderIds[cur:cur + t]]
                ms = torch.from_numpy(ms).to(x.device)
                ls = ms + i + 2
                plm = (x * (2 * ls - 1) * pmp1m[:, :t] -
                       (ls + ms - 1) * pmm[:, :t]) / (i + 2)
                alp.append(plm)
                pmm = pmp1m
                pmp1m = plm
                cur += t
        alp = torch.cat(alp, 1)
        return alp

    def precompff(self, l):
        f2m_1 = np.arange(l) + 1
        f2m_1 = f2m_1 * 2 - 1
        f2m_1 = np.cumprod(f2m_1)
        f2m_1[1:] = f2m_1[:-1]

        Klm = np.sqrt((2 * self.lIds + 1) * factorial(self.lIds - self.mIds) /
                      (4 * np.pi * factorial(self.lIds + self.mIds)))
        m_n0 = np.reshape(np.where(self.mIds), -1)
        Klm[m_n0] *= 2**0.5

        return f2m_1, Klm

    def genalpids(self, l):
        r_orderIds = np.zeros(l * (l + 1) // 2, dtype=int)
        n_per_deg = np.arange(l + 1)[1:]
        num_deg = np.cumsum(n_per_deg)
        i_order = num_deg - 1
        k = 0
        for i in range(l):
            r_orderIds[k:k + len(i_order)] = i_order
            k += len(i_order)
            i_order = i_order[:-1] + n_per_deg[i:-1]

        r_lids = np.zeros(l * (l + 1) // 2, dtype=int)
        r_mids = np.zeros(l * (l + 1) // 2, dtype=int)
        k = 0
        for i in range(l):
            r_lids[k:k + i + 1] = i
            r_mids[k:k + i + 1] = np.arange(i + 1)
            k += i + 1

        r_m0inorder = [0] + list(range(l, 0, -1))
        r_m0inorder = np.cumsum(np.asarray(r_m0inorder, dtype=int))[:l]

        tmp = np.ones_like(r_orderIds)
        tmp[r_m0inorder] = 0
        r_restinorder = np.reshape(np.where(tmp), -1)
        tmp = np.arange(len(r_orderIds))
        r_orderinorg = tmp.copy()
        r_orderinorg[r_orderIds] = tmp[:]

        return r_orderIds, r_lids, r_mids, num_deg, r_m0inorder, r_restinorder, r_orderinorg

    def genshids(self, l):
        sh_ordermIds = np.zeros(l * l, dtype=int)
        sh_orderlIds = np.zeros(l * l, dtype=int)
        sh_orderIds = np.zeros(l * l, dtype=int)

        sh_ordermIds[:len(self.m0inorder)] = self.mIds[self.orderIds][
            self.m0inorder]
        sh_orderlIds[:len(self.m0inorder)] = self.lIds[self.orderIds][
            self.m0inorder]
        k = len(self.m0inorder)
        sh_ordermIds[k:k + len(self.restinorder)] = self.mIds[self.orderIds][
            self.restinorder]
        sh_orderlIds[k:k + len(self.restinorder)] = self.lIds[self.orderIds][
            self.restinorder]
        k += len(self.restinorder)
        sh_ordermIds[k:k + len(self.restinorder)] = -self.mIds[self.orderIds][
            self.restinorder]
        sh_orderlIds[k:k + len(self.restinorder)] = self.lIds[self.orderIds][
            self.restinorder]

        print(k + len(self.restinorder))
        sh_orderIds = sh_orderlIds + sh_ordermIds + sh_orderlIds * sh_orderlIds
        tmp = np.arange(len(sh_orderIds))
        sh_orderinorg = tmp.copy()
        sh_orderinorg[sh_orderIds] = tmp[:]

        return sh_ordermIds, sh_orderlIds, sh_orderIds, sh_orderinorg


class SphericalHarm_table(object):
    def __init__(self, total_deg):
        self.total_deg = total_deg
        print(self.total_deg * self.total_deg)
        # TODO 
        self.C0 = np.sqrt(1 / np.pi)
        self.C1 = np.sqrt(3 / 4 / np.pi)
        self.C20 = np.sqrt(15 / np.pi)
        self.C21 = np.sqrt(5 / np.pi)
        self.C30 = np.sqrt(35.0 / 2 / np.pi)
        self.C31 = np.sqrt(105 / np.pi)
        self.C32 = np.sqrt(21 / 2 / np.pi)
        self.C33 = np.sqrt(7 / np.pi)
        self.C40 = np.sqrt(35.0 / np.pi)
        self.C41 = np.sqrt(35.0 / 2 / np.pi)
        self.C42 = np.sqrt(5 / np.pi)
        self.C43 = np.sqrt(5 / 2 / np.pi)
        self.C44 = np.sqrt(1 / np.pi)

    def sh_all(self, indirs, filp_dir=True):
        indirs = indirs.reshape(-1, 3)
        x = -indirs[..., [0]] if filp_dir else indirs[..., [0]]
        y = -indirs[..., [1]] if filp_dir else indirs[..., [1]]
        z = indirs[..., [2]]

        if self.total_deg == 1:
            return self.SH_l0(x, y, z)
        elif self.total_deg == 2:
            return self.SH_l1(x, y, z)
        elif self.total_deg == 3:
            return self.SH_l2(x, y, z)
        elif self.total_deg == 4:
            return self.SH_l3(x, y, z)
        elif self.total_deg == 5:
            return self.SH_l4(x, y, z)
        else:
            print(
                "Not supporting this order of SH table yet. Please use runtime SH computation."
            )
            exit()

    def SH_l0(self, x, y, z):
        l00 = 0.5 * self.C0 * torch.ones_like(x, device=x.device)
        return l00

    def SH_l1(self, x, y, z):

        l1_m1 = self.C1 * y
        l1_0 = self.C1 * z
        l1_1 = self.C1 * x

        return torch.cat([self.SH_l0(x, y, z), l1_m1, l1_0, l1_1], -1)

    def SH_l2(self, x, y, z):

        l2_m2 = 0.5 * self.C20 * x * y
        l2_m1 = 0.5 * self.C20 * z * y
        l2_0 = 0.25 * self.C21 * (-x * x - y * y + 2 * z * z)
        l2_1 = 0.5 * self.C20 * x * z
        l2_2 = 0.25 * self.C20 * (x * x - y * y)

        return torch.cat([self.SH_l1(x, y, z), l2_m2, l2_m1, l2_0, l2_1, l2_2], -1)

    def SH_l3(self, x, y, z):

        l3_m3 = 0.25 * self.C30 * (3 * x * x - y * y) * y
        l3_m2 = 0.5 * self.C31 * x * y * z
        l3_m1 = 0.25 * self.C32 * (4 * z * z - x * x - y * y) * y
        l3_0 = 0.25 * self.C33 * (2 * z * z - 3 * x * x - 3 * y * y) * z
        l3_1 = 0.25 * self.C32 * (4 * z * z - x * x - y * y) * x
        l3_2 = 0.25 * self.C31 * (x * x - y * y) * z
        l3_3 = 0.25 * self.C30 * (x * x - 3 * y * y) * x

        return torch.cat(
            [self.SH_l2(x, y, z), l3_m3, l3_m2, l3_m1, l3_0, l3_1, l3_2, l3_3], -1)

    def SH_l4(self, x, y, z):

        l4_m4 = 0.75 * self.C40 * x * y * (x * x - y * y)
        l4_m3 = 0.75 * self.C41 * (3 * x * x - y * y) * y * z
        l4_m2 = 0.75 * self.C42 * x * y * (7 * z * z - 1)
        l4_m1 = 0.75 * self.C43 * z * y * (7 * z * z - 3)
        l4_0 = 3 / 16 * self.C44 * (35 * z * z * z * z - 30 * z * z + 3)
        l4_1 = 0.75 * self.C43 * x * z * (7 * z * z - 3)
        l4_2 = 3 / 8 * self.C42 * (x * x - y * y) * (7 * z * z - 1)
        l4_3 = 0.75 * self.C41 * (x * x - 3 * y * y) * x * z
        l4_4 = 3 / 16 * self.C40 * (x * x * (x * x - 3 * y * y) - y * y * (3 * x * x - y * y))

        return torch.cat([
            self.SH_l3(x, y, z), l4_m4, l4_m3, l4_m2, l4_m1, l4_0, l4_1, l4_2,
            l4_3, l4_4], -1)

class IDE(object):
    def __init__(self, total_deg):
        self.total_deg = total_deg
        # TODO 
        self.C0 = np.sqrt(1 / np.pi)
        self.C1 = np.sqrt(3 / 4 / np.pi)
        self.C20 = np.sqrt(15 / np.pi)
        self.C21 = np.sqrt(5 / np.pi)
        self.C30 = np.sqrt(35.0 / 2 / np.pi)
        self.C31 = np.sqrt(105 / np.pi)
        self.C32 = np.sqrt(21 / 2 / np.pi)
        self.C33 = np.sqrt(7 / np.pi)
        self.C40 = np.sqrt(35.0 / np.pi)
        self.C41 = np.sqrt(35.0 / 2 / np.pi)
        self.C42 = np.sqrt(5 / np.pi)
        self.C43 = np.sqrt(5 / 2 / np.pi)
        self.C44 = np.sqrt(1 / np.pi)

        self.C80 = 1/256. * np.sqrt(17/np.pi)
        self.C81 = -3/64. * np.sqrt(17/(2*np.pi))
        self.C82 = 3/128. * np.sqrt(595/np.pi)
        self.C83 = -1/64. * np.sqrt(19635/(2*np.pi))
        self.C84 = 3/128. * np.sqrt(1309/(2*np.pi))
        self.C85 = -3/64. * np.sqrt(17017/(2*np.pi))
        self.C86 = 1/128. * np.sqrt(7293/np.pi)
        self.C87 = -3/64. * np.sqrt(12155/(2*np.pi))
        self.C88 = 3/256. * np.sqrt(12155/(2*np.pi))

    def encode(self, indirs, k, filp_dir=True):
        indirs = indirs.reshape(-1, 3)
        x = -indirs[..., [0]] if filp_dir else indirs[..., [0]]
        y = -indirs[..., [1]] if filp_dir else indirs[..., [1]]
        z = indirs[..., [2]]
        enc = []
        for i in range(self.total_deg):
            l = 2**i
            Y = getattr(self, f'SH_L{i}')(x, y, z)
            A = 1 + torch.expm1(-l*(l+1)/k)
            enc.append(A*Y)
        
        return torch.cat(enc, -1)

    def SH_L0(self, x, y, z):

        # l1_m1 = self.C1 * y
        l1_0 = self.C1 * z
        l1_1 = self.C1 * x

        return torch.cat([l1_0, l1_1], -1)

    def SH_L1(self, x, y, z): # l = 2

        # l2_m2 = 0.5 * self.C20 * x * y
        # l2_m1 = 0.5 * self.C20 * z * y
        l2_0 = 0.25 * self.C21 * (-x * x - y * y + 2 * z * z)
        l2_1 = 0.5 * self.C20 * x * z
        l2_2 = 0.25 * self.C20 * (x * x - y * y)

        return torch.cat([l2_0, l2_1, l2_2], -1)

    def SH_L2(self, x, y, z): # l = 4

        # l4_m4 = 0.75 * self.C40 * x * y * (x * x - y * y)
        # l4_m3 = 0.75 * self.C41 * (3 * x * x - y * y) * y * z
        # l4_m2 = 0.75 * self.C42 * x * y * (7 * z * z - 1)
        # l4_m1 = 0.75 * self.C43 * z * y * (7 * z * z - 3)
        l4_0 = 3 / 16 * self.C44 * (35 * z * z * z * z - 30 * z * z + 3)
        l4_1 = 0.75 * self.C43 * x * z * (7 * z * z - 3)
        l4_2 = 3 / 8 * self.C42 * (x * x - y * y) * (7 * z * z - 1)
        l4_3 = 0.75 * self.C41 * (x * x - 3 * y * y) * x * z
        l4_4 = 3 / 16 * self.C40 * (x * x * (x * x - 3 * y * y) - y * y * (3 * x * x - y * y))

        return torch.cat([
            # l4_m4, l4_m3, l4_m2, l4_m1, 
            l4_0, l4_1, l4_2, l4_3, l4_4], -1)
    
    def SH_L3(self, x, y, z): # l = 8
        l8_0 = self.C80 * (6435 * z**8 - 12012* z**6 + 6930*z**4 - 1260*z**2 + 35)
        l8_1 = self.C81 * x * (715 * z**7 - 1001*z**5 + 385*z**3 - 35*z)
        l8_2 = self.C82 * (x*x-y*y) * (143*z**6 - 143*z**4 + 33*z**2 - 1)
        l8_3 = self.C83 * x * (x**2 - 3*y**2) * (39*z**5 - 26*z**3 + 3*z)
        l8_4 = self.C84 * (x**4-6*x**2*y**2+y**4) * (65*z**4 - 26*z**2 + 1)
        l8_5 = self.C85 * (x**5 - 10*x**3*y**2+5*x*y**4) * (5*z**3 - z)
        l8_6 = self.C86 * (x**6 - 15*x**4*y**2+15*x**2*y**4 - y**6) * (15*z**2 - 1)
        l8_7 = self.C87 * (x**7 - 21*x**5*y**2 + 35*x**3*y**4 - 7*x*y**6) * z
        l8_8 = self.C88 * (x**8 - 28*x**6*y**2 + 70*x**4*y**4 - 28*x**2*y**6 + y**8)

        return torch.cat([
            l8_0, l8_1, l8_2, l8_3, l8_4, l8_5, l8_6, l8_7, l8_8
        ], -1)