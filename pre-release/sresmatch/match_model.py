import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

eps = 1e-8

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    #print(scores.max(),scores.min(),scores.mean())
    one = scores.new_tensor(1)
    ms, ns = (m*one), (n*one)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z.exp()


class attention_propagantion(nn.Module):
    def __init__(self, channel, head):
        nn.Module.__init__(self)
        self.head = head
        self.channel = channel
        self.head_dim = channel//head
        self.res_lam = nn.Parameter(torch.ones([1,head,1,1]))
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channel, channel, kernel_size=1), nn.Conv1d(channel, channel, kernel_size=1),\
            nn.Conv1d(channel, channel, kernel_size=1)
        self.mh_filter = nn.Conv1d(channel, channel, kernel_size=1)
        self.cat_filter = nn.Sequential(nn.Conv1d(2*channel, 2*channel, kernel_size=1,bias=False), nn.SyncBatchNorm(2*channel), nn.ReLU(),
                                        nn.Conv1d(2*channel, channel, kernel_size=1))

    def forward(self, desc1, desc2, neighb,neigh_mask,res_attention):
        b, c, n = desc1.shape
        k = neighb.shape[1]
        if (not self.training) and not (desc1.shape[-1]<2500 and desc2.shape[-1]<2500):
            query, key, value = self.query_filter(
            desc1), self.key_filter(desc2), self.value_filter(desc2)
            neighb = neighb.reshape(b, 1, -1)
            kv = torch.cat([key,value],dim=1)
            kv_neigh = kv.gather(dim=-1, index=neighb.expand(-1, 2*c, -1))
            key_neigh,value_neigh = kv_neigh.split([c,c],dim=1)
            key_neigh = key_neigh.view(b, self.head, self.head_dim, k, n)
            value_neigh = value_neigh.view(b, self.head, self.head_dim, k, n)
            query_t = query.view(b, self.head, 1, self.head_dim, -1)
            
            simi_einsum = torch.einsum('bhocn,bhckn->bhokn',query_t,key_neigh).squeeze(2)/self.head_dim ** 0.5+self.res_lam*neigh_mask.unsqueeze(1).expand(-1,self.head,-1,-1)
            score_exp = simi_einsum.clamp(min=-30,max=30).exp()
            score_exp_sum = score_exp.sum(dim=-2,keepdim=True)
            score = (score_exp/(score_exp_sum+1e-8)).unsqueeze(2)
            add_value = (score*value_neigh).sum(dim=-2).view(b, self.head_dim*self.head, -1)
        else:
            query,key,value=self.query_filter(desc1).view(b,self.head,self.head_dim,-1),self.key_filter(desc2).view(b,self.head,self.head_dim,-1),\
                            self.value_filter(desc2).view(b,self.head,self.head_dim,-1)
            neigh_mask = neigh_mask.unsqueeze(1).expand(-1,self.head,-1,-1)
            simi = torch.einsum('bhdn,bhdm->bhnm',query,key)/ self.head_dim ** 0.5+self.res_lam*res_attention.unsqueeze(1).expand(-1,self.head,-1,-1)
            if True:
                score_exp = simi.clamp(min=-30,max=30).exp()*neigh_mask
            else:
                score_exp = torch.masked_fill(simi.clamp(-30,30).exp(),torch.logical_not(neigh_mask.bool()),0)
            score_exp_sum = score_exp.sum(dim=-1,keepdim=True)
            score = score_exp/(score_exp_sum+1e-8)
            add_value=torch.einsum('bhnm,bhdm->bhdn',score,value).reshape(b,self.head_dim*self.head,-1)

        add_value = self.mh_filter(add_value)
        desc1_new = desc1+self.cat_filter(torch.cat([desc1, add_value], dim=1))
        return desc1_new



class match_block(nn.Module):
    def __init__(self, channel, head):
        nn.Module.__init__(self)
        self.self_attention = attention_propagantion(channel, head)
        self.cross_attention = attention_propagantion(channel, head)

    def forward(self, desc1, desc2, neigh_self1, neigh_self2, neigh_cross12, neigh_cross21,self_neigh1_mask,self_neigh2_mask,cross_neigh12_mask,cross_neigh21_mask,
                self_res1,self_res2,cross_res):
        desc1, desc2 = self.self_attention(desc1, desc1, neigh_self1,self_neigh1_mask,self_res1), self.self_attention(desc2, desc2, neigh_self2,self_neigh2_mask,self_res2)
        desc1, desc2 = self.cross_attention(desc1, desc2, neigh_cross12,cross_neigh12_mask,cross_res), self.cross_attention(desc2, desc1, neigh_cross21,cross_neigh21_mask,cross_res.transpose(-1,-2))
        return desc1,desc2

        
class matcher(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.use_score_encoding = False
        self.layer_num = 9
        self.net_channels = 128
        self.head = 4
        self.sink_iter = [100] #10 for scannet
        self.adjlayer = [0,4]
        self.channels = 128
        self.k=64

        self.position_encoder = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1,bias=False) if self.use_score_encoding else nn.Conv1d(2, 32, kernel_size=1,bias=False), 
                                              nn.SyncBatchNorm(32), nn.ReLU(),
                                              nn.Conv1d(32, 64, kernel_size=1,bias=False), nn.SyncBatchNorm(64),nn.ReLU(),
                                              nn.Conv1d(64, 128, kernel_size=1,bias=False), nn.SyncBatchNorm(128), nn.ReLU(),
                                              nn.Conv1d(128, 256, kernel_size=1,bias=False), nn.SyncBatchNorm(256), nn.ReLU(),
                                              nn.Conv1d(256, self.net_channels, kernel_size=1))

        self.position_encoder2 = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1,bias=False) if self.use_score_encoding else nn.Conv1d(2, 32, kernel_size=1,bias=False), 
                                              nn.SyncBatchNorm(32), nn.ReLU(),
                                              nn.Conv1d(32, 64, kernel_size=1,bias=False), 
                                              nn.SyncBatchNorm(64), nn.ReLU(),
                                              nn.Conv1d(64, 128, kernel_size=1,bias=False), 
                                              nn.SyncBatchNorm(128), nn.ReLU(),
                                              nn.Conv1d(128, self.net_channels//4, kernel_size=1))

        self.desc_compressor = nn.Sequential(nn.Conv1d(self.net_channels, self.net_channels*2, kernel_size=1,bias=False), 
                                              nn.InstanceNorm1d(self.net_channels*2),nn.SyncBatchNorm(self.net_channels*2), nn.ReLU(),
                                              nn.Conv1d(self.net_channels*2, self.net_channels*2, kernel_size=1,bias=False), 
                                              nn.InstanceNorm1d(self.net_channels*2),nn.SyncBatchNorm(self.net_channels*2), nn.ReLU(),
                                              nn.Conv1d(self.net_channels*2, self.net_channels, kernel_size=1))

        self.desc_compressor2 = nn.Sequential(nn.Conv1d(self.net_channels, self.net_channels*2, kernel_size=1,bias=False), 
                                              nn.InstanceNorm1d(self.net_channels*2),nn.SyncBatchNorm(self.net_channels*2), nn.ReLU(),
                                              nn.Conv1d(self.net_channels*2, self.net_channels*2, kernel_size=1,bias=False), 
                                              nn.InstanceNorm1d(self.net_channels*2),nn.SyncBatchNorm(self.net_channels*2), nn.ReLU(),
                                              nn.Conv1d(self.net_channels*2, self.net_channels, kernel_size=1))

        self.desc_mid_project = nn.Sequential(nn.Conv1d(self.net_channels, self.net_channels*2, kernel_size=1,bias=False),
                                              nn.SyncBatchNorm(self.net_channels*2), nn.ReLU(),
                                              nn.Conv1d(self.net_channels*2, self.net_channels//4, kernel_size=1))

        self.pos_mid_project = nn.Sequential(nn.Conv1d(self.net_channels, self.net_channels*2, kernel_size=1,bias=False), 
                                             nn.SyncBatchNorm(self.net_channels*2), nn.ReLU(),
                                             nn.Conv1d(self.net_channels*2, self.net_channels//4, kernel_size=1))
        self.drop_out = nn.Dropout(0.01)
        self.dustbin = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.blks = nn.Sequential(
            *[match_block(self.net_channels, self.head) for _ in range(self.layer_num)])
        self.final_project = nn.Conv1d(
            self.net_channels, self.net_channels, kernel_size=1)
        self.final_norm = nn.SyncBatchNorm(1,affine=True)
        
    def forward(self, data, test_mode=True):
        desc1, desc2 = data['desc1'], data['desc2']
        desc1, desc2 = torch.nn.functional.normalize(desc1, dim=-1), torch.nn.functional.normalize(desc2, dim=-1)
        desc1, desc2 = desc1.transpose(1, 2), desc2.transpose(1, 2)

        if test_mode:
            encode_x1, encode_x2 = data['x1'], data['x2']
        else:
            encode_x1, encode_x2 = data['aug_x1'], data['aug_x2']
        if not self.use_score_encoding:
            encode_x1, encode_x2 = encode_x1[:, :, :2], encode_x2[:, :, :2]



        encode_x1, encode_x2 = encode_x1.transpose(1, 2), encode_x2.transpose(1, 2)
        x1_pos_embedding, x2_pos_embedding = self.position_encoder(encode_x1), self.position_encoder(encode_x2)
        aug_desc = self.desc_compressor(torch.cat([desc1, desc2], dim=-1))
        aug_desc1_raw, aug_desc2_raw = torch.split(aug_desc, [desc1.shape[-1], desc2.shape[-1]], dim=-1)
        aug_desc1, aug_desc2 = aug_desc1_raw+x1_pos_embedding, aug_desc2_raw+x2_pos_embedding


        aug_desc_ = self.desc_compressor2(torch.cat([desc1, desc2], dim=-1))
        aug_desc1_, aug_desc2_ = torch.split(aug_desc_, [desc1.shape[-1], desc2.shape[-1]], dim=-1)
        cross_neigh12,cross_neigh21,cross_neigh12_mask,cross_neigh21_mask,cross_res= self.cross_neigh_mining(aug_desc1_,aug_desc2_,k=self.k//2)

        pos_embedding = self.position_encoder2(torch.cat([encode_x1,encode_x2],dim=-1))
        x1_pos_embedding_, x2_pos_embedding_ = torch.split(pos_embedding,[desc1.shape[-1], desc2.shape[-1]], dim=-1)
        self_neigh1,self_neigh1_mask,self_res1 = self.self_neigh_mining(x1_pos_embedding_,k=self.k,res=0)
        self_neigh2,self_neigh2_mask,self_res2 = self.self_neigh_mining(x2_pos_embedding_,k=self.k,res=0)
        for i in range(self.layer_num):
            if i in self.adjlayer and i!= 0:
                aug_desc_mid = torch.cat([aug_desc1, aug_desc2], dim=-1)
                aug_desc1_desc, aug_desc2_desc = torch.split(self.desc_mid_project(aug_desc_mid), [desc1.shape[-1], desc2.shape[-1]], dim=-1)
                aug_desc1_pos, aug_desc2_pos = torch.split(self.pos_mid_project(aug_desc_mid), [desc1.shape[-1], desc2.shape[-1]], dim=-1)
                cross_neigh12,cross_neigh21,cross_neigh12_mask,cross_neigh21_mask,cross_res = self.cross_neigh_mining(aug_desc1_desc,aug_desc2_desc,k=self.k//4,res=cross_res)
                self_neigh1,self_neigh1_mask,self_res1 = self.self_neigh_mining(aug_desc1_pos,k=self.k,res=self_res1,stage=i)
                self_neigh2,self_neigh2_mask,self_res2 = self.self_neigh_mining(aug_desc2_pos,k=self.k,res=self_res2,stage=i)
            cross_res_drop = self.drop_out(cross_res)
            aug_desc1,aug_desc2=self.blks[i](aug_desc1,aug_desc2,
                                            self_neigh1,self_neigh2,
                                            cross_neigh12,cross_neigh21,
                                            self_neigh1_mask,self_neigh2_mask,
                                            cross_neigh12_mask,cross_neigh21_mask,
                                            self_res1,self_res2,cross_res_drop)
        aug_desc1, aug_desc2 = self.final_project(
            aug_desc1), self.final_project(aug_desc2)
        desc_mat = torch.matmul(aug_desc1.transpose(1, 2), aug_desc2)
        desc_mat = self.final_norm(desc_mat.unsqueeze(1)).squeeze(1)
        p = log_optimal_transport(desc_mat, self.dustbin, self.sink_iter[-1])
        return {'p': p}

    def cross_neigh_mining(self,desc1,desc2,k,res=0):
        if not self.training:
            k = k+1

        simi = torch.einsum('bcn,bcm->bnm',desc1,desc2)+res
        if k> simi.shape[-1]:
            top12,ind12 = simi.topk(simi.shape[-1],dim=-1,sorted=False)
        else:
            if self.training:
                top12,ind12 = simi.topk(k,dim=-1,sorted=False)
            else:
                top12,ind12 = simi.topk(k,dim=-1,sorted=False)

        if k > simi.shape[-2]:
            top21,ind21 = simi.topk(simi.shape[-2],dim=-2,sorted=False)
        else:    
            top21,ind21 = simi.topk(k,dim=-2,sorted=False)

        if self.training:
            mask12 = (top12[:,:,-1]+eps).unsqueeze(-1)<=simi
            mask21 = (top21[:,-1,:]+eps).unsqueeze(1)<=simi
            return ind12.permute(0,2,1),ind21,mask12,mask21.transpose(-1,-2),simi
        else:
            if desc1.shape[-1]<2500 and desc2.shape[-1]<2500:
                mask12 = top12[:,:,-1].unsqueeze(-1)<=simi
                mask21 = top21[:,-1,:].unsqueeze(1)<=simi
                return ind12.permute(0,2,1),ind21,mask12,mask21.transpose(-1,-2),simi
            else:
                return ind12.permute(0,2,1),ind21,top12.permute(0,2,1),top21,simi

    def self_neigh_mining(self,pos,k,res=0,stage=0):
        b,c,n=pos.shape
        if not self.training:
            k = k+1
        if k>pos.shape[-1]:
            k = pos.shape[-1]
        simi = torch.einsum('bcn,bcm->bnm',pos,pos)
        tens = torch.full([n],10.0,device='cuda')
        diag_mask = torch.diag(tens)
        simi = simi - diag_mask + res

        if stage<1 and pos.shape[-1]>2500 and not self.training:
            top12,_=simi[:,:,:1000].topk(k//2,dim=-1,sorted=False)
            resample_mask = (top12[:,:,-1]+eps).unsqueeze(0)<simi
            resample_mask[:,:,:1000]=False
            simi_resample = torch.masked_fill(simi,resample_mask,-10)
            top12,ind12 = simi_resample.topk(k,dim=-1,sorted=False)
        else:
            top12,ind12 = simi.topk(k,dim=-1,sorted=False)

        if self.training:
            mask12 = (top12[:,:,-1]+eps).unsqueeze(-1)<=simi
            return ind12.permute(0,2,1),mask12,simi
        else:
            if pos.shape[-1]<2500:
                mask12 = top12[:,:,-1].unsqueeze(-1)<=simi
                return ind12.permute(0,2,1),mask12,simi
            else:
                return ind12.permute(0,2,1),top12.permute(0,2,1),simi