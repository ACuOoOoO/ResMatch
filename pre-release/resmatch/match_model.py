import torch
import torch.nn as nn
import torch.nn.functional as F

eps=1e-8

# sinkhorn algorithm from offical code of SuperGlue
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
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

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
    

#Safe softmax
def softmax_eps(x,eps=1e-12):
    x_exp=(x).clamp(max=50).exp()
    x_exp_sum=x_exp.sum(dim=-1,keepdim=True)
    return x_exp/(x_exp_sum+eps)

class attention_block(nn.Module):
    def __init__(self,channels,head,type,slope=0.1):
        assert type=='self' or type=='cross','invalid attention type'
        nn.Module.__init__(self)
        self.head=head
        self.type=type
        self.slope = slope
        self.head_dim=channels//head
        self.query_filter=nn.Conv1d(channels, channels, kernel_size=1)
        self.key_filter=nn.Conv1d(channels,channels,kernel_size=1)
        self.value_filter=nn.Conv1d(channels,channels,kernel_size=1)
        self.res_lam = nn.Parameter(torch.ones([1,head,1,1]))
        self.res_bias = nn.Parameter(torch.zeros([1,head,1,1]))
        self.attention_filter=nn.Sequential(nn.Conv1d(2*channels,2*channels, kernel_size=1),nn.SyncBatchNorm(2*channels), nn.ReLU(),
                                             nn.Conv1d(2*channels, channels, kernel_size=1))
        self.mh_filter=nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self,fea1,fea2,res_atten1=None,res_atten2=None):
        batch_size,n,m=fea1.shape[0],fea1.shape[2],fea2.shape[2]
        query1, key1, value1 = self.query_filter(fea1).view(batch_size,self.head_dim,self.head,-1), self.key_filter(fea1).view(batch_size,self.head_dim,self.head,-1), \
                               self.value_filter(fea1).view(batch_size,self.head_dim,self.head,-1)
        query2, key2, value2 = self.query_filter(fea2).view(batch_size,self.head_dim,self.head,-1), self.key_filter(fea2).view(batch_size,self.head_dim,self.head,-1), \
                               self.value_filter(fea2).view(batch_size,self.head_dim,self.head,-1)
        if(self.type=='self'):
            score1 = torch.einsum('bdhn,bdhm->bhnm',query1,key1)/self.head_dim**0.5+F.leaky_relu(self.res_lam*res_atten1.unsqueeze(1).expand(-1,self.head,-1,-1)+self.res_bias,negative_slope=self.slope)
            score2 = torch.einsum('bdhn,bdhm->bhnm',query2,key2)/self.head_dim**0.5+F.leaky_relu(self.res_lam*res_atten2.unsqueeze(1).expand(-1,self.head,-1,-1)+self.res_bias,negative_slope=self.slope)
            score1_sm,score2_sm=softmax_eps(score1),softmax_eps(score2)
            add_value1, add_value2 = torch.einsum('bhnm,bdhm->bdhn', score1_sm, value1), torch.einsum('bhnm,bdhm->bdhn',score2_sm, value2)
        else:
            score1 = torch.einsum('bdhn,bdhm->bhnm', query1, key2) / self.head_dim ** 0.5
            score2 = torch.einsum('bdhn,bdhm->bhnm', query2, key1) / self.head_dim ** 0.5
            score1=score1+F.leaky_relu(self.res_lam*res_atten1.unsqueeze(1).expand(-1,self.head,-1,-1)+self.res_bias,negative_slope=self.slope)
            score2=score2+F.leaky_relu(self.res_lam*res_atten2.unsqueeze(1).expand(-1,self.head,-1,-1)+self.res_bias,negative_slope=self.slope)
            
            score1_sm,score2_sm = softmax_eps(score1),softmax_eps(score2)
            
            add_value1, add_value2 =torch.einsum('bhnm,bdhm->bdhn',score1_sm,value2),torch.einsum('bhnm,bdhm->bdhn',score2_sm,value1)
        
        add_value1,add_value2=self.mh_filter(add_value1.contiguous().view(batch_size,self.head*self.head_dim,n)),self.mh_filter(add_value2.contiguous().view(batch_size,self.head*self.head_dim,m))
        fea11, fea22 = torch.cat([fea1, add_value1], dim=1), torch.cat([fea2, add_value2], dim=1)
        fea1, fea2 = fea1+self.attention_filter(fea11), fea2+self.attention_filter(fea22)

        return fea1,fea2


class matcher(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.use_score_encoding=False
        self.net_channels = 128
        self.head = 4
        self.layer_num=9
        self.sink_iter= [100] #10 for scannet test
        self.adjlayer = [0,4]
        
        self.position_encoder = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1,bias=False) if self.use_score_encoding else nn.Conv1d(2, 32, kernel_size=1,bias=False), 
                                                nn.SyncBatchNorm(32), nn.ReLU(),
                                                nn.Conv1d(32, 64, kernel_size=1,bias=False), nn.SyncBatchNorm(64),nn.ReLU(),
                                                nn.Conv1d(64, 128, kernel_size=1,bias=False), nn.SyncBatchNorm(128), nn.ReLU(),
                                                nn.Conv1d(128, 256, kernel_size=1,bias=False), nn.SyncBatchNorm(256), nn.ReLU(),
                                                nn.Conv1d(256, self.net_channels, kernel_size=1))

        self.res_pos_proj = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1,bias=False) if self.use_score_encoding else nn.Conv1d(2, 32, kernel_size=1,bias=False), 
                                            nn.SyncBatchNorm(32), nn.ReLU(),
                                            nn.Conv1d(32, 64, kernel_size=1,bias=False), 
                                            nn.SyncBatchNorm(64),nn.ReLU(),
                                            nn.Conv1d(64, 128, kernel_size=1,bias=False), 
                                            nn.SyncBatchNorm(128), nn.ReLU(),
                                            nn.Conv1d(128, self.net_channels, kernel_size=1))

        self.desc_compressor1 = nn.Sequential(nn.Conv1d(self.net_channels, self.net_channels*2, kernel_size=1,bias=False), 
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
                                              nn.Conv1d(self.net_channels*2, self.net_channels, kernel_size=1))

        self.pos_mid_project = nn.Sequential(nn.Conv1d(self.net_channels, self.net_channels*2, kernel_size=1,bias=False), 
                                             nn.SyncBatchNorm(self.net_channels*2), nn.ReLU(),
                                             nn.Conv1d(self.net_channels*2, self.net_channels, kernel_size=1))

        self.dustbin=nn.Parameter(torch.tensor(1,dtype=torch.float32))
        self.self_attention_block=nn.Sequential(*[attention_block(self.net_channels,self.head,'self') for _ in range(self.layer_num)])
        self.cross_attention_block=nn.Sequential(*[attention_block(self.net_channels,self.head,'cross') for _ in range(self.layer_num)])
        self.final_project=nn.Conv1d(self.net_channels, self.net_channels, kernel_size=1)
        self.final_norm = nn.SyncBatchNorm(1,affine=True)
        self.drop_out = nn.Dropout(0.01)

    def forward(self,data,test_mode=True):
        desc1, desc2 = data['desc1'], data['desc2']
        desc1, desc2 = torch.nn.functional.normalize(desc1,dim=-1), torch.nn.functional.normalize(desc2,dim=-1)
        desc1,desc2=desc1.transpose(1,2),desc2.transpose(1,2)   
        if test_mode:
            encode_x1,encode_x2=data['x1'],data['x2']
        else:
            encode_x1,encode_x2=data['aug_x1'], data['aug_x2']
        if not self.use_score_encoding:
            encode_x1,encode_x2=encode_x1[:,:,:2],encode_x2[:,:,:2]

        encode_x1,encode_x2=encode_x1.transpose(1,2),encode_x2.transpose(1,2)

        x1_pos_embedding, x2_pos_embedding = self.position_encoder(encode_x1), self.position_encoder(encode_x2)
        aug_desc = self.desc_compressor1(torch.cat([desc1,desc2],dim=-1))
        aug_desc1,aug_desc2=torch.split(aug_desc,[desc1.shape[-1],desc2.shape[-1]],dim=-1)
        aug_desc1,aug_desc2 = aug_desc1+x1_pos_embedding,aug_desc2+x2_pos_embedding

        #simi of desc    
        desc_ = self.desc_compressor2(torch.cat([desc1, desc2], dim=-1))/(aug_desc1.shape[1]**0.25)
        desc1_, desc2_ = torch.split(desc_, [desc1.shape[-1], desc2.shape[-1]], dim=-1)
        cross_res = torch.einsum('bcn,bcm->bnm',desc1_,desc2_)
        eye_mask1 = torch.eye(aug_desc1.shape[-1],device='cuda')
        eye_mask2 = torch.eye(aug_desc2.shape[-1],device='cuda')

        #relative pos
        pos_embedding = self.res_pos_proj(torch.cat([encode_x1,encode_x2],dim=-1))/(aug_desc1.shape[1]**0.25)
        x1_pos_embedding_, x2_pos_embedding_ = torch.split(pos_embedding,[desc1.shape[-1], desc2.shape[-1]], dim=-1)
        self_res1 = torch.einsum('bcn,bcm->bnm',x1_pos_embedding_,x1_pos_embedding_)-eye_mask1
        self_res2 = torch.einsum('bcn,bcm->bnm',x2_pos_embedding_,x2_pos_embedding_)-eye_mask2
        for i in range(self.layer_num):
            if i in self.adjlayer and i!= 0:
                #adjustment
                aug_desc_mid = torch.cat([aug_desc1, aug_desc2], dim=-1)
                aug_desc1_desc, aug_desc2_desc = torch.split(self.desc_mid_project(aug_desc_mid)/(aug_desc1.shape[1]**0.25), [desc1.shape[-1], desc2.shape[-1]], dim=-1)
                aug_desc1_pos, aug_desc2_pos = torch.split(self.pos_mid_project(aug_desc_mid)/(aug_desc1.shape[1]**0.25), [desc1.shape[-1], desc2.shape[-1]], dim=-1)
                cross_res = (torch.einsum('bcn,bcm->bnm',aug_desc1_desc,aug_desc2_desc)+cross_res)/2
                self_res1 = (torch.einsum('bcn,bcm->bnm',aug_desc1_pos,aug_desc1_pos)+self_res1)/2-eye_mask1
                self_res2 = (torch.einsum('bcn,bcm->bnm',aug_desc2_pos,aug_desc2_pos)+self_res2)/2-eye_mask2
            cross_res_drop1 = self.drop_out(cross_res)
            cross_res_drop2 = self.drop_out(cross_res)
            aug_desc1,aug_desc2=self.self_attention_block[i](aug_desc1,aug_desc2,self_res1,self_res2)
            aug_desc1,aug_desc2=self.cross_attention_block[i](aug_desc1,aug_desc2,cross_res_drop1,cross_res_drop2.transpose(-1,-2))

        aug_desc1,aug_desc2=self.final_project(aug_desc1),self.final_project(aug_desc2)
        desc_mat = torch.matmul(aug_desc1.transpose(1, 2), aug_desc2)
        desc_mat = self.final_norm(desc_mat.unsqueeze(1)).squeeze(1)
        p = log_optimal_transport(desc_mat, self.dustbin,self.sink_iter[-1])
        return {'p':p}
