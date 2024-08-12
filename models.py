from ast import arg
import enum
import os
import math
import sys
import numpy as np
import torch
import random
import hydra
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import perturbations
import cv2
from torchsummary import summary
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True # type: ignore
    # torch.backends.cudnn.benchmark = False # type: ignore # if benchmark=True, deterministic will be False
    # prevent only sample one video
    #np.random.seed(seed)
    #random.seed(seed)


# spitial attention
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        # pe is of shape max_len(5000) x 2048(last layer of FC)
        pe = torch.zeros(max_len, d_model,device='cuda')
        # position is of shape 5000 x 1
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape 1 x 5000 x 2048
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + self.pe[:, :x.size(1)] # type: ignore
       return self.dropout(x)
class sub_linear(nn.Module):
    def __init__(self,dim):
        super(sub_linear,self).__init__()
        self.in_feature = nn.Linear(dim,dim//2)   # 2048 -> 1024
        self.hidden = nn.Linear(dim//2,dim//2)
        self.out_feature = nn.Linear(dim//2,dim)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.in_feature(x)    # 2048 -> 1024
        out = self.relu(out)
        out = self.hidden(out)
        out = self.relu(out)
        out = self.out_feature(out)
        return out + x              # b x 16 x 2048
class spatial_attention(nn.Module):
    def __init__(self,trans_linear_in_dim,patch_numbers):
        super(spatial_attention, self).__init__()
        self.query_projection = nn.Linear(trans_linear_in_dim,trans_linear_in_dim)
        self.key_projection = nn.Linear(trans_linear_in_dim,trans_linear_in_dim)
        self.value_projection = nn.Linear(trans_linear_in_dim,trans_linear_in_dim)

        self.position_encoding = PositionalEncoding(trans_linear_in_dim,0.1)
        self.softmax = nn.Softmax(dim=-1)

        # treat gamma as a learnable parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sub_linear = sub_linear(trans_linear_in_dim)

    def forward(self,x):
        '''
            input: [seq_len,patch_numbers,channel]
            return: attention + input, same shape
        '''

        x = self.position_encoding(x)
        residual = x

        query = self.query_projection(x) #[]
        key = self.key_projection(x)
        value = self.value_projection(x)
        
        attention,attn_map = self.Attention(query,key,value,True,0.5)

        out = self.gamma * attention + residual
        out = self.sub_linear(out)
        return out,attn_map

    # N: token number, D: token dim
    # Q,K,V: query (seq_len, 4x4, channel)
    # use_DropKey: whether use DropKey 
    # mask_ratio: ratio to mask
    def Attention(self,Q, K, V, use_DropKey, mask_ratio):
        attn = torch.bmm((Q * (Q.shape[1] ** -0.5)) , K.permute(0,2,1)) # (frame_counts, 4x4, 4x4)
        # use DropKey as regularizer
        if use_DropKey == True:
            m_r = torch.ones_like(attn) * mask_ratio 
            attn = attn + torch.bernoulli(m_r) * -1e12
        attn_map = attn.clone()
        attn_map = nn.Softmax(dim=-1)(attn_map)
        attn = nn.Softmax(dim=-1)(attn)
        x = torch.bmm(attn , V) # (frame_counts, 4x4, channel)
        return x,attn_map


# temporal attention
class channel_linear(nn.Module):
    def __init__(self,channek_dim):
        super(channel_linear,self).__init__()
        self.in_feature = nn.Linear(channek_dim,channek_dim)
        self.out_feature = nn.Linear(channek_dim,channek_dim)
        self.relu = nn.ReLU()
    def forward(self,x):
        # [B,seq_len,channel]
        out = self.in_feature(x)
        out = self.relu(out)
        out = self.out_feature(out)
        return out            # need residual connection???

class channel_attention(nn.Module):
    def __init__(self,channel_dim):
        super(channel_attention,self).__init__()
        self.channel_dim = channel_dim
        self.conv1 = nn.Conv1d(channel_dim,channel_dim,1)
        self.channel_linear = channel_linear(channel_dim)

    def forward(self,x):
        '''
            input: [frame_counts,16,channel]
            return: attention + input, same shape
        '''
        fc = x.shape[0]
        residual1 = x
        x = x.transpose(2,1).reshape(-1,self.channel_dim,16)
        x = self.conv1(x).transpose(2,1)
        x = self.channel_linear(x).reshape(fc,16,self.channel_dim) + residual1
        # [seq_len,16,channel]
        return x


# sampler

class Generator(nn.Module):
    def __init__(self, args,insize=512, outsize=512, z_dim=64, bias=False):
        super().__init__()
        self.insize = args.trans_linear_in_dim
        self.outsize = args.trans_linear_in_dim
        self.z_dim = z_dim
        self.bias = bias
        self.encoder = nn.Linear(self.insize, z_dim * 2)
        if bias:
            self.gen = nn.Linear(z_dim, self.outsize + 1)
        else:
            self.gen = nn.Linear(z_dim, self.outsize)

    def forward(self, task_context):
        # task_context: , insize
        distribution = self.encoder(task_context) # , mu_size * 2
        mu = distribution[:self.z_dim]
        log_var = distribution[self.z_dim:]

        z_signal = torch.randn(1, self.z_dim).cuda()
        z_mu = mu.unsqueeze(0) # 1, z_dim
        z_log_var = log_var.unsqueeze(0) # 1, z_dim
        z = z_mu + torch.exp(z_log_var/2) * z_signal # 1, z_dim

        weight_bias = self.gen(z) # 1, out_size * 2
        weight = weight_bias[:, :self.outsize] # 1, out_size
        weight = weight / torch.norm(weight, 2) # normalize 
        if self.bias:
            bias = weight_bias[0, self.outsize:] # ,1
            return weight, bias
        else:
            return weight

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.dim = d_model
        self.seq_len = max_len
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.view(-1, self.seq_len, self.dim)
        x = x + self.pe[:,:]
        return x.view(-1, self.dim)
    
def order_topk(x, k=10):
    if len(x.shape) > 2:
        x = x[:,:,0]
    return F.one_hot(torch.sort(torch.topk(x, k=10, dim=-1)[-1])[0], list(x.shape)[-1]).transpose(-1,-2).float()
class Sampler(nn.Module):
    def __init__(self,args):
        super(Sampler,self).__init__()
        self.args = args
        self.generator = Generator(args)
        if self.args.ada:
            self.evaluator = nn.Sequential(PositionalEmbedding(d_model=self.args.trans_linear_in_dim*2, max_len=self.args.sampler_seq_len),
                                            nn.Linear(self.args.trans_linear_in_dim*2, self.args.trans_linear_in_dim),
                                            nn.ReLU(),
                                            )
        else:
            self.evaluator = nn.Sequential(PositionalEmbedding(d_model=self.args.trans_linear_in_dim*2, max_len=self.args.sampler_seq_len),
                                            nn.Linear(self.args.trans_linear_in_dim*2, self.args.trans_linear_in_dim),
                                            nn.ReLU(),
                                            nn.Linear(512, 1),
                                            )
        self.func = order_topk
    def return_topk_func(self):
        iteration = self.args.current_iter
        if iteration == 1:
            #print('Initial sigma as', self.args.sigma)
            self.current_sigma = self.args.sigma
        sigma_min = 0.01
        step = self.args.sigma_step
        if self.args.sigma_decay and (iteration + 1) % step == 0:
            self.current_sigma = self.args.sigma * ((0.8) ** ((iteration + 1) // step))
            if self.current_sigma < sigma_min:
                self.current_sigma = sigma_min
            #print('Decay sigma, the current sigma is {:.2}:'.format(self.current_sigma))
        topk_func = perturbations.perturbed(self.func, 
                                            num_samples=self.args.num_samples,
                                            sigma=self.current_sigma,
                                            noise=self.args.noise,
                                            batched=True)
        return topk_func


    def fast_topk(self, x):
        return F.one_hot(torch.sort(torch.topk(x, k=self.args.k, dim=-1)[-1])[0], list(x.shape)[-1]).transpose(-1,-2).float()
    

    def forward(self, x, weight = None, category='support'):
        # [B, 1024, 40]
        x = x.transpose(-1,-2) # [B, 40, 1024]
        n,dim = x.shape[1],x.shape[2]
        feature = x

        # Calculate the global feature of the whole video as a kind of guidance
        if self.args.shot > 1:
            info_g = x.view(self.args.way, -1, *x.shape[1:]) # way, shot, frame, dim
            shot = info_g.size(1)
            info_g = info_g.mean(dim=-3, keepdim=True).mean(dim=-2) # way, 1, dimseq_len
            info_g = info_g.repeat(shot, 1, 1).expand(-1, self.args.sampler_seq_len, -1) # way * shot, 1, dim --> way * shot, frame, dim
        else:
            info_g = x.mean(dim=-2, keepdim=True).expand(-1, self.args.sampler_seq_len, -1) # (way * shot, 1, dim) expand to --> (way * shot, frame, dim)

        # Calculate the task feature
        task_f = info_g[:,0,:].mean(dim=0) # ,dim

        # Feed into Evaluator to get scores
        x = torch.cat((x, info_g), dim=-1) # way * shot, frame, 2*dim

        score = self.evaluator(x.view(n, dim*2)) # way * shot * frame, 1 (if task_ada, way * shot * frame, weight_dim)
        #==== Dynamic linear weight generation =====
        if self.args.ada:
            if category == 'support':
                weight = self.generator(task_f) # weight: 1, weight_dim, bias: weight_dim
            else:
                assert weight!=None # If query samples, weight should be given
            score = F.linear(score, weight)

        # Normalize score with min-max
        score = score.view(-1, self.args.sampler_seq_len) # way * shot, frame
        safe_min_value = 1e-4
        norm_score = (score - score.min(dim=1, keepdim=True)[0]) / (score.max(dim=1, keepdim=True)[0]- score.min(dim=1, keepdim=True)[0] + safe_min_value) # way * shot, frame

        if self.training:
            score = norm_score.unsqueeze(-1).expand(-1,-1,self.args.k) # way * shot, frame, k
            topk_func = self.return_topk_func()
            indices = topk_func(score) # way * shot, frame, k
        else:
            indices = self.fast_topk(norm_score)
        #seleted_score = torch.bmm(indices.transpose(-1,-2), )
        indices = indices.transpose(-1,-2) # way * shot, k, frame
        selected_feature = torch.bmm(indices, feature)

        return indices,selected_feature,feature



# tsn-former
class Positional_Encoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(Positional_Encoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self,args):
        super(ScaledDotProductAttention, self).__init__()
        self.args = args

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        score = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(self.args.d_k) # score: [batch_size, n_heads, len_q, len_k]
        att = nn.Softmax(dim=-1)(score) # att: [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(att, V)  # context [batch_size, n_heads, len_q, d_v]
        return context, att

class MultiHeadAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadAttention,self).__init__()
        self.args = args
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False)
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False)
    def forward(self,q,k,v):
        '''
        w_q,w_k,w_v: [batch_size, seq_len, d_model]
        '''
        residual, batch_size = q, q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(q).view(batch_size, -1, self.args.n_heads, self.args.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(k).view(batch_size, -1, self.args.n_heads, self.args.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(v).view(batch_size, -1, self.args.n_heads, self.args.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        content,attention = ScaledDotProductAttention(self.args)(Q,K,V)
        content = content.transpose(1,2).contiguous().view(batch_size, -1, self.args.n_heads * self.args.d_v)
        output = self.fc(content)
        output = nn.LayerNorm(self.args.d_model).cuda()(output + residual)
        return output, attention


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args):
        super(PoswiseFeedForwardNet,self).__init__()
        self.args = args
        self.fc = nn.Sequential(
            nn.Linear(self.args.d_model, self.args.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.args.d_ff, self.args.d_model, bias=False)
        )
    def forward(self,input):
        residual = input 
        output = self.fc(input)

        return nn.LayerNorm(self.args.d_model).cuda()(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self,args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)
    def forward(self,query_sequence):
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        encoder_output, _ = self.enc_self_attn(query_sequence,query_sequence,query_sequence)
        encoder_output = self.pos_ffn(encoder_output)
        return encoder_output

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder,self).__init__()
        self.positionalencoding = Positional_Encoding(args.d_model,0.1)
        self.encoder_layers = nn.ModuleList([EncoderLayer(args) for _ in range(6)])
    def forward(self,query_sequence):
        query_sequence = self.positionalencoding(query_sequence)
        for encoder_layer in self.encoder_layers:
            query_sequence = encoder_layer(query_sequence)
        return query_sequence
        
class Decoderlayer(nn.Module):
    def __init__(self,args):
        super(Decoderlayer,self).__init__()
        self.dec_self_attn = MultiHeadAttention(args)
        self.enc_dec_attention = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self,query_sequence,support_sequence):
        support_sequence, _ = self.dec_self_attn(support_sequence,support_sequence,support_sequence)
        decoder_output, _ = self.enc_dec_attention(query_sequence,support_sequence,support_sequence)
        decoder_output = self.pos_ffn(decoder_output)
        return decoder_output

class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder,self).__init__()
        self.positionalencoding = Positional_Encoding(args.d_model,0.1)
        self.decoder_layers = nn.ModuleList([Decoderlayer(args) for _ in range(6)])
    def forward(self,query_sequence,support_sequence):
        support_sequence = self.positionalencoding(support_sequence)
        for decoder_layer in self.decoder_layers:
            query_sequence = decoder_layer(query_sequence,support_sequence)
        return query_sequence
        

class TSN_Transformer(nn.Module):
    def __init__(self,args):
        super(TSN_Transformer,self).__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        #self.average = nn.AdaptiveMaxPool1d(500)
        #self.projection = nn.Linear(800*512,2)
        self.start_projection = nn.Sequential(nn.Conv1d(args.trans_linear_in_dim,int(args.trans_linear_in_dim/2),1),
                                        nn.Conv1d(int(args.trans_linear_in_dim/2),int(args.trans_linear_in_dim/4),1),
                                        nn.Conv1d(int(args.trans_linear_in_dim/4),1,1))
        self.end_projection = nn.Sequential(nn.Conv1d(args.trans_linear_in_dim,int(args.trans_linear_in_dim/2),1),
                                        nn.Conv1d(int(args.trans_linear_in_dim/2),int(args.trans_linear_in_dim/4),1),
                                        nn.Conv1d(int(args.trans_linear_in_dim/4),1,1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,query_sequence,support_sequence):
        '''
        query_sequence: [batch_size, seq_len, d_model]
        support_sequence: [batch_size, seq_len, d_model]
        '''
        predict_len = query_sequence.shape[1]
        encoder_output = self.encoder(query_sequence)
        out = self.decoder(encoder_output,support_sequence) # [batch_size, seq_len, d_model]
        outs = torch.randn(1,1000,self.args.trans_linear_in_dim).to('cuda')
        if predict_len != 1000:
            outs[:,:predict_len,:] = out
            outs[:,predict_len:,:] = -torch.inf
        #out = self.average(out.transpose(1,2)).transpose(1,2)
        start = self.start_projection(outs.transpose(1,2))
        end = self.end_projection(outs.transpose(1,2))
        #nos = self.reg_number_of_segment(torch.stack((start,end),dim=1).reshape(-1))
        start = self.softmax(start.squeeze(1).squeeze(0)[:predict_len])
        end = self.softmax(end.squeeze(1).squeeze(0)[:predict_len])
        outs = torch.stack((start,end),dim=0)
        #return outs,nos
        return outs


class RSTRM(nn.Module):
    def __init__(self,args):
        super(RSTRM,self).__init__()
        self.args = args
        self.tsn_transformer = TSN_Transformer(args) # [bs,seq_len,channel(dmodel)],[bs,seq_len,channel(dmodel)] -> 2
        self.sampler = Sampler(args) # inuput:[bs,channel,40] -> indices [bs,8,40] * reshape_feature [bs,40,-1] = [bs,8,512,4,4]

        if args.dataset == 'ActivityNet':
            self.class_numbers = 200
        else:
            self.class_numbers = 19

        self.spatial_attention = spatial_attention(self.args.trans_linear_in_dim,self.args.patch_numbers)
        self.support_channel_attention = channel_attention(self.args.trans_linear_in_dim)
        self.query_channel_attention = channel_attention(self.args.trans_linear_in_dim)
        self.averagepool = nn.AdaptiveMaxPool3d((args.sampler_seq_len,None,None))
        self.squeeze_query_patch = nn.Conv3d(self.args.trans_linear_in_dim,self.args.trans_linear_in_dim,(1,4,4))
        self.squeeze_support_patch = nn.Conv3d(self.args.trans_linear_in_dim,self.args.trans_linear_in_dim,(1,4,4))

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

        if args.use_conv:
            self.classfication = nn.Sequential(nn.Conv3d(args.trans_linear_in_dim,512,(1,4,4)),
                                               nn.ReLU())
            self.last = nn.Linear(512*20,self.class_numbers)
        else:
            self.classfication = nn.Linear(int(self.args.classify_len)*16*args.trans_linear_in_dim,self.class_numbers)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self,query_feature,support_feature):
        # dynamic seq_len ??????
        
        query_feature = query_feature.squeeze(0) # [1,channel,seq_len,4,4] -> [channel,seq_len,4,4]
        query_feature = query_feature.reshape(-1,self.args.trans_linear_in_dim,self.args.patch_numbers).transpose(2,1) # [channel,seq_len,4,4] -> [seq_len,4x4,channel]
        query_spatial_attention_feature,query_attn_map = self.spatial_attention(query_feature) # [seq_len,4x4,channel] 160,16,512

        query_spitial_channel_attention_feature = self.query_channel_attention(query_spatial_attention_feature) # [seq_len,16,channel]

        # for classify query feature use sampler
        query_spitial_channel_attention_feature = query_spitial_channel_attention_feature.reshape(self.args.trans_linear_in_dim,-1,4,4).unsqueeze(0) # [1,512,seq_len,4,4]
        average_qf = self.averagepool(query_spitial_channel_attention_feature) # [bs,channel,40,4,4]  seq_len ->100????
        sampler_infea = average_qf.mean(dim=-1).mean(dim=-1) # [bs,channel,40]
        indices,_,_ = self.sampler(sampler_infea) # inuput:[bs,channel,40] -> indices [bs,10,40] * reshape_feature [bs,40,-1] = [bs,10,512,4,4]
        sampled_feature = torch.bmm(indices,average_qf.reshape(1,self.args.sampler_seq_len,-1)).reshape(1,-1,self.args.k,4,4) # [bs,512,10,4,4]

        ## logits
        # for 1 way 1 shot, we only need to tackle the support video once
        support_feature = support_feature.squeeze(0)
        support_feature = support_feature.reshape(-1,self.args.trans_linear_in_dim,self.args.patch_numbers).transpose(2,1) # [channel,seq_len,4,4] -> [seq_len,4x4,channel]
        support_spatial_attention_feature,support_attn_map = self.spatial_attention(support_feature) # [seq_len,4x4,channel]

        # same channel attention  support_spitial_channel_attention_feature = self.support_channel_attention(support_spatial_attention_feature) # [seq_len,16,channel]
        support_spitial_channel_attention_feature = self.query_channel_attention(support_spatial_attention_feature) # [seq_len,16,channel]
        support_seq_len = support_spitial_channel_attention_feature.shape[0]
        # choose 10 seq_len to contact from support feature evenly [seq_len,16,channel] -> [50,16,channel]
        choosed_support_feature = support_spitial_channel_attention_feature[torch.linspace(0, support_seq_len-1, self.args.k).long()] # [seq_len,16,channel] -> [sampler_seq_len,16,channel]

        # contact 10 seq_len query video [10,512,4,4] and support video [seq_len,16,512] for classification
        classify_f = torch.concat((sampled_feature.squeeze(0).reshape(-1,self.args.patch_numbers,self.args.trans_linear_in_dim),choosed_support_feature),dim=0) #[60,16,512] 50 + 50
        # need conv???
        #logits = self.last(self.classfication(classify_f.reshape(self.args.trans_linear_in_dim,-1,4,4).unsqueeze(0)).reshape(-1)) # [1,512,20,4,4] -> [1,512,20,1,1] -> [1,512,20]
        logits = self.classfication(classify_f.reshape(-1))
        # logits = nn.Softmax(dim=-1)(logits) .........fxxx

        # use conv3d instead of mean
        #aug_query_feature = (sampled_feature.mean(dim=2,keepdim=True) + query_spitial_channel_attention_feature).mean(-1).mean(-1) #[1,512,seq_len]
        aug_query_feature = self.squeeze_query_patch((sampled_feature.mean(dim=2,keepdim=True) + query_spitial_channel_attention_feature)).squeeze(-1).squeeze(-1) #[1,512,seq_len,4,4] -> [1,512,seq_len]
        #aug_query_feature = query_spitial_channel_attention_feature.mean(-1).mean(-1) #[1,512,seq_len], add sampled_feature is better
        query_seq_len = aug_query_feature.shape[2]


        # for 1 way n shot
        #aug_query_feature = F.interpolate(aug_query_feature,size=(support_seq_len),mode='nearest') #[1,512,seq_len]
        reg = self.tsn_transformer(aug_query_feature.transpose(1,2),support_spitial_channel_attention_feature.mean(-2).unsqueeze(0))

        # according to the softmaxed reg predict the start and end of the action, find first 30% probability points to classfiy weather it is a action
        start = reg[0]; end = reg[1]
        sorted_start = torch.topk(start,int(query_seq_len),dim=-1)[1]
        sorted_end = torch.topk(end,int(query_seq_len),dim=-1)[1]
        # start_feature = query_spitial_channel_attention_feature[:,:,sorted_start,:,:]
        # end_feature = query_spitial_channel_attention_feature[:,:,sorted_end,:,:]
        for id,idx in enumerate(sorted_start):
            if idx < 3:
                continue
            if self.cosine_similarity(query_spitial_channel_attention_feature[:,:,idx,:,:].reshape(-1),query_spitial_channel_attention_feature[:,:,idx-2,:,:].reshape(-1)) > 0.95:
                sorted_start.pop(id)
        for id,idx in enumerate(sorted_end):
            if idx > query_seq_len - 3:
                continue
            if self.cosine_similarity(query_spitial_channel_attention_feature[:,:,idx,:,:].reshape(-1),query_spitial_channel_attention_feature[:,:,idx+2,:,:].reshape(-1)) > 0.95:
                sorted_end.pop(id)

        return logits,sorted_start,sorted_end,query_attn_map,support_attn_map

### more support feature
if __name__ == '__main__':
    @hydra.main(config_path='config', config_name='config',version_base=None)
    def main(cfg):
        model = RSTRM(cfg).cuda()
        support_imgs = torch.rand(1,512,36,4,4).cuda()
        target_imgs = torch.rand(1,512,250,4,4).cuda()
        a,b,c,*_ = model(target_imgs,support_imgs)
        print(a.shape)
        print(b.shape)
        return

    #seed_everything()
    main()
