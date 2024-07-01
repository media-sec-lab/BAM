import torch
import torch.nn as nn
import torch.nn.functional as F
import s3prl.hub as hub
import math

from models.modules.gap import MessageControlGraphAttentionLayer
from models.modules.attention import SelfWeightedPooling
from models.modules.resnet import ResNet1D,BottleNeck

class BAM(nn.Module):
    def __init__(self, args, config) -> None:
        super(BAM, self).__init__()
        self.config = config
        self.args = args

        self.ssl_layer = getattr(hub, config.ssl_name)(ckpt=config.ssl_ckpt, fairseq=True)
        self.att_pool = SelfWeightedPooling(config.embed_dim, num_head=config.pool_head_num, mean_only=True)
        # self.proj = nn.Linear(in_features=config.ssl_feat_dim, out_features=config.embed_dim)
        self.selu = nn.SELU()

        self.inter_layer = inter_frame_attention(in_dim=config.embed_dim, out_dim=config.embed_dim, head_num=config.gap_head_num)
        self.intra_layer = intra_frame_attention(in_channel=config.local_channel_dim, in_dim=config.embed_dim, out_dim=config.embed_dim)

        self.b_output_layer = nn.Sequential(
            nn.Linear(in_features=2*config.embed_dim, out_features=1),
            nn.Sigmoid()
        )

        self.pool_frame_num = int(args.resolution // 0.02)
        
        self.gap_layers= nn.ModuleList([
            MessageControlGraphAttentionLayer(in_dim=config.embed_dim, out_dim=config.embed_dim, head_num=config.gap_head_num)
            for _ in range(config.gap_layer_num)])
        
        self.boundary_proj = nn.Linear(in_features=2*config.embed_dim, out_features=config.embed_dim)
        self.out_layer = nn.Linear(in_features=3*config.embed_dim,out_features=2)


    def forward(self, x, ret_emb=False):
        "x: (bs, frame, feat_dim)"
        x = F.pad(x,(0,256),mode='constant', value=0)
        x = self.ssl_layer(x)["hidden_states"][-1]
        
        b, f, d = x.size()
        x = x.view(-1, self.pool_frame_num, d)
        x = self.att_pool(x)
        x = x.view(b,-1, d * self.config.pool_head_num)
        # x = self.proj(x)

        # boundary
        f_inter = self.inter_layer(x)
        f_intra = self.intra_layer(x)
        b_embedding = torch.cat([f_inter, f_intra], dim=-1)
        b_pred = self.b_output_layer(b_embedding).squeeze(-1)
        binary = torch.where(b_pred.detach() > 0.5, 1, 0)

        # spoof
        s_embedding = x
        for layer in self.gap_layers:
            s_embedding = layer(s_embedding, binary)

        # fusion              
        b_embedding_detach = self.selu(self.boundary_proj(b_embedding.detach()))                                                                                                                                                            
        embedding = torch.cat([s_embedding,b_embedding_detach], dim=-1)
        output = self.out_layer(embedding)

        if ret_emb:
            return embedding
        else:
            return output, b_pred

class inter_frame_attention(nn.Module):
    """ attention among ssl encoder output features."""
    def __init__(self,in_dim, out_dim, head_num):
        super(inter_frame_attention, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, head_num)

        # project
        self.proj_with_att = nn.Linear(head_num * in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # activate
        self.act = nn.SELU(inplace=True)

    def forward(self,x):
        att_map = self._derive_att_map(x)
        att_map = F.softmax(att_map, dim=-2)
        x = self._project(x, att_map)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _project(self, x, att_map):
        x1 = torch.matmul(att_map.permute(0,3,1,2).contiguous(), x.unsqueeze(1).expand(-1,self.head_num,-1,-1))
        x1 = self.proj_with_att(x1.permute(0,2,3,1).flatten(start_dim=2))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        return att_map

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

class intra_frame_attention(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_channel, in_dim, out_dim):
        super(intra_frame_attention, self).__init__()
        self.resnet = ResNet1D(BottleNeck,[2,2,2,2], in_channel=in_channel)
        self.proj_layer = nn.Linear(in_features=1024*int(in_dim/32), out_features=out_dim)

    def forward(self,x):
        m_batch, T, D = x.size()
        out = self.resnet(x.view(-1,1,D))
        out = self.proj_layer(out.view(m_batch,T,-1))
        return out

if __name__ == '__main__':
    print('define of BAM model.')