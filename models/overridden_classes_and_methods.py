import torch
import torch.nn.functional as F

import torch.nn as nn
import math

def overidden_decoder_forward(self, tgt, memory, tgt_mask = None,
                memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None,
                past=None, return_present=False):
    output = tgt
    presents = [] # accumulate over layers
    
    # import pdb
    # pdb.set_trace()
    for i, mod in enumerate(self.layers):
        
        output, present = mod(output, memory, tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        layer_past=past[i, ...] if past is not None else None, return_present=return_present)
        presents.append(present)
    
    if self.norm is not None:
        output = self.norm(output)

    return output, torch.stack(presents) # (n_layer, 2, bsz*num_heads, len, head_dim)


def overridden_decoder_layer_forward(self, tgt, memory, tgt_mask = None,
                memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None,
                layer_past=None, return_present=False):
    '''
    Added `layer_past` and `return_present`.
    The rest are copied and pasted from nn.TransformerDecoderLayer.forward()
    '''
    import pdb
    if return_present: # only at inference
        assert not self.training

    x = tgt
    if self.norm_first:
        out, present_sa = self._sa_block(
            self.norm1(x), tgt_mask, tgt_key_padding_mask, 
            layer_past=layer_past
        )
        x = x + out
        # for mha, keys and values are src_emb, so no need to pass???????
        out, _ = self._mha_block(
            self.norm2(x), memory, memory_mask, memory_key_padding_mask
        )
        # pdb.set_trace()
        x = x + out
        x = x + self._ff_block(self.norm3(x))
        # pdb.set_trace()
    else:
        # import pdb
        # if torch.isnan(torch.sum(x)):
        #     pdb.set_trace()
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, layer_past))
        # if torch.isnan(torch.sum(x)):
        #     pdb.set_trace()
        x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, layer_past))
        # if torch.isnan(torch.sum(x)):
        #     pdb.set_trace()
        x = self.norm3(x + self._ff_block(x))
        # if torch.isnan(torch.sum(x)):
        #     pdb.set_trace()
    return x, present_sa


# self-attention block
def overridden_sa_block(self, x, attn_mask, key_padding_mask, layer_past=None):
    import pdb
    # pdb.set_trace()
    x, present = self.self_attn(query=x, key=x, value=x,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                        need_weights=False, layer_past=layer_past)
    return self.dropout1(x), present



# multihead attention block
def overridden_mha_block(self, x, mem, attn_mask, key_padding_mask, layer_past=None):
    import pdb
    # pdb.set_trace()
    # no need to return present keys and values in this case, since the are from src_emb
    x, _ = self.multihead_attn(query=x, key=mem, value=mem,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False, layer_past=layer_past)
    return self.dropout2(x), _

linear = torch._C._nn.linear # nn.functional 1994

def _in_projection_packed(
    q,
    k,
    v,
    w,
    b,
) -> list[torch.Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    # else:
    #     w_q, w_k, w_v = w.chunk(3)
    #     if b is None:
    #         b_q = b_k = b_v = None
    #     else:
    #         b_q, b_k, b_v = b.chunk(3)
    #     return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


###########################
def clamp_inf(t, clamp_max_only=True):
    '''
    Clamp the tensor values
    '''
    if torch.isinf(t).any():
        clamp_value = torch.finfo(t.dtype).max - 100
        if clamp_max_only:
            t = torch.clamp(t, max=clamp_value) # only clamp the positive infs
        else:
            t = torch.clamp(t, max=clamp_value, min=-clamp_value) # only clamp the positive infs
    return t
###########################


class MultiheadAttention(nn.MultiheadAttention):
    
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None,
                layer_past=None):
        '''
        This function overrides nn.MultiheadAttention.forward() and combines nn.functional.multi_head_attention_forward
        '''
        # q, k, v: (bs, seqlen, dim)
        assert query.dim() == 3, 'query is not batched'
        assert query.shape[-1] == key.shape[-1] == value.shape[-1], 'only supports qkv with the same dimension.'

        if layer_past is not None:
            attn_mask = None

        # class attributes:
        # self.embed_dim, self.num_heads
        embed_dim = self.embed_dim
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight # (3*dim, dim)=(2304, 768)
        in_proj_bias = self.in_proj_bias # (3*dim)=(2304)
        out_proj_weight = self.out_proj.weight
        out_proj_bias = self.out_proj.bias

        bsz = query.shape[0]

        
        # be careful with the dimension. 
        # transpose everything into (seqlen, bs dim) so that we won't mess up things later
        if key is value and query is key: # self attention
            query = key = value = query.transpose(1, 0)

        elif key is value and query is not key: # multihead attention
            query, key = map(lambda x: x.transpose(1,0), [query, key])
            value = key
        else:
            raise NotImplementedError
        
        
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

        # projection, no change in shapes
        # q,k,v, query,key,value: (seqlen, bs, dim)
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias) # 5058
        
        # reshape q, k, v for multihead attention and make em batch first
        import pdb
        # pdb.set_trace()
        q = q.contiguous().view(q.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        # pdb.set_trace()
        #### CAUTION: now shape becomes: (bsz * num_heads, len, head_dim)
        if layer_past is not None:
            assert q.shape[1] == 1, "using layer past should pass a one-step query."
        
        # store present k,v: (2, bsz * num_heads, len, head_dim)
        present = torch.stack([
            k,
            v
        ])

        # concat past k v with present ones
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)


        # set up shape vars
        tgt_len = q.shape[-2]
        src_len = k.shape[-2]

        # check attention mask shape when training/regular inference
        if self.training or (not self.training and layer_past is None):
            if attn_mask is not None: # only for `tgt` `self-attention`
                import pdb
                # pdb.set_trace()
                if attn_mask.dim() == 2:
                    correct_2d_size = (tgt_len, src_len)
                    if attn_mask.shape != correct_2d_size:
                        raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                    attn_mask = attn_mask.unsqueeze(0)
                elif attn_mask.dim() == 3:
                    correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                    if attn_mask.shape != correct_3d_size:
                        raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
                else:
                    raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        
        # when it's training or regular inference, 
        # merge key padding and attention masks
        if self.training or (layer_past is None and not self.training):
            if key_padding_mask is not None:
                assert key_padding_mask.shape == (bsz, src_len), \
                    f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
                key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                    expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
                if attn_mask is None:
                    attn_mask = key_padding_mask
                elif attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask.logical_or(key_padding_mask)
                else:
                    attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
        import pdb

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        # print(attn_mask.shape)
        # pdb.set_trace()

        # adjust dropout probability
        dropout_p = 0.0 if not self.training else self.dropout

        # (deep breath) calculate attention and out projection
        B_nheads, T, head_dim = q.shape
        q_scaled = q / math.sqrt(head_dim)

        # compute attention: (B_nheads, T, head_dim)@(B_nheads, head_dim, T) -> (B_nheads, T, T)
        
        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        # pdb.set_trace()
        attn_output_weights = clamp_inf(attn_output_weights, clamp_max_only=False) # clamp before sending it into softmax
        
        # normalized attention scores along the last dimension
        # pdb.set_trace()
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)
        
        # compute value: (B_nheads, T, T)@(B_nheads, T, head_dim) -> (B_nheads, T, head_dim)
        # pdb.set_trace()
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = clamp_inf(attn_output, clamp_max_only=False) # clamp after matrix multiplication

        # output linear projection
        # pdb.set_trace()
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        # pdb.set_trace()
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = clamp_inf(attn_output, clamp_max_only=False) # clamp after matrix multiplication

        # reshape to (seqlen, bs, dim)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # transpose it back to batch_first:
        attn_output = attn_output.transpose(0,1)      
        
        assert not need_weights, 'currently does not support need_weights=True'

        return attn_output, present # (bs, seqlen, dim), (2, bsz*num_heads, len, head_dim)
