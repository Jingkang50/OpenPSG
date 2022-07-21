import torch
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER


@TRANSFORMER.register_module()
class DualTransformer(BaseModule):
    """Modify the DETR transformer with two decoders.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder1 ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        decoder2 ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """
    def __init__(self,
                 encoder=None,
                 decoder1=None,
                 decoder2=None,
                 init_cfg=None):
        super(DualTransformer, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder1 = build_transformer_layer_sequence(decoder1)
        self.decoder2 = build_transformer_layer_sequence(decoder2)
        self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query1_embed, query2_embed, pos_embed):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoders,
                with shape [bs, h, w].
            query1_embed (Tensor): The first query embedding for decoder, with
                shape [num_query, c].
            query2_embed (Tensor): The second query embedding for decoder, with
                shape [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoders, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query1_embed = query1_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        query2_embed = query2_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(query=x,
                              key=None,
                              value=None,
                              query_pos=pos_embed,
                              query_key_padding_mask=mask)
        target1 = torch.zeros_like(query1_embed)
        target2 = torch.zeros_like(query2_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        # first decoder
        out_dec1 = self.decoder1(query=target1,
                                 key=memory,
                                 value=memory,
                                 key_pos=pos_embed,
                                 query_pos=query1_embed,
                                 key_padding_mask=mask)
        out_dec1 = out_dec1.transpose(1, 2)
        # second decoder
        out_dec2 = self.decoder2(query=target2,
                                 key=memory,
                                 value=memory,
                                 key_pos=pos_embed,
                                 query_pos=query2_embed,
                                 key_padding_mask=mask)
        out_dec2 = out_dec2.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return out_dec1, out_dec2, memory
