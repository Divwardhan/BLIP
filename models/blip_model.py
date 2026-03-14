import torch
import torch.nn as nn

from models.vision_encoder.vision_encoder import VisionEncoder
from models.text_encoder.text_encoder import TextEncoder
from models.multimodal_encoder.multimodal_encoder import MultiModalEncoder
from models.text_decoder.text_decoder import TextDecoder

from models.heads.contrastive_head import ContrastiveHead
from models.heads.itm_head import ITMHead


class BLIP(nn.Module):

    def __init__(
        self,
        vocab_size=30522,
        image_size=224,
        embed_dim=768
    ):
        super().__init__()

        # encoders
        self.vision_encoder = VisionEncoder(img_size=image_size)
        self.text_encoder = TextEncoder(vocab_size=vocab_size)

        # multimodal fusion
        self.multimodal_encoder = MultiModalEncoder()

        # caption decoder
        self.text_decoder = TextDecoder(vocab_size=vocab_size)

        # training heads
        self.contrastive_head = ContrastiveHead(embed_dim)
        self.itm_head = ITMHead(embed_dim)

    def forward(self, images, input_ids):

        # --------------------------------
        # Vision Encoder
        # --------------------------------

        image_cls, image_tokens = self.vision_encoder(images)

        # --------------------------------
        # Text Encoder
        # --------------------------------

        text_cls, text_tokens = self.text_encoder(input_ids)

        # --------------------------------
        # Contrastive Branch
        # --------------------------------

        contrastive_logits = self.contrastive_head(
            image_cls,
            text_cls
        )

        # --------------------------------
        # Multimodal Encoder (ITM)
        # --------------------------------

        multimodal_cls, _ = self.multimodal_encoder(
            text_tokens,
            image_tokens
        )

        itm_logits = self.itm_head(multimodal_cls)

        # --------------------------------
        # Caption Decoder
        # --------------------------------

        caption_logits = self.text_decoder(
            input_ids,
            image_tokens
        )

        return {
            "contrastive_logits": contrastive_logits,
            "itm_logits": itm_logits,
            "caption_logits": caption_logits
        }