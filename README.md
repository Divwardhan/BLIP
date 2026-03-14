# BLIP Architecture Overview

BLIP (Bootstrapping Language-Image Pretraining) is a Vision-Language Model designed for tasks such as image captioning, visual question answering, and image-text retrieval.
The architecture combines a vision encoder, text encoder, multimodal encoder, and text decoder to support multiple training objectives.

## Architecture Flow

```
                IMAGE
                  │
                  ▼
          Vision Encoder (ViT)
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
   Image Emb   Image Tokens   Image Tokens
       │            │               │
       │            │               │
       │      Multimodal Encoder   Text Decoder
       │            │             (Captioning)
       │            │               │
Text Encoder        │               │
       │            │               │
       ▼            ▼               ▼
 Contrastive     ITM Head      Caption Output
    Loss       (Match / No)   (Language Loss)
```

## Components

### Vision Encoder (ViT)

* Takes an input image
* Splits the image into patches
* Converts patches into embeddings using a Vision Transformer
* Produces image tokens and a global image embedding

### Text Encoder

* Processes input text using a transformer encoder
* Produces text embeddings used for contrastive learning and multimodal fusion

### Multimodal Encoder

* Combines image tokens and text tokens
* Uses cross-attention to learn joint representations
* Used for image-text matching tasks

### Text Decoder

* A transformer-based decoder
* Generates text autoregressively from image representations
* Used for image captioning

## Training Objectives

BLIP is trained using multiple objectives simultaneously.

### Image-Text Contrastive Loss (ITC)

Aligns image embeddings with text embeddings so matching pairs are close in embedding space.

### Image-Text Matching Loss (ITM)

Binary classification task predicting whether an image and text correspond.

### Language Modeling Loss (LM)

Used by the text decoder to generate captions word-by-word.

## Supported Tasks

* Image Captioning
* Image-Text Retrieval
* Visual Question Answering (VQA)
* Image-Text Matching

## Key Idea

BLIP uses a unified architecture with shared vision and language modules, enabling it to handle multiple vision-language tasks with a single pretrained model.



## File Structure 
```
blip/

├── models/
│
│   ├── vision_encoder/
│   │   ├── patch_embedding.py
│   │   ├── vit_block.py
│   │   └── vision_encoder.py
│
│   ├── text_encoder/
│   │   ├── text_embeddings.py
│   │   ├── transformer_block.py
│   │   └── text_encoder.py
│
│   ├── multimodal_encoder/
│   │   ├── cross_attention.py
│   │   ├── multimodal_block.py
│   │   └── multimodal_encoder.py
│
│   ├── text_decoder/
│   │   ├── decoder_block.py
│   │   └── text_decoder.py
│
│   ├── heads/
│   │   ├── contrastive_head.py
│   │   └── itm_head.py
│
│   └── blip_model.py
│
├── losses/
│   ├── contrastive_loss.py
│   ├── itm_loss.py
│   └── caption_loss.py
│
├── datasets/
│   └── image_caption_dataset.py
│
├── utils/
│   ├── positional_encoding.py
│   ├── attention_masks.py
│   └── projection_layers.py
│
├── configs/
│   └── blip_config.py
│
├── train.py
└── inference.py
```