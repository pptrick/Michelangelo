# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizer
from collections import OrderedDict


class AbstractEncoder(nn.Module):
    embedding_dim: int

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class FrozenCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        tokenizer_version=None,
        device="cuda",
        max_length=77,
        zero_embedding_radio: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_version or version)

        self.device = device
        self.max_length = max_length
        self.zero_embedding_radio = zero_embedding_radio

        self.clip_dict = OrderedDict()
        self.clip_name = os.path.split(version)[-1]

        transformer = CLIPModel.from_pretrained(version).text_model

        for param in transformer.parameters():
            param.requires_grad = False
        self.clip_dict[self.clip_name] = transformer

        self._move_flag = False

    @property
    def clip(self):
        return self.clip_dict[self.clip_name]

    def move(self):
        if self._move_flag:
            return

        self.clip_dict[self.clip_name] = self.clip_dict[self.clip_name].to(self.device)
        self._move_flag = True

    def unconditional_embedding(self, batch_size):
        empty_text = [""] * batch_size
        empty_z = self.forward(empty_text)
        return empty_z

    def forward(self, text):
        self.move()

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.clip(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        batch_size = len(text)
        batch_mask = torch.rand((batch_size,))
        for i in range(batch_size):
            if batch_mask[i] < self.zero_embedding_radio:
                text[i] = ""

        return self(text)

class FrozenAlignedCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        tokenizer_version=None,
        device="cuda",
        max_length=77,
        zero_embedding_radio: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_version or version)

        self.device = device
        self.max_length = max_length
        self.zero_embedding_radio = zero_embedding_radio

        self.clip_dict = OrderedDict()
        self.clip_name = os.path.split(version)[-1]

        transformer = CLIPModel.from_pretrained(version).text_model

        for param in transformer.parameters():
            param.requires_grad = False
        self.clip_dict[self.clip_name] = transformer

        self._move_flag = False

    @property
    def clip(self):
        return self.clip_dict[self.clip_name]

    def move(self):
        if self._move_flag:
            return

        self.clip_dict[self.clip_name] = self.clip_dict[self.clip_name].to(self.device)
        self._move_flag = True

    def unconditional_embedding(self, batch_size):
        empty_text = [""] * batch_size
        empty_z = self.forward(empty_text)
        return empty_z

    def forward(self, text):
        self.move()

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.clip(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        batch_size = len(text)
        batch_mask = torch.rand((batch_size,))
        for i in range(batch_size):
            if batch_mask[i] < self.zero_embedding_radio:
                text[i] = ""

        return self(text)


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
            self,
            version="openai/clip-vit-large-patch14",
            device="cuda",
            zero_embedding_radio=0.1,
            normalize_embedding=True,
            num_projection_vector=0,
            linear_mapping_bias=True,
            reverse_visual_projection=False,
    ):
        super().__init__()

        self.device = device

        self.clip_dict = OrderedDict()
        self.clip_name = os.path.split(version)[-1]

        clip_model = CLIPModel.from_pretrained(version)
        clip_model.text_model = None
        clip_model.text_projection = None
        clip_model = clip_model.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.clip_dict[self.clip_name] = clip_model

        self.transform = transforms.Compose(
            [
                transforms.Resize(224, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(224),  # crop a (224, 224) square
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.zero_embedding_radio = zero_embedding_radio

        self.num_projection_vector = num_projection_vector
        self.reverse_visual_projection = reverse_visual_projection
        self.normalize_embedding = normalize_embedding

        embedding_dim = (
            clip_model.visual_projection.in_features
            if reverse_visual_projection
            else clip_model.visual_projection.out_features
        )
        self.embedding_dim = embedding_dim
        if self.num_projection_vector > 0:
            self.projection = nn.Linear(
                embedding_dim,
                clip_model.visual_projection.out_features * num_projection_vector,
                bias=linear_mapping_bias,
            )
            nn.init.normal_(self.projection.weight, std=embedding_dim ** -0.5)

        self._move_flag = False

    @property
    def clip(self):
        return self.clip_dict[self.clip_name]

    def unconditional_embedding(self, batch_size):
        zero = torch.zeros(
            batch_size,
            1,
            self.embedding_dim,
            device=self.device,
            dtype=self.clip.visual_projection.weight.dtype,
        )
        if self.num_projection_vector > 0:
            zero = self.projection(zero).view(batch_size, self.num_projection_vector, -1)
        return zero

    def forward(self, image, value_range=(-1, 1), zero_embedding_radio=0):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.device, dtype=self.clip.visual_projection.weight.dtype)

        if self.reverse_visual_projection:
            z = self.clip.vision_model(self.transform(image))[1]
        else:
            z = self.clip.get_image_features(self.transform(image))

        if self.normalize_embedding:
            z = z / z.norm(dim=-1, keepdim=True)
        if z.ndim == 2:
            z = z.unsqueeze(dim=-2)

        if zero_embedding_radio > 0:
            mask = torch.rand((len(image), 1, 1), device=z.device, dtype=z.dtype) < zero_embedding_radio
            z = z * mask.to(z)

        if self.num_projection_vector > 0:
            z = self.projection(z).view(len(image), self.num_projection_vector, -1)

        return z

    def move(self):
        if self._move_flag:
            return

        self.clip_dict[self.clip_name] = self.clip_dict[self.clip_name].to(self.device)
        self._move_flag = True

    def encode(self, image):
        self.move()
        return self(image, zero_embedding_radio=self.zero_embedding_radio)


class FrozenCLIPImageGridEmbedder(AbstractEncoder):

    def __init__(
            self,
            version="openai/clip-vit-large-patch14",
            device="cuda",
            zero_embedding_radio=0.1,
    ):
        super().__init__()

        self.device = device

        self.clip_dict = OrderedDict()
        self.clip_name = os.path.split(version)[-1]

        clip_model: CLIPModel = CLIPModel.from_pretrained(version)
        clip_model.text_model = None
        clip_model.text_projection = None
        clip_model = clip_model.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.clip_dict[self.clip_name] = clip_model

        self.transform = transforms.Compose(
            [
                transforms.Resize(224, transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(224),  # crop a (224, 224) square
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.zero_embedding_radio = zero_embedding_radio
        self.embedding_dim = clip_model.vision_embed_dim

        self._move_flag = False

    @property
    def clip(self):
        return self.clip_dict[self.clip_name]

    def move(self):
        if self._move_flag:
            return

        self.clip_dict[self.clip_name] = self.clip_dict[self.clip_name].to(self.device)
        self._move_flag = True

    def unconditional_embedding(self, batch_size):
        zero = torch.zeros(
            batch_size,
            self.clip.vision_model.embeddings.num_positions,
            self.embedding_dim,
            device=self.device,
            dtype=self.clip.visual_projection.weight.dtype,
        )
        return zero

    def forward(self, image, value_range=(-1, 1), zero_embedding_radio=0):
        self.move()

        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.device, dtype=self.clip.visual_projection.weight.dtype)

        z = self.clip.vision_model(self.transform(image)).last_hidden_state

        if zero_embedding_radio > 0:
            mask = torch.rand((len(image), 1, 1), device=z.device, dtype=z.dtype) >= zero_embedding_radio
            z = z * mask.to(z)

        return z

    def encode(self, image):
        return self(image, zero_embedding_radio=self.zero_embedding_radio)
