import os
import torch
import torch.nn.functional as F

from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler
)

from meshylangelo.vae.sita_vae import SITA_VAE
from meshylangelo.diffusion.denoiser import AttnUnetDenoiser
from meshylangelo.modules.condition_encoders import FrozenCLIPImageGridEmbedder


class Trainer:
    def __init__(
        self,
        outdir = "exp",
        device = "cuda",
        denoiser_ckpt_path = None,
        vae_ckpt_path = "./meshylangelo/vae/checkpoints/shapevae-256.ckpt",
        lr:float=1e-4,
        n_epoch:int=100):
        
        self.outdir = outdir
        print(f"[INFO] initializing trainer, saving results to {self.outdir}...")
        os.makedirs(self.outdir, exist_ok=True)
        
        self.device = device
        self.lr = lr # TODO: set a proper learning rate
        self.n_epoch = n_epoch
        
        # TODO: currently hard coding, maybe consider change to DiT?
        self.denoiser = AttnUnetDenoiser(
            device=None, dtype=None,
            input_channels=64,
            output_channels=64,
            n_ctx=256,
            width=768,
            layers=6,
            heads=12,
            context_dim=1024,
            init_scale=1.0,
            skip_ln=True,
            use_checkpoint=True
        )
        
        if denoiser_ckpt_path is not None:
            print(f"[INFO] denoiser using pretrained checkpoint: {denoiser_ckpt_path}")
            self.denoiser.load_state_dict(
                torch.load(denoiser_ckpt_path, map_location="cpu")
            )
        
        self.denoiser = self.denoiser.to(device=device)
        
        self.condition_encoder = FrozenCLIPImageGridEmbedder(
            version="openai/clip-vit-large-patch14",
            device=device,
            zero_embedding_radio=0.1
        )
        
        self.condition_encoder = self.condition_encoder.to(device=device)
        
        self.first_stage_model = SITA_VAE(
            clip_model_version="openai/clip-vit-large-patch14",
        )
        
        print(f"[INFO] vae using pretrained checkpoint: {vae_ckpt_path}")
        self.first_stage_model.load_state_dict(
            torch.load(vae_ckpt_path, map_location="cpu"), strict=False
        )
        
        self.first_stage_model = self.first_stage_model.to(device=device)
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            variance_type="fixed_small",
            clip_sample=False,
            # prediction_type="epsilon"
        )
        
        self.denoise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            # prediction_type="epsilon"
        )
        
        self.optimizer = torch.optim.AdamW(
            params=list(self.denoiser.parameters()),
            lr=lr,
            betas=[0.9, 0.99],
            eps=1.e-6,
            weight_decay=1.e-2
        )
        
        # TODO: lr scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.6)
        
    def compute_loss(self, model_outputs, split):
        """

        Args:
            model_outputs (dict):
                - x_0:
                - noise:
                - noise_prior:
                - noise_pred:
                - noise_pred_prior:

            split (str):

        Returns:

        """

        pred = model_outputs["pred"]

        if self.noise_scheduler.prediction_type == "epsilon":
            target = model_outputs["noise"]
        elif self.noise_scheduler.prediction_type == "sample":
            target = model_outputs["x_0"]
        else:
            raise NotImplementedError(f"Prediction Type: {self.noise_scheduler.prediction_type} not yet supported.")

        total_loss = F.mse_loss(pred, target, reduction="mean")

        loss_dict = {
            f"{split}/epoch": self.epoch,
            f"{split}/total_loss": total_loss.clone().detach().item()
        }

        return total_loss, loss_dict
    
    def run_step(self, batch:dict):
        if "latents" in batch:
            # TODO: should we add z_scale_factor?
            latents = batch["latents"]
            latents = latents.to(self.device)
        else:
            raise NotImplementedError(f"Data type in batch ({[k for k in batch]}) not yet supported.")
        
        conditions = self.condition_encoder(batch["images"])
        
        # sample noise: [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        
        # sample timesteps:
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        
        # add noise
        noisy_z = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # diffusion model forward
        noise_pred = self.denoiser(noisy_z, timesteps, conditions)
        
        diffusion_outputs = {
            "x_0": noisy_z,
            "noise": noise,
            "pred": noise_pred
        }
        
        return diffusion_outputs
    
    def train_step(self, batch: Dict[str, Union[torch.FloatTensor, List[str]]]):
        
        self.denoiser.train()
        diffusion_outputs = self.run_step(batch)
        loss, loss_dict = self.compute_loss(diffusion_outputs, "train")
        
        for item in loss_dict:
            self.writer.add_scalar(item, loss_dict[item], self.global_iters)
            
        self.pbar.set_postfix({'loss': loss_dict["train/total_loss"]})
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.scheduler.step()
        
    def train(self, dataloader):
        # initialize training save dir
        self.train_outdir = Path(self.outdir) / "training"
        self.checkpoint_dir = self.train_outdir / "checkpoints"
        os.makedirs(self.train_outdir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # initialize summary writer
        self.writer = SummaryWriter(log_dir=self.train_outdir / "tensorboard")
        self.epoch = 0
        self.global_iters = 0
        
        for epoch in range(self.n_epoch):
            self.epoch = epoch
            self.pbar = tqdm(dataloader, desc=f"training epoch {self.epoch}:")
            for batch in self.pbar:
                self.train_step(batch=batch)
                self.global_iters += 1
            
            # save checkpoints
            torch.save(self.denoiser.state_dict(), self.checkpoint_dir / "latest_denoiser.ckpt")
            
    def sample(
        self,
        input: Dict[str, Union[torch.FloatTensor, List[str]]],
        sample_times: int = 1,
        steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        eta: float = 0.0, **kwargs):
        
        if steps is None:
            steps = self.denoise_scheduler.num_inference_steps
            
        do_classifier_free_guidance = guidance_scale > 0
        
        cond = self.condition_encoder(input["images"])
        
        if do_classifier_free_guidance:
            un_cond = self.condition_encoder.unconditional_embedding(batch_size=len(cond))
            cond = torch.cat([un_cond, cond], dim=0)
            
        outputs = []
        latents = None
        
        for _ in range(sample_times):
            sample_loop = ddim_sample(
                self.denoise_scheduler,
                self.denoiser,
                shape=self.first_stage_model.latent_shape,
                cond=cond,
                steps=steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=self.device,
                eta=eta,
                disable_prog=False
            )
            for sample, t in sample_loop:
                latents = sample
            # TODO： write decode, without z scale factor
            decode_out = self.first_stage_model.decode(latents)
            mesh_out = self.first_stage_model.latent2mesh(
                decode_out,
                bounds=1.0,
                octree_depth=8
            )
            outputs += mesh_out
            
        return outputs
    

def ddim_sample(ddim_scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True):

    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2

    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * ddim_scheduler.init_noise_sigma
    # set timesteps
    ddim_scheduler.set_timesteps(steps)
    timesteps = ddim_scheduler.timesteps.to(device)
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, and between [0, 1]
    extra_step_kwargs = {
        "eta": eta,
        "generator": generator
    }

    # reverse
    for i, t in enumerate(tqdm(timesteps, disable=disable_prog, desc="DDIM Sampling:", leave=False)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2)
            if do_classifier_free_guidance
            else latents
        )
        # latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=torch.long, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model.forward(latent_model_input, timestep_tensor, cond)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )
            # text_embeddings_for_guidance = encoder_hidden_states.chunk(
            #     2)[1] if do_classifier_free_guidance else encoder_hidden_states
        # compute the previous noisy sample x_t -> x_t-1
        latents = ddim_scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs
        ).prev_sample

        yield latents, t
        
        
        
        