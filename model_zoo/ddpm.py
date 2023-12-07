# Based on Inferer module from MONAI:
# -----------------------------------------------------------------------------------------------
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from net_utils.simplex_noise import generate_noise

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from net_utils.diffusion_unet import DiffusionModelUNet
from net_utils.schedulers.ddpm import DDPMScheduler
from net_utils.schedulers.ddim import DDIMScheduler

from skimage import exposure
from scipy.ndimage.filters import gaussian_filter
import lpips
import cv2

from tqdm import tqdm
has_tqdm = True


class AnomalyMap():
    def __init__(self):

        self.device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True,
                                     spatial=True, lpips=True).to(self.device)
        super(AnomalyMap, self).__init__()

    def dilate_masks(self, masks):
        """
        :param masks: masks to dilate
        :return: dilated masks
        """
        kernel = np.ones((3, 3), np.uint8)

        dilated_masks = torch.zeros_like(masks)
        for i in range(masks.shape[0]):
            mask = masks[i][0].detach().cpu().numpy()
            if np.sum(mask) < 1:
                dilated_masks[i] = masks[i]
                continue
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            dilated_mask = torch.from_numpy(dilated_mask).to(masks.device).unsqueeze(dim=0)
            dilated_masks[i] = dilated_mask

        return dilated_masks

    def compute_residual(self, x_rec, x, hist_eq=False):
        """
        :param x_rec: reconstructed image
        :param x: original image
        :param hist_eq: whether to perform histogram equalization
        :return: residual image
        """
        if hist_eq:
            x_rescale = exposure.equalize_adapthist(x.cpu().detach().numpy())
            x_rec_rescale = exposure.equalize_adapthist(x_rec.cpu().detach().numpy())
            x_res = np.abs(x_rec_rescale - x_rescale)
        else:
            x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())

        return x_res

    def lpips_loss(self, anomaly_img, ph_img, retPerLayer=False):
        """
        :param anomaly_img: anomaly image
        :param ph_img: pseudo-healthy image
        :param retPerLayer: whether to return the loss per layer
        :return: LPIPS loss
        """
        if len(ph_img.shape) < 2:
            print('Image should have 2 dimensions at lease (LPIPS)')
            return
        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
        if len(ph_img.shape) == 3:
            ph_img = torch.unsqueeze(ph_img, 0)
            anomaly_img = torch.unsqueeze(anomaly_img, 0)

        saliency_maps = []
        for batch_id in range(anomaly_img.size(0)):
            lpips = self.l_pips_sq(2*anomaly_img[batch_id:batch_id + 1, :, :, :]-1, 2*ph_img[batch_id:batch_id + 1, :, :, :]-1,
                                   normalize=True, retPerLayer=retPerLayer)
            if retPerLayer:
                lpips = lpips[1][0]
            saliency_maps.append(lpips[0,:,:,:].cpu().detach().numpy())
        return np.asarray(saliency_maps)


class DDPM(nn.Module):

    def __init__(self, spatial_dims=2,
                 in_channels=1,
                 out_channels=1,
                 num_channels=(128, 256, 256),
                 attention_levels=(False, True, True),
                 num_res_blocks=1,
                 num_head_channels=256,
                 method="autoDDPM",
                 train_scheduler="ddpm",
                 inference_scheduler="ddpm",
                 inference_steps=1000,
                 noise_level_recon=300,
                 noise_level_inpaint=50,
                 noise_type="gaussian",
                 prediction_type="epsilon",
                 resample_steps=4,
                 masking_threshold=-1,
                 threshold_low=1,
                 threshold_high=10000,
                 image_path="",):
        super().__init__()
        self.unet = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
        )
        self.method = method
        self.noise_level_recon = noise_level_recon
        self.noise_level_inpaint = noise_level_inpaint
        self.prediction_type = prediction_type
        self.resample_steps = resample_steps
        self.masking_threshold = masking_threshold
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.image_path = image_path

        # set up scheduler and timesteps
        if train_scheduler == "ddpm":
            self.train_scheduler = DDPMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)
        else:
            raise NotImplementedError(f"{train_scheduler} does is not implemented for {self.__class__}")

        if inference_scheduler == "ddim":
            self.inference_scheduler = DDIMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)
        else:
            self.inference_scheduler = DDPMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)

        self.inference_scheduler.set_timesteps(inference_steps)
        self.device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

        self.ano_map = AnomalyMap()

    def forward(self, inputs, noise=None, timesteps=None, condition=None):
        # only for torch_summary to work
        if noise is None:
            noise = torch.randn_like(inputs)
        if timesteps is None:
            timesteps = torch.randint(0, self.train_scheduler.num_train_timesteps,
                                      (inputs.shape[0],), device=inputs.device).long()

        noisy_image = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=timesteps)
        return self.unet(x=noisy_image, timesteps=timesteps, context=condition)

    @torch.no_grad()
    def get_anomaly(self, inputs: torch.Tensor,
                    noise_level: int | None = 500,
                    intermediate_steps: int | None = 100,
                    conditioning: torch.Tensor | None = None,
                    save_intermediates: bool | None = False,
                    verbose: bool = False,
                    method: str = 'autoDDPM'):
        assert method == 'anoDDPM' or method =='autoDDPM', 'Method should be either anoDDPM or autoDDPM'
        if method == 'anoDDPM':
            x_rec, _ = self.sample_from_image(inputs, noise_level=noise_level, verbose=verbose)
            x_rec = torch.clamp(x_rec, 0, 1)
            anomaly_maps = np.abs(inputs.cpu().detach().numpy() - x_rec.cpu().detach().numpy())
            anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)
            x_rec_dict = {'x_rec': x_rec}
        else:
            anomaly_maps, anomaly_scores, x_rec_dict = self.get_autoDDPM_anomaly(inputs=inputs,
                                                                                  noise_level_recon=noise_level,
                                                                                  noise_level_inpaint=self.noise_level_inpaint,
                                                                                  save_intermediates=save_intermediates,
                                                                                  verbose=verbose)
        return anomaly_maps, anomaly_scores, x_rec_dict

    @torch.no_grad()
    def get_autoDDPM_anomaly(self, inputs: torch.Tensor,
                             noise_level_recon: int | None = 200,
                             noise_level_inpaint: int | None = 50,
                             save_intermediates: bool | None = False,
                             verbose: bool = False):

        x_rec, _ = self.sample_from_image(inputs, noise_level=noise_level_recon,
                                                save_intermediates=save_intermediates, verbose=verbose)
        x_rec = torch.clamp(x_rec, 0, 1)
        x_res = self.ano_map.compute_residual(inputs, x_rec, hist_eq=False)
        lpips_mask = self.ano_map.lpips_loss(inputs, x_rec, retPerLayer=False)
        #
        # anomalous: high value, healthy: low value
        x_res = np.asarray([(x_res[i] / np.percentile(x_res[i], 95)) for i in range(x_res.shape[0])]).clip(0, 1)
        combined_mask_np = lpips_mask * x_res
        combined_mask = torch.Tensor(combined_mask_np).to(self.device)
        masking_threshold = self.masking_threshold if self.masking_threshold >=0 else torch.tensor(np.asarray([(
            np.percentile(combined_mask[i].cpu().detach().numpy(), 95)) for i in range(combined_mask.shape[0])]).clip(0,
                                                                                                                   1))
        combined_mask_binary = torch.cat([torch.where(combined_mask[i] > masking_threshold[i], torch.ones_like(
            torch.unsqueeze(combined_mask[i],0)), torch.zeros_like(combined_mask[i]))
                                          for i in range(combined_mask.shape[0])], dim=0)

        combined_mask_binary_dilated = self.ano_map.dilate_masks(combined_mask_binary)
        mask_in_use = combined_mask_binary_dilated

        # In-painting setup
        # 1. Mask the original image (get rid of anomalies) and the reconstructed image (keep pseudo-healthy
        # counterparts)
        x_masked = (1 - mask_in_use) * inputs
        x_rec_masked = mask_in_use * x_rec
        #
        #
        # 2. Start in-painting with reconstructed image and not pure noise
        noise = torch.randn_like(x_rec, device=self.device)
        timesteps = torch.full([inputs.shape[0]], noise_level_inpaint, device=self.device).long()
        inpaint_image = self.inference_scheduler.add_noise(
            original_samples=x_rec, noise=noise, timesteps=timesteps
        )

        # 3. Setup for loop
        timesteps = self.inference_scheduler.get_timesteps(noise_level_inpaint)
        progress_bar = iter(timesteps)
        num_resample_steps = self.resample_steps
        stitched_images = []

        # 4. Inpainting loop
        with torch.no_grad():
            with autocast(enabled=True):
                for t in progress_bar:
                    for u in range(num_resample_steps):
                        # 4a) Get the known portion at t-1
                        if t > 0:
                            noise = torch.randn_like(inputs, device=self.device)
                            timesteps_prev = torch.full([inputs.shape[0]], t - 1, device=self.device).long()
                            noised_masked_original_context = self.inference_scheduler.add_noise(
                                original_samples=x_masked, noise=noise, timesteps=timesteps_prev
                            )
                        else:
                            noised_masked_original_context = x_masked
                        #
                        # 4b) Perform a denoising step to get the unknown portion at t-1
                        if t > 0:
                            timesteps = torch.full([inputs.shape[0]], t, device=self.device).long()
                            model_output = self.unet(x=inpaint_image, timesteps=timesteps)
                            inpainted_from_x_rec, _ = self.inference_scheduler.step(model_output, t,
                                                                                          inpaint_image)
                        #
                        # 4c) Combine the known and unknown portions at t-1
                        inpaint_image = torch.where(
                            mask_in_use == 1, inpainted_from_x_rec, noised_masked_original_context
                        )

                        ## 4d) Perform resampling: sample x_t from x_t-1 -> get new image to be inpainted
                        # in the masked region
                        if t > 0 and u < (num_resample_steps - 1):
                            inpaint_image = (
                                    torch.sqrt(1 - self.inference_scheduler.betas[t - 1]) * inpaint_image
                                    + torch.sqrt(self.inference_scheduler.betas[t - 1])
                                    * torch.randn_like(inputs, device=self.device)
                            )

        final_inpainted_image = inpaint_image
        x_res_2 = self.ano_map.compute_residual(inputs, final_inpainted_image.clamp(0, 1), hist_eq=False)
        x_lpips_2 = self.ano_map.lpips_loss(inputs, final_inpainted_image, retPerLayer=False)
        anomaly_maps = x_res_2 * combined_mask.cpu().detach().numpy()
        anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)

        return anomaly_maps, anomaly_scores, {'x_rec_orig': x_rec, 'x_res_orig': combined_mask,
                                              'mask': mask_in_use, 'stitch': x_masked + x_rec_masked,
                                              'x_rec': final_inpainted_image}

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        noise_level: int | None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            noise_level: noising step until which noise is added before sampling
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        image = input_noise
        timesteps = self.inference_scheduler.get_timesteps(noise_level)
        if verbose and has_tqdm:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            model_output = self.unet(
                image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning
            )

            # 2. compute previous image: x_t -> x_t-1
            image, _ = self.inference_scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image

    @torch.no_grad()
    # function to noise and then sample from the noise given an image to get healthy reconstructions of anomalous input images
    def sample_from_image(
        self,
        inputs: torch.Tensor,
        noise_level: int | None = 500,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Sample to specified noise level and use this as noisy input to sample back.
        Args:
            inputs: input images, NxCxHxW[xD]
            noise_level: noising step until which noise is added before 
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        noise = generate_noise(
            self.train_scheduler.noise_type, inputs, noise_level)

        t = torch.full((inputs.shape[0],),
                       noise_level, device=inputs.device).long()
        noised_image = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=t)
        image = self.sample(input_noise=noised_image, noise_level=noise_level, save_intermediates=save_intermediates,
                            intermediate_steps=intermediate_steps, conditioning=conditioning, verbose=verbose)
        return image, {'z': None}

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods for an input.
        Args:
            inputs: input images, NxCxHxW[xD]
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
        """

        if self.train_scheduler._get_name() != "DDPMScheduler":
            raise NotImplementedError(
                f"Likelihood computation is only compatible with DDPMScheduler,"
                f" you are using {self.train_scheduler._get_name()}"
            )
        if verbose and has_tqdm:
            progress_bar = tqdm(self.train_scheduler.timesteps)
        else:
            progress_bar = iter(self.train_scheduler.timesteps)
        intermediates = []
        total_kl = torch.zeros(inputs.shape[0]).to(inputs.device)
        for t in progress_bar:
            # Does this change things if we use different noise for every step?? before it was just one gaussian noise for all steps
            noise = generate_noise(self.train_scheduler.noise_type, inputs, t)

            timesteps = torch.full(
                inputs.shape[:1], t, device=inputs.device).long()
            noisy_image = self.train_scheduler.add_noise(
                original_samples=inputs, noise=noise, timesteps=timesteps)
            model_output = self.unet(
                x=noisy_image, timesteps=timesteps, context=conditioning)
            # get the model's predicted mean, and variance if it is predicted
            if model_output.shape[1] == inputs.shape[1] * 2 and self.train_scheduler.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(
                    model_output, inputs.shape[1], dim=1)
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = self.train_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.train_scheduler.alphas_cumprod[t -
                                                                    1] if t > 0 else self.train_scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if self.train_scheduler.prediction_type == "epsilon":
                pred_original_sample = (
                    noisy_image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif self.train_scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif self.train_scheduler.prediction_type == "v_prediction":
                pred_original_sample = (
                    alpha_prod_t**0.5) * noisy_image - (beta_prod_t**0.5) * model_output
            # 3. Clip "predicted x_0"
            if self.train_scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (
                alpha_prod_t_prev ** (0.5) * self.train_scheduler.betas[t]) / beta_prod_t
            current_sample_coeff = self.train_scheduler.alphas[t] ** (
                0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = pred_original_sample_coeff * \
                pred_original_sample + current_sample_coeff * noisy_image

            # get the posterior mean and variance
            posterior_mean = self.train_scheduler._get_mean(
                timestep=t, x_0=inputs, x_t=noisy_image)
            posterior_variance = self.train_scheduler._get_variance(
                timestep=t, predicted_variance=predicted_variance)

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = torch.log(
                predicted_variance) if predicted_variance else log_posterior_variance

            if t == 0:
                # compute -log p(x_0|x_1)
                kl = -self._get_decoder_log_likelihood(
                    inputs=inputs,
                    means=predicted_mean,
                    log_scales=0.5 * log_predicted_variance,
                    original_input_range=original_input_range,
                    scaled_input_range=scaled_input_range,
                )
            else:
                # compute kl between two normals
                kl = 0.5 * (
                    -1.0
                    + log_predicted_variance
                    - log_posterior_variance
                    + torch.exp(log_posterior_variance -
                                log_predicted_variance)
                    + ((posterior_mean - predicted_mean) ** 2) *
                    torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(axis=1)
            if save_intermediates:
                intermediates.append(kl.cpu())

        if save_intermediates:
            return total_kl, intermediates
        else:
            return total_kl

    def _approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal. Code adapted from https://github.com/openai/improved-diffusion.
        """

        return 0.5 * (
            1.0 + torch.tanh(torch.sqrt(torch.Tensor([2.0 / math.pi]).to(
                x.device)) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def _get_decoder_log_likelihood(
        self,
        inputs: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image. Code adapted from https://github.com/openai/improved-diffusion.
        Args:
            input: the target images. It is assumed that this was uint8 values,
                        rescaled to the range [-1, 1].
            means: the Gaussian mean Tensor.
            log_scales: the Gaussian log stddev Tensor.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
        """
        assert inputs.shape == means.shape
        bin_width = (scaled_input_range[1] - scaled_input_range[0]) / (
            original_input_range[1] - original_input_range[0]
        )
        centered_x = inputs - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + bin_width / 2)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - bin_width / 2)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            inputs < -0.999,
            log_cdf_plus,
            torch.where(inputs > 0.999, log_one_minus_cdf_min,
                        torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == inputs.shape
        return log_probs
