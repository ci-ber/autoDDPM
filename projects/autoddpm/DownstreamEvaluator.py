import logging
import io
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns
import umap.umap_ as umap
#
from torch.nn import L1Loss
from torch.cuda.amp import autocast
from torchvision.transforms import transforms
from torchvision.utils import save_image
import torch.nn.functional as F

#
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter

from PIL import Image
import cv2
import numpy as np
#
import lpips
#
from dl_utils import *
from optim.metrics import *
from optim.losses.image_losses import NCC
from core.DownstreamEvaluator import DownstreamEvaluator
import os
import copy
from model_zoo import VGGEncoder
from optim.losses.image_losses import CosineSimLoss


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_= True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)
        self.l_cos = CosineSimLoss(device='cuda')
        self.l_ncc = NCC(win=[9, 9])

        # self.l_pips_vgg = lpips.LPIPS(pretrained=True, net='vgg', use_dropout=False, eval_mode=False, spatial=False, lpips=True).to(self.device)
        # self.l_pips_alex = lpips.LPIPS(pretrained=True, net='alex', use_dropout=False, eval_mode=False, spatial=False, lpips=True).to(self.device)

        self.global_= True

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        self.pathology_localization(global_model, self.model.masking_threshold)

    def _log_visualization(self, to_visualize, i, count):
        """
        Helper function to log images and masks to wandb
        :param: to_visualize: list of dicts of images and their configs to be visualized
            dict needs to include:
            - tensor: image tensor
            dict may include:
            - title: title of image
            - cmap: matplotlib colormap name
            - vmin: minimum value for colorbar
            - vmax: maximum value for colorbar
        :param: epoch: current epoch
        """
        diffp, axarr = plt.subplots(1, len(to_visualize), gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(len(to_visualize) * 4, 4))
        for idx, dict in enumerate(to_visualize):
            if 'title' in dict:
                axarr[idx].set_title(dict['title'])
            axarr[idx].axis('off')
            tensor = dict['tensor'][i].cpu().detach().numpy().squeeze() if isinstance(dict['tensor'], torch.Tensor) else dict['tensor'][i].squeeze()
            axarr[idx].imshow(tensor, cmap=dict.get('cmap', 'gray'), vmin=dict.get('vmin', 0), vmax=dict.get('vmax', 1))
        diffp.set_size_inches(len(to_visualize) * 4, 4)

        wandb.log({f'Anomaly_masks/Example_Atlas_{count}': [wandb.Image(diffp, caption="Atlas_" + str(count))]})

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
        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)

        loss_lpips = self.l_pips_sq(anomaly_img, ph_img, normalize=True, retPerLayer=retPerLayer)
        if retPerLayer:
            loss_lpips = loss_lpips[1][0]
        return loss_lpips.cpu().detach().numpy()

    def find_mask_size_thresholds(self, dataset):
        """
        :param dataset: dataset to find mask size thresholds
        :return: lower and upper tail thresholds
        """
        mask_sizes = []
        for _, data in enumerate(dataset):
            if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
            else:
                    data0 = data[0]
            x = data0.to(self.device)
            masks = data[1].to(self.device)
            masks[masks>0] = 1
            
            for i in range(len(x)):
                if torch.sum(masks[i][0]) > 1:
                    mask_sizes.append(torch.sum(masks[i][0]).item())
                    
        unique_mask_sizes = np.unique(mask_sizes)
        print(type(unique_mask_sizes))
        lower_tail_threshold = np.percentile(unique_mask_sizes, 25)
        upper_tail_threshold = np.percentile(unique_mask_sizes, 75)

        _ = plt.figure()
        # plt.figure()
        plt.hist(mask_sizes, bins=100)
        plt.xlabel('Mask Sizes')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mask Sizes')

        plt.axvline(lower_tail_threshold, color='r', linestyle='--', label=f'25th Percentile: {lower_tail_threshold}')
        plt.axvline(upper_tail_threshold, color='g', linestyle='--', label=f'75th Percentile: {upper_tail_threshold}')
        print(lower_tail_threshold, upper_tail_threshold)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        wandb.log({"Anomaly/Mask sizes1": [wandb.Image(Image.open(buf), caption="Mask Sizes")]})

        plt.clf()

    def pathology_localization(self, global_model, th):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("################ MANIFOLD LEARNING TEST #################")
        lpips_alex = lpips.LPIPS(net='alex')

        self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MAE': [],
            'LPIPS': [],
            'SSIM': [],
        }
        pred_dict = dict()

        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MAE': [],
                'LPIPS': [],
                'SSIM': [],
            }
            global_counter = 0
            threshold_masks = []
            anomalous_pred = []
            healthy_pred = []

            logging.info('DATASET: {}'.format(dataset_key))

            for idx, data in enumerate(dataset):

                # Call this to get the mask size thresholds for the dataset
                # self.find_mask_size_thresholds(dataset)

                # New per batch
                to_visualize = []
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks>0] = 1
            
                x_rec = torch.zeros_like(x)
                if not os.path.exists(os.path.join(os.path.dirname(self.model.image_path), f'rec_{self.model.noise_level_recon}', f'image_{global_counter}.png')):
                        os.makedirs(os.path.join(os.path.dirname(self.model.image_path), f'rec_{self.model.noise_level_recon}'), exist_ok=True)
                        os.makedirs(os.path.join(os.path.dirname(self.model.image_path), f'original_{self.model.noise_level_recon}'), exist_ok=True)
                        x_rec, _ = self.model.sample_from_image(x, noise_level=self.model.noise_level_recon)
                        x_rec = torch.clamp(x_rec, 0, 1)
                for i in range(x.shape[0]):
                    path_to_rec = os.path.join(os.path.dirname(self.model.image_path), f'rec_{self.model.noise_level_recon}', f'image_{global_counter}.png')
                    path_to_image = os.path.join(os.path.dirname(self.model.image_path), f'original_{self.model.noise_level_recon}', f'image_{global_counter}.png')
                    global_counter += 1
                    if os.path.exists(path_to_rec):
                        # load reconstructed images
                        noised = Image.open(path_to_rec).convert('L')
                        x_rec[i] = transforms.ToTensor()(noised)
                        # load original images
                        orig = Image.open(path_to_image).convert('L')
                        x[i] = transforms.ToTensor()(orig)
                    else:
                        save_image(x_rec[i], path_to_rec)
                        save_image(x[i], path_to_image)
                        print("Saved image at path:", path_to_rec)

                x_res = self.compute_residual(x, x_rec, hist_eq=False)
                lpips_mask = self.lpips_loss(x, x_rec, retPerLayer=False)

                # anomalous: high value, healthy: low value
                x_res = np.asarray([ (x_res[i] / np.percentile(x_res[i], 95)) for i in range(x_res.shape[0]) ]).clip(0, 1)
                combined_mask_np = lpips_mask * x_res
                combined_mask = torch.Tensor(combined_mask_np).to(self.device)
                combined_mask_binary = torch.where(combined_mask > th, torch.ones_like(combined_mask), torch.zeros_like(combined_mask))
                combined_mask_binary_dilated = self.dilate_masks(combined_mask_binary)
                mask_in_use = combined_mask_binary_dilated

                to_visualize = [
                    {'title': f'l1/95th perc {x_res.max():.3f}', 'tensor': x_res, 'cmap': 'plasma'}, 
                    {'title': f'lpips {lpips_mask.max():.3f}', 'tensor': lpips_mask, 'cmap': 'plasma', 'vmax': .4},
                    {'title': f'combined {combined_mask.max():.3f}', 'tensor': combined_mask, 'cmap': 'plasma', 'vmax': .3},
                    {'title': f'combined_binary', 'tensor': combined_mask_binary,},
                    {'title': f'comb_bin_dilated', 'tensor': combined_mask_binary_dilated,},
                ]
                threshold_masks.append(np.percentile(combined_mask.cpu().detach().numpy(), 95).mean()) # approximator for good threshold


                ### Inpainting setup (inspired by RePaint)
                # 1. Mask the original image (get rid of anomalies) and the reconstructed image (keep reconstructed spots of original anomalies to start inpainting from)
                x_masked = (1 - mask_in_use) * x
                x_rec_masked = mask_in_use * x_rec

                to_visualize.append({'title': f'x_masked', 'tensor': x_masked,})
                to_visualize.append({'title': f'x_rec_masked', 'tensor': x_rec_masked,})
                to_visualize.append({'title': f'just stitched', 'tensor': x_masked + x_rec_masked,})

                # 2. Start inpainting with reconstructed image and not pure noise
                noise = torch.randn_like(x_rec, device=self.device)
                timesteps = torch.full([x.shape[0]], self.model.noise_level_inpaint, device=self.device).long()
                inpaint_image = self.model.inference_scheduler.add_noise(
                    original_samples=x_rec, noise=noise, timesteps=timesteps
                )

                # 3. Setup for loop
                timesteps = self.model.inference_scheduler.get_timesteps(self.model.noise_level_inpaint)
                from tqdm import tqdm
                try:
                    progress_bar = tqdm(timesteps)
                except:
                    progress_bar = iter(timesteps)
                num_resample_steps = self.model.resample_steps
                # stitched_images = []
                
                # 4. Inpainting loop
                os.makedirs(self.model.image_path, exist_ok=True)
                if os.path.exists(os.path.join(self.model.image_path, f'image_{idx * len(x)}.png')):
                    x_rec_inpainted = torch.zeros_like(x)
                    for i in range(x.shape[0]):
                        count = str(idx * len(x) + i)
                        path_to_rec = os.path.join(self.model.image_path, f'image_{count}.png')
                        inpainted = Image.open(path_to_rec).convert('L')
                        x_rec_inpainted[i] = transforms.ToTensor()(inpainted).to('cuda') 
                    final_inpainted_image = x_rec_inpainted
                else:
                    with torch.no_grad():
                        with autocast(enabled=True):
                            for t in progress_bar:
                                for u in range(num_resample_steps):
                                    # 4a) Get the known portion at t-1
                                    if t > 0:
                                        noise = torch.randn_like(x, device=self.device)
                                        timesteps_prev = torch.full([x.shape[0]], t - 1, device=self.device).long()
                                        noised_masked_original_context = self.model.inference_scheduler.add_noise(
                                            original_samples=x_masked, noise=noise, timesteps=timesteps_prev
                                        )
                                    else:
                                        noised_masked_original_context = x_masked

                                    # 4b) Perform a denoising step to get the unknown portion at t-1
                                    if t > 0:
                                        timesteps = torch.full([x.shape[0]], t, device=self.device).long()
                                        model_output = self.model.unet(x=inpaint_image, timesteps=timesteps)
                                        inpainted_from_x_rec, _ = self.model.inference_scheduler.step(model_output, t, inpaint_image)

                                    # 4c) Combine the known and unknown portions at t-1
                                    inpaint_image = torch.where(
                                        mask_in_use == 1, inpainted_from_x_rec, noised_masked_original_context
                                    )
                                    # torch.cat([noised_masked_original_context, inpainted_from_x_rec, val_image_inpainted], dim=2)
                                    # stitched_images.append(inpaint_image)

                                    # 4d) Perform resampling: sample x_t from x_t-1 -> get new image to be inpainted in the masked region
                                    if t > 0 and u < (num_resample_steps - 1):
                                        inpaint_image = (
                                            torch.sqrt(1 - self.model.inference_scheduler.betas[t - 1]) * inpaint_image
                                            + torch.sqrt(self.model.inference_scheduler.betas[t - 1]) * torch.randn_like(x, device=self.device)
                                        )

                    # store inpainted images
                    for i in range(x.shape[0]):
                        count = str(idx * len(x) + i)
                        path_to_inpainted = os.path.join(self.model.image_path, f'image_{count}.png')
                        save_image(inpaint_image[i][0], path_to_inpainted)
                        print(f'Saved inpainted image to {path_to_inpainted}')
                    final_inpainted_image = inpaint_image
                
                print("95th percentile: ", sum(threshold_masks) / len(threshold_masks))

                # 5. Compute new residual and anomaly maps
                x_res_2 = self.compute_residual(x, final_inpainted_image.clamp(0, 1), hist_eq=False)
                x_lpips_2 = self.lpips_loss(x, final_inpainted_image, retPerLayer=False)

                to_visualize.append({'title': 'final inpainted image', 'tensor': final_inpainted_image})
                to_visualize.append({'title': f'l1 w/ inp. (now eval) {x_res_2.max():.3f}', 'tensor': x_res_2, 'cmap': 'plasma', 'vmax': .9})
                to_visualize.append({'title': f'lpips w/ inpainted {x_lpips_2.max():.3f}', 'tensor': x_lpips_2, 'cmap': 'plasma', 'vmax': .4})
                to_visualize.append({'title': f'comb. w/ inpainted {(x_lpips_2*x_res_2).max():.3f}', 'tensor': x_lpips_2*x_res_2, 'cmap': 'plasma', 'vmax': .2})
                to_visualize.append({'title': f'comb before*after {(combined_mask_np*x_lpips_2*x_res_2).max():.3f}', 'tensor': combined_mask_np*x_lpips_2*x_res_2, 'cmap': 'plasma', 'vmax': .07})
                to_visualize.append({'title': 'x', 'tensor': x})
                to_visualize.append({'title': 'x_rec', 'tensor': x_rec})
                to_visualize.append({'title': 'gt', 'tensor': masks})

                for i in range(len(x)):
                    if torch.sum(masks[i][0]) > self.model.threshold_low and torch.sum(masks[i][0]) <= self.model.threshold_high: # get the desired sizes of anomalies
                        count = str(idx * len(x) + i)
                        # Don't use images with large black artifacts:
                        if int(count) in [100, 105, 112, 121, 186, 189, 210,214, 345, 382, 424, 425, 435, 434, 441, 462, 464, 472, 478, 504]:
                            print("skipping ", count)
                            continue
                        
                        # Example visualizations
                        if int(count) % 12 == 0 or int(count) in [0, 66, 325, 352, 545, 548, 231, 609, 616, 11, 254, 539, 165, 545, 550, 92, 616, 628, 630, 636, 651]: 
                            self._log_visualization(to_visualize, i, count)

                        x_i = x[i][0]
                        rec_2_i = final_inpainted_image[i][0]

                        # Evaluate on residual and combined maps from first step
                        res_2_i_np = x_res_2[i][0] * combined_mask[i][0].cpu().detach().numpy()
                        anomalous_pred.append(res_2_i_np.max())

                        pred_.append(res_2_i_np)
                        label_.append(masks[i][0].cpu().detach().numpy())

                        # Similarity metrics: x_rec vs. x
                        loss_mae = self.criterion_rec(rec_2_i, x_i)
                        test_metrics['MAE'].append(loss_mae.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), rec_2_i.cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        ssim_ = ssim(rec_2_i.cpu().detach().numpy(), x_i.cpu().detach().numpy(), data_range=1.)
                        test_metrics['SSIM'].append(ssim_)
                    
                    elif torch.sum(masks[i][0]) <= 1: # use slices without anomalies as "healthy" examples on same domain
                        res_2_i_np_healthy = x_res_2[i][0] * combined_mask[i][0].cpu().detach().numpy()
                        healthy_pred.append(res_2_i_np_healthy.max())

            
            pred_dict[dataset_key] = (pred_, label_)

            for metric in test_metrics:
                logging.info('{}: {} mean: {} +/- {}'.format(dataset_key, metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        
        for dataset_key in self.test_data_dict.keys():
            # Get some stats on prediction set
            pred_ood, label_ood = pred_dict[dataset_key]
            predictions = np.asarray(pred_ood)
            labels = np.asarray(label_ood)
            predictions_all = np.reshape(np.asarray(predictions), (len(predictions), -1))  # .flatten()
            labels_all = np.reshape(np.asarray(labels), (len(labels), -1))  # .flatten()
            print(f'Nr of preditions: {predictions_all.shape}')
            print(f'Predictions go from {np.min(predictions_all)} to {np.max(predictions_all)} with mean: {np.mean(predictions_all)}')
            print(f'Labels go from {np.min(labels_all)} to {np.max(labels_all)} with mean: {np.mean(labels_all)}')
            print('Shapes {} {} '.format(labels.shape, predictions.shape))

            # Compute global anomaly localization metrics
            dice_scores = []

            auprc_, _, _, _ = compute_auprc(predictions_all, labels_all)
            logging.info(f'Global AUPRC score: {auprc_}')
            wandb.log({f'Metrics/Global_AUPRC_{dataset_key}': auprc_})

            # Compute dice score for linear thresholds from 0 to 1
            ths = np.linspace(0, 1, 101)
            for dice_threshold in ths:
                dice = compute_dice(copy.deepcopy(predictions_all), copy.deepcopy(labels_all), dice_threshold)
                dice_scores.append(dice)
            highest_score_index = np.argmax(dice_scores)
            highest_score = dice_scores[highest_score_index]

            logging.info(f'Global highest DICE: {highest_score}')
            wandb.log({f'Metrics/Global_highest_DICE': highest_score})

        # Plot box plots over the metrics per image
        logging.info('Writing plots...')
        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)
            wandb.log({f'Metrics/{self.name}_{metric}': fig_bp})
