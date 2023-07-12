import logging
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
from torchvision import transforms
#
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter
from torchvision.utils import save_image

from PIL import Image
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
        self.compute_scores = True
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

        # visualizations of metrics for different noise levels
        m_per_t = []
        example_recons = []
        dices = []
        auprcs = []
        if self.model.threshold_low == 1 and self.model.threshold_high == 71:
            size = 'Small'
        elif self.model.threshold_low == 71 and self.model.threshold_high == 570:
            size = 'Medium'
        elif self.model.threshold_low == 570 and self.model.threshold_high >= 10000:
            size = 'Large'
        elif self.model.threshold_low == 1 and self.model.threshold_high >= 10000:
            size = 'All'
        elif self.model.threshold_low == -1 and self.model.threshold_high == 1:
            size = 'healthy'
        else:
            raise ValueError('Invalid threshold values')


        nls = np.arange(50, 301, 50)
        for nl in nls:
            metrics, recon_list, auprc, dice = self.pathology_localization(global_model, nl)
            auprcs.append(auprc)
            dices.append(dice)
            m_per_t.append(metrics)
            example_recons.append(recon_list)

        print("noise levels: ", nls)
        print(f"auprc: {auprcs}")
        print(f"dice: {dices}")

        for i, metric in enumerate(m_per_t[0].keys()):
            fig = plt.figure()
            x = nls
            y = np.array([np.nanmean(m[metric]) for m in m_per_t])
            std = np.array([np.nanstd(m[metric]) for m in m_per_t])
            plt.plot(x, y, label=metric)
            plt.fill_between(x, y - std, y + std, alpha=0.3)
            plt.title(metric)
            plt.legend()
            plt.tight_layout()

            logging.info(f'To reproduce plots for {metric}')
            logging.info(f'x: {nls}')
            logging.info(f'y: {[round(i, 4) for i in y]}')
            logging.info(f'std: {[round(i, 4) for i in std]}')
            wandb.log({f'Metrics/{metric}': wandb.Image(plt)})
            plt.clf()

        for image_idx in range(len(example_recons[0])):
            fig2, axs = plt.subplots(5, len(nls), sharex=True, sharey=True, figsize=(len(nls)*2, 5*2))
            for i, nl in enumerate(nls):
                stack = example_recons[i][image_idx]
                images = np.vsplit(stack, 5)
                for j, image in enumerate(images):
                    vmax = 1
                    if j in [2, 3]:
                        cmap = 'plasma'
                        if j == 3:
                            vmax = .5
                    else:
                        cmap = 'gray'
                    axs[j, i].imshow(image, cmap=cmap, vmax=vmax)

                    # Hide tick marks and labels
                    axs[j, i].set_xticks([])
                    axs[j, i].set_yticks([])
                    axs[j, i].tick_params(axis='both', which='both', length=0)
                axs[0, i].set_title(nl)
            fig2.subplots_adjust(hspace=0)

            plt.suptitle(f'Reconstructions for {size} anomalies for different noise levels')
            wandb.log({f'Reconstructions/Reconstruction_{image_idx}': wandb.Image(plt)})
            plt.clf()

        # plot dice scores
        _ = plt.figure()
        plt.plot(nls, dices, label='Dice')
        plt.title('Dice')
        plt.legend()
        plt.tight_layout()
        wandb.log({f'Metrics/Dice': wandb.Image(plt)})

        # plot auprc scores
        _ = plt.figure()
        plt.plot(nls, auprcs, label='AUPRC')
        plt.title('AUPRC')
        plt.legend()
        plt.tight_layout()
        wandb.log({f'Metrics/AUPRC': wandb.Image(plt)})


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

    def pathology_localization(self, global_model, nl=0):        
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
        self.model.noise_level_recon = nl
        metrics = {
            'MAE': [],
            'LPIPS': [],
            'SSIM': [],
        }
        pred_dict = dict()
        example_recons = []

        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            global_counter = 0
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MAE': [],
                'LPIPS': [],
                'SSIM': [],
            }

            logging.info('DATASET: {}'.format(dataset_key))

            for idx, data in enumerate(dataset):
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
                if not os.path.exists(os.path.join(self.model.image_path, f'rec_{self.model.noise_level_recon}', f'image_{global_counter}.png')):
                        os.makedirs(os.path.join(self.model.image_path, f'rec_{self.model.noise_level_recon}'), exist_ok=True)
                        os.makedirs(os.path.join(self.model.image_path, f'original_{self.model.noise_level_recon}'), exist_ok=True)
                        x_rec, _ = self.model.sample_from_image(x, noise_level=self.model.noise_level_recon)
                        x_rec = torch.clamp(x_rec, 0, 1)
                for i in range(x.shape[0]):
                    path_to_rec = os.path.join(self.model.image_path, f'rec_{self.model.noise_level_recon}', f'image_{global_counter}.png')
                    path_to_image = os.path.join(self.model.image_path, f'original_{self.model.noise_level_recon}', f'image_{global_counter}.png')
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

                x_res = self.compute_residual(x, x_rec, hist_eq=True)
                lpips_mask = self.lpips_loss(x, x_rec, retPerLayer=False)

                to_visualize = [
                    {'title': 'original', 'tensor': x, 'cmap': 'gray', 'vmax': 1},
                    {'title': 'reconstruction', 'tensor': x_rec, 'cmap': 'gray', 'vmax': 1},
                    {'title': f'l1 {x_res.max():.3f}', 'tensor': x_res, 'cmap': 'plasma',}, 
                    {'title': f'lpips {lpips_mask.max():.3f}', 'tensor': lpips_mask, 'cmap': 'plasma', 'vmax': .5},
                ]
                
                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    if torch.sum(masks[i][0]) > self.model.threshold_low and torch.sum(masks[i][0]) <= self.model.threshold_high: # sort out where GT mask is not existent
                        # Don't use images with large black artifacts:
                        if int(count) in [100, 105, 112, 121, 186, 189, 210,214, 345, 382, 424, 425, 435, 434, 441, 462, 464, 472, 478, 504]:
                            # print("skipping ", count)
                            continue
                        
                        # Example visualizations
                        # if int(count) in [0, 66, 325, 352, 545, 548, 231, 609, 616, 11, 254, 539, 165, 545, 550, 92, 616, 628, 630, 636, 651]:
                        #     self._log_visualization(to_visualize, i, count, nl)

                        x_i = x[i][0]
                        x_rec_i = x_rec[i][0]
                        x_res_i_np = x_res[i][0]
                        lpips_mask_i_np = lpips_mask[i][0]

                        if int(count) in [548, 69, 414, 92, 545, 552, 115, 598]:
                            example_recons.append(np.vstack([x_i.cpu().detach().numpy(), x_rec_i.cpu().detach().numpy(), x_res_i_np, lpips_mask_i_np, masks[i][0].cpu().detach().numpy()]))
                      
                        pred_.append(x_res_i_np)
                        label_.append(masks[i][0].cpu().detach().numpy())

                        # Similarity losses: x_rec vs. x
                        loss_mae = self.criterion_rec(x_rec_i, x_i)
                        test_metrics['MAE'].append(loss_mae.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        ssim_ = ssim(x_rec_i.cpu().detach().numpy(), x_i.cpu().detach().numpy(), data_range=1.)
                        test_metrics['SSIM'].append(ssim_)

            pred_dict[dataset_key] = (pred_, label_)

            logging.info('Noise level: {}'.format(nl))
            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
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
            wandb.log({f'noise_level_recon_{nl}/Metrics/{self.name}_{metric}': fig_bp})

        return metrics, example_recons, auprc_, highest_score
