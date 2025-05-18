import torch
import torch.nn as nn
import lpips  # You need to install the LPIPS library: pip install lpips

class HDRLossWithLPIPS(nn.Module):
    def __init__(self, alpha_map=None, use_alpha=True, lpips_weight=1.0):
        super(HDRLossWithLPIPS, self).__init__()

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')


        self.use_alpha = use_alpha
        self.alpha_map = alpha_map
        self.lpips_weight = lpips_weight
        self.l1_loss = nn.L1Loss(reduction='none')
        self.lpips = lpips.LPIPS(net='vgg').to(device)  # Choose 'alex' or 'vgg'
    

    def forward(self, pred, target_log_hdr):
        l1_diff = self.l1_loss(pred, target_log_hdr)

        if self.use_alpha and self.alpha_map is not None:
            alpha = self.alpha_map.expand_as(l1_diff) if self.alpha_map.shape != l1_diff.shape else self.alpha_map
            l1_loss = torch.mean(alpha * l1_diff)
        else:
            l1_loss = torch.mean(l1_diff)

        # Assume pred and target_log_hdr are already in [-1, 1]
        lpips_loss = self.lpips(pred, target_log_hdr).mean()

        total_loss = l1_loss + self.lpips_weight * lpips_loss
        return total_loss

