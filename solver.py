import torch
from torch.autograd import grad
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.nn as nn

class LMGrad():
    def __init__(self):
        super().__init__()
        self.measurement = None
        self.mask = None

        self.num_updates = 1000
        self.lr = 1e-3 
        self.prev_final_loss = float('inf')  

        self.lambda_reg = 1e-4 
        
    def set_init(self):
        self.prev_final_loss = float('inf')
    
    def set_input(self, measurement, mask):
        self.measurement = measurement
        self.mask = mask # The value set to 1 in unmasked region

    @torch.no_grad()
    def hard_consistency(self, pred_x0, model, index):
        x0 = model.decode_first_stage(pred_x0)  # D_phi(z_0)
        final_hat_x0 = self.mask * self.measurement + (1. - self.mask) * x0
        if index == 0:
            return final_hat_x0
            
        with torch.no_grad():
            final_hat_z0 = model.get_first_stage_encoding(model.encode_first_stage(final_hat_x0))
            #final_hat_z0 = model.first_stage_model.encode(x0).mode() * 0.18215
        return final_hat_z0