import torch
import sys


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                print('TaskVector:' + finetuned_checkpoint)
                pretrained_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
                finetuned_state_dict_ori = torch.load(finetuned_checkpoint, map_location='cpu')
                finetuned_state_dict_ori = finetuned_state_dict_ori['model_state_dict']
                finetuned_state_dict = {}
                for key, value in finetuned_state_dict_ori.items():
                    if key in ['graph_pred_linear.weight', 'graph_pred_linear.bias']:
                        continue
                    new_key = key.replace('gnn.', '')
                    finetuned_state_dict[new_key] = value
                
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    if key in ['batch_norms.0.running_mean', 'batch_norms.0.running_var',
                               'batch_norms.1.running_mean', 'batch_norms.1.running_var', 
                               'batch_norms.2.running_mean', 'batch_norms.2.running_var', 
                               'batch_norms.3.running_mean', 'batch_norms.3.running_var', 
                               'batch_norms.4.running_mean', 'batch_norms.4.running_var']:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        with torch.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(coefficients[k] * taskvectors[k][key] for k in range(len(taskvectors)))
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
    
    def keep_top_20_percent(self):
        with torch.no_grad():
            self.topvector = {}
            for key in self.vector:
                
                #flat_matrix = self.vector[key].view(-1)
                
                abs_flat = self.vector[key].abs()
                threshold = torch.quantile(abs_flat, q = 0.8)  
                #print(threshold)
                
                mask = abs_flat >= threshold
                # print(mask)
                # sys.exit()
                self.topvector[key] = self.vector[key] * mask
                # sparsed_vector = self.vector[key] * mask
                # scales = torch.sqrt(torch.mean(self.vector[key]**2)) / torch.sqrt(torch.mean(sparsed_vector**2))
                # # print(scales)
                # self.vector[key] = sparsed_vector * scales
    
    def rescale(self):
        with torch.no_grad():
            new_vector = {}
            scales = 0.
            new_scales = 0.
            for key in self.vector:
                scales += torch.mean(torch.abs(self.vector[key]))
                new_scales += torch.mean(torch.abs(self.topvector[key]))
            rescaler = scales / new_scales
            print(rescaler)
            for key in self.vector:
                new_vector[key] = self.topvector[key] * rescaler
            self.rescaledvector = new_vector
    
    def svd(self, tensor: torch.Tensor,density: float):
        if density >= 1:
            return tensor
        if density <= 0:
            return torch.zeros_like(tensor)
        if len(tensor.shape) == 1:
            # rank=1
            return tensor

        driver = None
        if tensor.is_cuda:
            driver = 'gesvda'

        U, S, Vh = torch.linalg.svd(tensor, full_matrices=True, driver=driver)
        new_rank = int(density * len(S))
        U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]
        res = U @ torch.diag(S) @ Vh
        return res
    
    def sparsify(self, density):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.svd(self.vector[key], density)
            self.vector = new_vector
        

