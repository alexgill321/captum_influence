from typing import Any, Dict, List, Union, Optional
from torch.cuda.amp import autocast
import captum._utils.common as common
from captum.influence._core.influence import DataInfluence
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
import torch.distributions as dist
import tqdm


class EKFACInfluence(DataInfluence):
    def __init__(
        self,
        module: Module,
        layers: Union[str, List[str]],
        influence_src_dataset: Dataset,
        model_id: str = "",
        batch_size: int = None,
        cov_batch_size: int = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            module (Module): An instance of pytorch model. This model should define all of its
                layers as attributes of the model. The output of the model must be logits for the
                classification task.
            layers (Union[str, List[str]]): A list of layer names for which the influence will
                be computed.
            influence_src_dataset (torch.utils.data.Dataset): Pytorch dataset that is used to create
                a pytorch dataloader to iterate over the dataset. This is the dataset for which we will
                be seeking for influential instances. In most cases this is the training dataset.
            activation_dir (str): Path to the directory where the activation computations will be stored.
            model_id (str): The name/version of the model for which layer activations are being computed.
                Activations will be stored and loaded under the subdirectory with this name if provided.
            batch_size (int): Batch size for the dataloader used to iterate over the influence_src_dataset.
            query_batch_size (int): Batch size for the dataloader used to iterate over the query dataset.
            cov_batch_size (int): Batch size for the dataloader used to compute the activations.
            **kwargs: Any additional arguments that are necessary for specific implementations of the
                'DataInfluence' abstract class.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.module = module.to(self.device)
        self.layers = [layers] if isinstance(layers, str) else layers
        self.influence_src_dataset = influence_src_dataset
        self.model_id = model_id

        self.influence_src_dataloader = DataLoader(
            self.influence_src_dataset, batch_size=batch_size, shuffle=False
        )
        self.cov_src_dataloader = DataLoader(
            self.influence_src_dataset, batch_size=cov_batch_size, shuffle=False
        )
    
    def influence(
            self,
            query_dataset: Dataset,
            topk: int = 1,
            additional_forward_args: Optional[Any]= None,
            eps: float = 1e-5,
            load_src_from_disk: bool = True,
            **kwargs: Any,
        ) -> Dict:

        influences: Dict[str, Any] = {}
        query_grads: Dict[str, List[Tensor]] = {}
        influence_src_grads: Dict[str, List[Tensor]] = {}

        query_dataloader = DataLoader(
            query_dataset, batch_size=1, shuffle=False
        )

        layer_modules = [
            common._get_module_from_name(self.module, layer) for layer in self.layers
        ]

        G_list = self._compute_EKFAC_params()

        criterion = torch.nn.NLLLoss()
        print(f'Cacultating query gradients on trained model')
        for layer in layer_modules:
            query_grads[layer] = []
            influence_src_grads[layer] = []

        for _, (inputs, targets) in tqdm.tqdm(enumerate(query_dataloader), total=len(query_dataloader)):
            self.module.zero_grad()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.module(inputs)

            loss = criterion(outputs, targets.view(-1))
            loss.backward()

            for layer in layer_modules:
                Qa = G_list[layer]['Qa']
                Qs = G_list[layer]['Qs']
                eigenval_diag = G_list[layer]['lambda']
                if layer.bias is not None:
                    grad_bias = layer.bias.grad
                    grad_weights = layer.weight.grad
                    grad_bias = grad_bias.reshape(-1, 1)
                    grads = torch.cat((grad_weights, grad_bias), dim=1)
                else:
                    grads = layer.weight.grad

                p1 = torch.matmul(Qs, torch.matmul(grads, torch.t(Qa)))
                p2 = torch.reciprocal(eigenval_diag+eps).reshape(p1.shape[0], -1)
                ihvp = torch.flatten(torch.matmul(torch.t(Qs), torch.matmul((p1/p2), Qa)))
                query_grads[layer].append(ihvp)

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        print(f'Cacultating training src gradients on trained model')
        for i, (inputs, targets) in tqdm.tqdm(enumerate(self.influence_src_dataloader), total=len(self.influence_src_dataloader)):
            self.module.zero_grad()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.module(inputs)
            loss = criterion(outputs, targets.view(-1))
            for single_loss in loss:
                single_loss.backward(retain_graph=True)

                for layer in layer_modules:
                    if layer.bias is not None:
                        grad_bias = layer.bias.grad
                        grad_weights = layer.weight.grad
                        grad_bias = grad_bias.reshape(-1, 1)
                        grads = torch.cat([grad_weights, grad_bias], dim=1)
                    else:
                        grads = layer.weight.grad
                    influence_src_grads[layer].append(torch.flatten(grads))

            # Calculate influences by batch to save memory
            for layer in layer_modules:
                query_grad_matrix = torch.stack(query_grads[layer], dim=0)
                influence_src_grad_matrix = torch.stack(influence_src_grads[layer], dim=0)
                tinf = torch.matmul(query_grad_matrix, torch.t(influence_src_grad_matrix))
                tinf = tinf.detach().cpu()
                if layer not in influences:
                    influences[layer] = tinf
                else:
                    influences[layer] = torch.cat((influences[layer], tinf), dim=1)
                influence_src_grads[layer] = []
                
        return influences
            

    def _compute_EKFAC_params(self, n_samples: int = 2):
        ekfac = EKFAC(self.module, 1e-5)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        for _, (input, _) in tqdm.tqdm(enumerate(self.cov_src_dataloader), total=len(self.cov_src_dataloader)):
            input = input.to(self.device)
            outputs = self.module(input)
            output_probs = torch.softmax(outputs, dim=-1)
            distribution = dist.Categorical(output_probs)
            for _ in range(n_samples):
                samples = distribution.sample()
                loss = loss_fn(outputs, samples)
                loss.backward(retain_graph=True)
                ekfac.step()
                self.module.zero_grad()
                ekfac.zero_grad()
        
        G_list = {}
        # Compute average A and S
        for group in ekfac.param_groups:
            G_list[group['mod']] = {}
            with autocast():
                A = torch.stack(group['A']).mean(dim=0)
                S = torch.stack(group['S']).mean(dim=0)

                print(f'Activation cov matrix shape {A.shape}')
                print(f'Layer output cov matrix shape {S.shape}')
            
                # Compute eigenvalues and eigenvectors of A and S
                la, Qa = torch.linalg.eigh(A)
                ls, Qs = torch.linalg.eigh(S)
                eigenval_diags = torch.outer(la, ls).flatten(start_dim=0)

            G_list[group['mod']]['Qa'] = Qa
            G_list[group['mod']]['Qs'] = Qs
            G_list[group['mod']]['lambda'] = eigenval_diags
            
        return G_list
    
def generate_dataset_influence_src_grads(
        path: str,
        model: Module,
        identifier: Optional[str] = None,
        layer: Optional[str] = None,
        num_id: Optional[str] = None,

    ) -> Optional[Union[ISGDataset, List[ISGDataset]]]:
    r"""
    Generates the gradients of the training dataset with respect to the model
    """

class ISGDataset(Dataset):
    def __init__(
            self, 
            path: str,
            model_id: str, 
            layer: Optional[str] = None, 
            identifier: Optional[str] = None, 
            num_id: Optional[str] = None,
        ) -> None:
        r"""
        Loads into memory the gradients of the training dataset asscociated with the input
        'model_id' and 'layer' if provided.

        Args:
            path(str): Path to the directory where the gradients are stored.
            model_id(str): The name/version of the model for which gradients
              are being computed and stored.
            identifier(str or None): An optional identifier for the layer activations.
            Can be used to distinguish between different training batches.
            layer (str or None): The name of the layer for which gradients are computed.
            num_id(str or None): An optional string representing the batch number for
                which gradients are computed.
        """
        self.isg_filesearch = 

class EKFAC(Optimizer):
    def __init__(self, net, eps):
        self.eps = eps
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self.net = net
        self.calc_act = True

        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class, 'A': [], 'S': []}
                self.params.append(d)
        super(EKFAC, self).__init__(self.params, {})

    def step(self):
        for group in self.param_groups:
            mod = group['mod']
            x = self.state[mod]['x']
            gy = self.state[mod]['gy']

            # Computation of activation cov matrix for batch
            x = x.data.t()

            # Append column of ones to x if bias is not None
            if mod.bias is not None:
                ones = torch.ones_like(x[:1])
                x = torch.cat([x, ones], dim=0)
            
            if self.calc_act:
                # Calculate covariance matrix for activations (A_{l-1})
                group['A'].append(torch.mm(x, x.t()) / float(x.shape[1]))

            # Computation of psuedograd of layer output cov matrix for batch
            gy = gy.data.t()

            # Calculate covariance matrix for layer outputs (S_{l})
            group['S'].append(torch.mm(gy, gy.t()) / float(gy.shape[1]))

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)
        



            


