import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import captum._utils.common as common
from captum.influence._core.influence import DataInfluence
from captum._utils.av import AV
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer, params_t
import torch.distributions as dist


class EKFACInfluence(DataInfluence):
    def __init__(
        self,
        module: Module,
        layers: Union[str, List[str]],
        influence_src_dataset: Dataset,
        activation_dir: str,
        model_id: str = "",
        batch_size: int = 1,
        query_batch_size: int = 1,
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
            **kwargs: Any additional arguments that are necessary for specific implementations of the
                'DataInfluence' abstract class.
        """
        self.module = module
        self.layers = [layers] if isinstance(layers, str) else layers
        self.influence_src_dataset = influence_src_dataset
        self.activation_dir = activation_dir
        self.model_id = model_id
        self.batch_size = batch_size
        self.query_batch_size = query_batch_size

        self.influence_src_dataloader = DataLoader(
            self.influence_src_dataset, batch_size=batch_size, shuffle=False
        )
    
    def influence(
            self,
            inputs: Dataset,
            topk: int = 1,
            additional_forward_args: Optional[Any] = None,
            load_src_from_disk: bool = True,
            **kwargs: Any,
        ) -> Dict:

        inputs_batch_size = (
            inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]
        )

        influences: Dict[str, Any] = {}
        query_grads: Dict[str, List[Tensor]] = {}
        influence_src_grads: Dict[str, List[Tensor]] = {}

        query_dataloader = DataLoader(
            inputs, batch_size=self.query_batch_size, shuffle=False
        )

        layer_modules = [
            common._get_module_from_name(self.module, layer) for layer in self.layers
        ]

        G_list = self._compute_EKFAC_GNH()

        for i, (queries, targets) in enumerate(query_dataloader):
            criterion = torch.nn.CrossEntropyLoss()
            self.module.zero_grad()
            queries, targets = inputs
            outputs = self.module(queries)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()

            for layer in layer_modules:
                if layer.bias is not None:
                    grad_bias = layer.bias.grad
                    grad_weights = layer.weight.grad
                    grads = torch.cat([grad_weights.view(-1), grad_bias.view(-1)], dim=1)
                else:
                    grads = layer.weight.grad.view(-1)
                for grad in grads:
                    query_grads[layer].append(grad)

        for i, (inputs, targets) in enumerate(self.influence_src_dataloader):
            self.module.zero_grad()
            outputs = self.module(inputs)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()

            for layer in layer_modules:
                if layer.bias is not None:
                    grad_bias = layer.bias.grad
                    grad_weights = layer.weight.grad
                    grads = torch.cat([grad_weights.view(-1), grad_bias.view(-1)], dim=1)
                else:
                    grads = layer.weight.grad.view(-1)
                for grad in grads:
                    influence_src_grads[layer].append(grad)
        
        for layer in layer_modules:
            query_grads[layer] = torch.stack(query_grads[layer])
            influence_src_grads[layer] = torch.stack(influence_src_grads[layer])
            influences[layer] = torch.matmul(influence_src_grads[layer], torch.matmul(G_list[layer], query_grads[layer]).t())

        return influences
            

    def _compute_EKFAC_GNH(self, n_samples: int = 2):
        ekfac = EKFACDistilled(self.module, 1e-5)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        for i, (input, _) in enumerate(self.influence_src_dataloader):
            outputs = self.module(input)
            output_probs = torch.softmax(outputs, dim=-1)
            distribution = dist.Categorical(output_probs)
            for j in range(n_samples):
                samples = distribution.sample()
                loss = loss_fn(outputs, samples)
                loss.backward()
                ekfac.step()
                self.module.zero_grad()
        
        G_list = []
        # Compute average A and S
        for group in ekfac.param_groups:
            A = torch.stack(group['A']).mean(dim=0)
            S = torch.stack(group['S']).mean(dim=0)
        
            # Compute eigenvalues and eigenvectors of A and S
            la, Qa = torch.linalg.eigh(A, UPLO='U')
            ls, Qs = torch.linalg.eigh(S, UPLO='U')

            # Compute Kronecker product of eigenvalues and eigenvectors
            eigenvec_kron = torch.kron(Qa, Qs)

            eigenval_kron = torch.kron(torch.diag(la),torch.diag(ls))

            # Compute GNH
            G_list.append(torch.matmul(eigenvec_kron, torch.matmul(eigenval_kron, eigenvec_kron.t())))
            
        return G_list
            

class EKFACDistilled(Optimizer):
    def __init__(self, net, eps):
        self.eps = eps
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self.net = net
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
        super(EKFACDistilled, self).__init__(self.params, {})

    def step(self):
        for group in self.param_groups:
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]

            self._compute_kfe(group, state)

            self._precond(weight, bias, group, state)

    def calc_cov(self, calc_act: bool = True):
        for group in self.param_groups:
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None

            state = self.state[weight]

            mod = group['mod']
            x = self.state[group['mod']]['x']
            gy = self.state[group['mod']]['gy']

            # Computation of activation cov matrix for batch
            x = x.data.t()

            # Append column of ones to x if bias is not None
            if mod.bias is not None:
                ones = torch.ones_like(x[:1])
                x = torch.cat([x, ones], dim=0)
            
            if calc_act:
                # Calculate covariance matrix for activations (A_{l-1})
                A = torch.mm(x, x.t()) / float(x.shape[1])
                group['A'].append(A)

            # Computation of psuedograd of layer output cov matrix for batch
            gy = gy.data.t()

            # Calculate covariance matrix for layer outputs (S_{l})
            S = torch.mm(gy, gy.t()) / float(gy.shape[1])

            group['S'].append(S)

    def _compute_kfe(self, group, state):
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        
        # Computation of xxt
        x = x.data.t() # transpose of activations

        # Append column of ones to x if bias is not None
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)

        # Calculate covariance matrix for activations (A_{l-1})
        xxt = torch.mm(x, x.t()) / float(x.shape[1])

        # Calculate eigenvalues and eigenvectors of covariance matrix (lambdaA, QA)
        la, state['Qa'] = torch.linalg.eigh(xxt, UPLO='U')

        # Computation of ggt
        gy = gy.data.t()

        # Calculate covariance matrix for layer outputs (S_{l})
        ggt = torch.mm(gy, gy.t()) / float(gy.shape[1])

        # Calculate eigenvalues and eigenvectors of covariance matrix (lambdaS, QS)
        ls, state['Qs'] = torch.linalg.eigh(ggt, UPLO='U')

        # Outer product of the eigenvalue vectors. Of shape (len(s) x len(a))
        state['m2'] = ls.unsqueeze(1) * la.unsqueeze(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        Qa = state['Qa']
        Qs = state['Qs']
        m2 = state['m2']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        g = weight.grad.data
        s = g.shape
        s_x = x.size()
        s_gy = gy.size()
        bs = x.size(0)

        # Append column of ones to x if bias is not None
        if bias is not None:
            ones = torch.ones_like(x[:,:1])
            x = torch.cat([x, ones], dim=1)
        
        # KFE of activations ??
        x_kfe = torch.mm(x, Qa)

        # KFE of layer outputs ??
        gy_kfe = torch.mm(gy, Qs)

        m2 = torch.mm(gy_kfe.t()**2, x_kfe**2) / bs

        g_kfe = torch.mm(gy_kfe.t(), x_kfe) / bs

        g_nat_kfe = g_kfe / (m2 + self.eps)

        g_nat = torch.mm(g_nat_kfe, Qs.t())

        if bias is not None:
            gb = g_nat[:, -1].contiguous().view(*bias.shape)
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        
        g_nat = g_nat.contiguous().view(*s)
        weight.grad.data = g_nat

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

        



            


