import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import captum._utils.common as common
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import glob
import os

class ISG:
    r"""
    This class provides functionality to store and load influence source gradients
    generated for pre-defined neural network layers. 
    It also provides functionality to check if influence src gradients already exist
    in the manifold and other auxiliary functions.

    This class also defines a torch `Dataset`, representing influence src gradients,
    which enables lazy access to ISGs and layers stored in the manifold.
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
            self.isg_filesearch = ISG._construct_filesearch(
                path, model_id, identifier, layer, num_id
            )

            files = glob.glob(self.isg_filesearch)

            self.files = ISG.sort_files(files)

        def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, ...]]:
            assert idx < len(self.files), "Index out of range"
            fl = self.files[idx]
            isg = torch.load(fl)
            return isg
        
        def __len__(self) -> int:
            return len(self.files)

    ISG_DIR_NAME: str = "isg"

    def __init__(self) -> None:
        pass



    @staticmethod
    def generate_dataset_influence_src_gradients(
        path: str,
        model: Module,
        model_id: str,
        layers: Union[str, List[str]],
        dataloader: DataLoader,
        identifier: Optional[str] = None,
        load_from_disk: bool = True,
        return_isg: bool = False,
    ) -> Optional[Union[ISGDataset, List[ISGDataset]]]:
        
        unsaved_layers = ISG._manage_loading_layers(
            path,
            model_id,
            layers,
            load_from_disk,
            identifier,
        )
        if len(unsaved_layers) >= 0:
            for i, data in enumerate(dataloader):
                ISG._compute_and_save_influence_src_gradients(
                    path,
                    model,
                    model_id,
                    layers,
                    ISG._unpack_data(data),
                    identifier,
                    str(i),
                )

    @staticmethod
    def _compute_and_save_influence_src_gradients(
        path: str,
        model: Module,
        model_id: str,
        layers: Union[str, List[str]],
        inputs: Tuple[Tensor, Tuple[Tensor, ...]],
        identifier: str,
        num_id: str,
        additional_forward_args: Any = None,
        load_from_disk: bool = True,
    ) -> None:

        unsaved_layers = ISG._manage_loading_layers(
            path,
            model_id,
            layers,
            load_from_disk,
            identifier,
            num_id,
        )
        layer_modules = [
            common._get_module_from_name(model, layer) for layer in unsaved_layers
        ]
        if len(unsaved_layers) > 0:
            layer_gradients = 

    @staticmethod
    def exists(
        path: str,
        model_id: str,
        identifier: Optional[str] = None,
        layer: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> bool:
        r"""
        Verifies whether the model + layer influence src gradients exist under the
        path.

        Args:
            path (str): The path where the influence src gradients
                    for the `model_id` are stored.
            model_id (str): The name/version of the model for which influence src
                    gradients are being computed and stored.
            identifier (str or None): An optional identifier for the layer ISGs.
                    Can be used to distinguish between gradients for different
                    training batches. For example, the id could be a suffix composed of
                    a train/test label and numerical value, such as "-train-xxxxx".
                    The numerical id is often a monotonic sequence taken from datetime.
            layer (str or None): The layer for which the influence src gradients are
                    computed.
            num_id (str): An optional string representing the batch number for which
                    the influence src gradients are computed

        Returns:
            exists (bool): Indicating whether the influence src gradients for the `layer`
                    and `identifier` (if provided) and num_id (if provided) were stored
                    in the manifold. If no `identifier` is provided, will return `True`
                    if any layer activation exists, whether it has an identifier or
                    not, and vice-versa.
        """
        isg_dir = ISG._assemble_model_dir(path, model_id)
        isg_filesearch = ISG._construct_file_search(
            path, model_id, identifier, layer, num_id
        )
        return os.path.exists(isg_dir) and len(glob.glob(isg_filesearch)) > 0
    
    @staticmethod
    def _assemble_model_dir(path: str, model_id: str) -> str:
        r"""
        Returns a directory path for the given source path `path` and `model_id.`
        This path is suffixed with the '/' delimiter.
        """
        return "/".join([path, ISG.ISG_DIR_NAME, model_id, ""])

    @staticmethod
    def _manage_loading_layers(
        path: str,
        model_id: str,
        layers: Union[str, List[str]],
        load_from_disk: bool = True,
        identifier: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> List[str]:
    
        layers = [layers] if isinstance(layers, str) else layers
        unsaved_layers = []

        if load_from_disk:
            for layer in layers:
                if not ISG.exists(path, model_id, identifier, layer, num_id):
                    unsaved_layers.append(layer)
        else:
            unsaved_layers = layers
            warnings.warn(
                "Overwriting activations: load_from_disk is set to False. Removing all "
                f"activations matching specified parameters {{path: {path}, "
                f"model_id: {model_id}, layers: {layers}, identifier: {identifier}}} "
                "before generating new activations."
            )
            for layer in layers:
                files = glob.glob(
                    ISG._construct_file_search(path, model_id, identifier, layer)
                )
                for fl in files:
                    os.remove(fl)

        return unsaved_layers
    
    @staticmethod
    def _construct_file_search(
        source_dir: str,
        model_id: str,
        identifier: Optional[str] = None,
        layer: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> str:
        r"""
        Returns a search string that can be used by glob to search `source_dir/model_id`
        for the desired layer/identifier pair. Leaving `layer` as None will search ids
        over all layers, and leaving `identifier` as none will search layers over all
        ids.  Leaving both as none will return a path to glob for every activation.
        Assumes identifier is always specified when saving activations, so that
        activations live at source_dir/model_id/identifier/layer
        (and never source_dir/model_id/layer)
        """

        av_filesearch = ISG._assemble_model_dir(source_dir, model_id)

        av_filesearch = os.path.join(
            av_filesearch, "*" if identifier is None else identifier
        )

        av_filesearch = os.path.join(av_filesearch, "*" if layer is None else layer)

        av_filesearch = os.path.join(
            av_filesearch, "*.pt" if num_id is None else "%s.pt" % num_id
        )

        return av_filesearch