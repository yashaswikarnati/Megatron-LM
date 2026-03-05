# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
This script provides a basic training loop for MIMO models.
"""

import logging
import os
import sys
from functools import partial
from typing import Any, Dict, Iterator

# Enable debug logging for MIMO modules when MIMO_DEBUG is set.
if os.environ.get("MIMO_DEBUG"):
    logging.basicConfig(level=logging.DEBUG, format="%(name)s:%(levelname)s: %(message)s")
    for _mod in ("megatron.core.models.mimo",):
        logging.getLogger(_mod).setLevel(logging.DEBUG)

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)

# Add the parent directory to the path to import from megatron
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
from data.energon_avlm_task_encoder import llava_avlm_dataloader_provider
try:
    from data.energon_multimodal_provider import (
        train_valid_test_dataloaders_provider as energon_multimodal_dataloader_provider,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"energon_multimodal_provider not available: {e}")
    energon_multimodal_dataloader_provider = None
from data.energon_vlm_task_encoder import llava_vlm_dataloader_provider
from data.mock import (
    train_valid_test_datasets_provider as mock_train_valid_test_datasets_provider,
)
from model_providers.llava_avlm import model_provider_llava_avlm
from model_providers.llava_vlm import model_provider_llava_vlm
from model_providers.mock import model_provider_mock_vlm_single_encoder
from model_providers.nemotron_moe_vlm import (
    add_nemotron_moe_vlm_args,
    model_provider_nemotron_moe_vlm,
)
from utils.data_helpers import broadcast_nested_data_batch

from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain

_MODEL_PROVIDERS = {
    "mock": model_provider_mock_vlm_single_encoder,
    "llava_vlm": model_provider_llava_vlm,
    "video_llava_vlm": partial(model_provider_llava_vlm, is_video_input=True),
    "llava_avlm": model_provider_llava_avlm,
    "nemotron_moe_vlm": model_provider_nemotron_moe_vlm,
}

_DATASET_PROVIDERS = {
    "mock": mock_train_valid_test_datasets_provider,
    "llava_vlm": llava_vlm_dataloader_provider,
    "video_llava_vlm": partial(llava_vlm_dataloader_provider, is_video_input=True),
    "llava_avlm": llava_avlm_dataloader_provider,
}
if energon_multimodal_dataloader_provider is not None:
    _DATASET_PROVIDERS["energon_multimodal"] = energon_multimodal_dataloader_provider

# Model-provider-specific extra args.  Each provider can optionally export an
# ``add_<name>_args(parser)`` function that registers CLI flags specific to
# that provider.  All registered functions are called unconditionally at parse
# time (argparse ignores unused flags).
_MODEL_PROVIDER_EXTRA_ARGS = {
    "nemotron_moe_vlm": add_nemotron_moe_vlm_args,
}

def add_mimo_args(parser):
    """Add MIMO-specific arguments to the parser."""
    group = parser.add_argument_group('MIMO', 'MIMO specific arguments')

    # MIMO-specific parameters
    group.add_argument('--dataset-provider', type=str, default='mock', help='Dataset provider to choose from [mock, llava_vlm, video_llava_vlm, llava_avlm]')
    group.add_argument('--model-provider', type=str, default='mock', help='Model provider to choose from [mock, llava_vlm, video_llava_vlm, llava_avlm, nemotron_moe_vlm]')

    # mock dataloader related args
    # can control mock samples with total seq length and image seq length
    group.add_argument('--image-size', type=int, default=224, help='Image size for vision encoder')
    group.add_argument('--total-seq-length', type=int, default=512, help='Total sequence length')
    group.add_argument('--pad-token-id', type=int, default=0, help='Padding token ID')
    group.add_argument('--image-token-id', type=int, default=32000, help='Image token ID')
    group.add_argument(
        '--image-seq-length', type=int, default=197, help='Number of image tokens to pad'
    )
    group.add_argument(
        '--audio-encoder-model', type=str, default=None, help='Audio encoder model name'
    )
    group.add_argument(
        '--hf-assign-unused-tokens', type=str, nargs='+', default=None,
                       help='Assigning unused tokens to special tokens. Example: '
                       '--hf-assign-unused-tokens "<audio>,32002" "<video>,32003"'
    )
    # checkpoint related args
    group.add_argument('--language-model-checkpoint', type=str, default=None, help='Path to language model checkpoint to load')
    group.add_argument('--vision-encoder-checkpoint', type=str, default=None, help='Path to vision encoder checkpoint to load')
    # energon dataloader related args
    group.add_argument('--packing-buffer-size', type=int, default=None, help='Packing buffer size when using sequence packing')

    # Register model-provider-specific args
    for add_args_fn in _MODEL_PROVIDER_EXTRA_ARGS.values():
        add_args_fn(parser)

    return parser


def get_batch(data_iterator: Iterator[Dict[str, Any]]):
    """Generate a batch for MIMO model training.

    Args:
        data_iterator: Iterator over the dataset

    Returns:
        tuple: Batch data for model training
    """
    args = get_args()

    # Assert that context parallelism and pipeline parallelism are not supported yet
    assert (
        getattr(args, 'context_parallel_size', 1) == 1
    ), "Context parallelism is not supported yet in MIMO implementation"

    assert (getattr(args, 'pipeline_model_parallel_size', 1) == 1), \
        "Pipeline parallelism is not supported yet in MIMO implementation"
    
    # Broadcast data - only get data on tensor parallel rank 0
    # data iterator is None on other tp ranks
    # TP Rank-0 reads next batch.
    if get_tensor_model_parallel_rank() == 0:
        try:
            data = next(data_iterator)
            has_data = torch.tensor([1], dtype=torch.uint8, device='cuda')
        except StopIteration:
            has_data = torch.tensor([0], dtype=torch.uint8, device='cuda')
            data = None
    else:
        has_data = torch.empty(1, dtype=torch.uint8, device='cuda')
        data = None

    src = get_tensor_model_parallel_src_rank()
    group = get_tensor_model_parallel_group()
    torch.distributed.broadcast(has_data, src, group=group)

    if has_data.item() == 0:
        # iterator exhausted on all ranks
        # we need this to avoid race condition when first tp rank hits StopIteration
        return None

    # MiMo forward pass expects 
    # input_ids: torch.Tensor,
    # position_ids: Optional[torch.Tensor] = None,
    # attention_mask: Optional[torch.Tensor] = None,
    # loss_mask: Optional[torch.Tensor] = None,
    # labels: Optional[torch.Tensor] = None,
    # modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    # modality_seq_lengths: Optional[Dict[str, torch.Tensor]] = None,

    # For the modality inputs, the keys can be arbitrary
    # so we do a broadcast of the schema followed by a broadcast of the actual data
    # check broadcast_nested_data_batch for more details
    batch = broadcast_nested_data_batch(data)
    return batch

def loss_func(loss_mask, output_tensor):
    """Simple loss function for MIMO model training.

    Args:
        loss_mask: mask indicating which tokens contribute to the loss
        output_tensor: model output tensor

    Returns:
        tuple: (loss, num_tokens, metrics_dict)
    """
    losses = output_tensor.float()

    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])

    return (total_loss, total_tokens, {'lm loss': (reporting_loss)})


def forward_step(data_iterator, model):
    """Forward step for MIMO model training.

    Args:
        data_iterator: iterator over the dataset
        model: MIMO model instance

    Returns:
        tuple: (output_tensor, loss_function)
    """
    data_batch = get_batch(data_iterator)
    output_tensor, loss_mask = model(**data_batch)
    # Return output and loss function
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(*provider_args, **provider_kwargs):
    """Dataset provider for MIMO model training.

    Args:
        *provider_args: Additional arguments for the dataset provider
        **provider_kwargs: Additional keyword arguments for the dataset provider
    """
    runtime_args = get_args()
    try:
        dataset_provider = _DATASET_PROVIDERS[runtime_args.dataset_provider]
    except KeyError as e:
        raise ValueError(
            f"Unsupported dataset provider '{runtime_args.dataset_provider}'. "
            f"Available providers: {list(_DATASET_PROVIDERS.keys())}"
        ) from e

    return dataset_provider(*provider_args, **provider_kwargs)

def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = 32000,
    audio_special_token_id: int = 32002,
    **extra_kwargs,
):
    """Model provider for MIMO model training.

    Args:
        pre_process: Whether to pre-process the model
        post_process: Whether to post-process the model
        add_encoder: Whether to add an encoder to the model (not supported yet)(default: True)
        add_decoder: Whether to add a decoder to the model (not supported yet)(default: True)
        image_special_token_id: Special token ID for the image modality (default: 32000)
        audio_special_token_id: Special token ID for the audio modality (default: 32002)
    """
    runtime_args = get_args()

    try:
        builder_fn = _MODEL_PROVIDERS[runtime_args.model_provider]
    except KeyError as e:
        raise ValueError(
            f"Unsupported model provider '{runtime_args.model_provider}'. "
            f"Available providers: {list(_MODEL_PROVIDERS.keys())}"
        ) from e

    # Build provider-specific kwargs.
    # nemotron_moe_vlm reads --image-token-id from args directly (no wrapper).
    if runtime_args.model_provider == "mock":
        kwargs = {
            "special_token_id": image_special_token_id,
        }
    elif runtime_args.model_provider in ("llava_vlm", "video_llava_vlm"):
        kwargs = {
            "image_special_token_id": image_special_token_id,
        }
    elif runtime_args.model_provider == "llava_avlm":
        kwargs = {
            "image_special_token_id": image_special_token_id,
            "audio_special_token_id": audio_special_token_id,
        }
    elif runtime_args.model_provider == "nemotron_moe_vlm":
        kwargs = {}
    else:
        raise ValueError(
            f"Unknown model provider: {runtime_args.model_provider}. "
            f"Must be one of {list(_MODEL_PROVIDERS.keys())}"
        )

    return builder_fn(
        pre_process,
        post_process,
        add_encoder,
        add_decoder,
        **kwargs,
    )


if __name__ == "__main__":
    
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={},
        extra_args_provider=add_mimo_args,
    )
