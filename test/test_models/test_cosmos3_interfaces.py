from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys

import pytest
import torch


class _Registry:

    @staticmethod
    def register_module():

        def decorator(cls):
            return cls

        return decorator


@dataclass
class _SequencePlan:
    has_text: bool = True
    has_vision: bool = True
    condition_frame_indexes_vision: list = field(default_factory=list)
    has_action: bool = True
    condition_frame_indexes_action: list = field(default_factory=list)
    action_start_frame_offset: int = 0


def _package(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


def _load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def cosmos_modules():
    """Load interface modules without importing GPU-only backbones."""
    root = Path(__file__).parents[2]
    module_names = [
        'fluxvla',
        'fluxvla.engines',
        'fluxvla.models',
        'fluxvla.models.vlas',
        'fluxvla.models.vlas.cosmos3',
        'fluxvla.models.vlas.cosmos3.flow_utils',
        'fluxvla.models.third_party_models',
        'fluxvla.models.third_party_models.cosmos3',
        'fluxvla.models.third_party_models.cosmos3.data',
        'fluxvla.models.third_party_models.cosmos3.data.vfm',
        ('fluxvla.models.third_party_models.cosmos3.data.vfm.'
         'sequence_packing'),
    ]
    previous = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules[name] = _package(name)

    sys.modules['fluxvla.engines'].TRANSFORMS = _Registry()
    sequence_packing = sys.modules[
        'fluxvla.models.third_party_models.cosmos3.data.vfm.sequence_packing']
    sequence_packing.SequencePlan = _SequencePlan
    sequence_packing.PackedSequence = object

    flow_utils = sys.modules['fluxvla.models.vlas.cosmos3.flow_utils']
    flow_utils._as_text_ids = lambda *args, **kwargs: []
    flow_utils._expand_sampler_timestep = lambda *args, **kwargs: None
    flow_utils._sample_arch_invariant_noise = lambda *args, **kwargs: None

    inference = _load_module(
        'fluxvla.models.vlas.cosmos3.inference_mixin',
        root / 'fluxvla/models/vlas/cosmos3/inference_mixin.py')
    transforms = _load_module(
        '_transform_cosmos3_interface_test_module',
        root / 'fluxvla/transforms/transform_cosmos3.py')
    yield inference, transforms

    sys.modules.pop('fluxvla.models.vlas.cosmos3.inference_mixin', None)
    sys.modules.pop('_transform_cosmos3_interface_test_module', None)
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_padded_state_action_tokens_are_limited_to_raw_action_dim(
        cosmos_modules):
    inference, _ = cosmos_modules
    model = inference.Cosmos3InferenceMixin()
    model.max_action_dim = 64
    padded_state = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64)

    tokens = model._prepare_action_tokens(
        padded_state,
        dtype=torch.float32,
        device=torch.device('cpu'),
        raw_action_dim_value=8,
    )

    torch.testing.assert_close(tokens[..., :8], padded_state[..., :8])
    assert torch.count_nonzero(tokens[..., 8:]) == 0


def test_predict_action_forwards_sampling_overrides_and_rejects_batches(
        cosmos_modules):
    inference, _ = cosmos_modules

    class PolicyStub(inference.Cosmos3InferenceMixin):
        action_horizon = 4
        ori_action_dim = 8

        def _predict_action_joint(self, **kwargs):
            self.predict_kwargs = kwargs
            return {'actions': torch.zeros(1, self.action_horizon, 64)}

    model = PolicyStub()
    images = torch.zeros(1, 3, 1, 2, 2)
    tokens = torch.ones(1, 3, dtype=torch.long)

    actions = model.predict_action(
        images=images,
        lang_tokens=tokens,
        embodiment_ids=torch.tensor([1]),
        raw_action_dim=torch.tensor(8),
        seed=23,
        guidance=2.5,
        shift=7.0,
        num_inference_steps=12,
        negative_text_token_ids=torch.tensor([[5, 6]]),
    )

    assert actions.shape == (1, 4, 8)
    assert model.predict_kwargs['seed'] == 23
    assert model.predict_kwargs['guidance'] == 2.5
    assert model.predict_kwargs['shift'] == 7.0
    assert model.predict_kwargs['num_inference_steps'] == 12
    torch.testing.assert_close(
        model.predict_kwargs['negative_text_token_ids'],
        torch.tensor([[5, 6]]),
    )

    with pytest.raises(ValueError, match='batch size 1'):
        model.predict_action(
            images=images.repeat(2, 1, 1, 1, 1),
            lang_tokens=tokens.repeat(2, 1),
        )


def test_sampling_overrides_do_not_mutate_model_defaults(cosmos_modules):
    inference, _ = cosmos_modules
    model = inference.Cosmos3InferenceMixin()
    model.rectified_flow_inference_config = {
        'scheduler_type': 'UniPC',
        'num_steps': 30,
        'shift': 10.0,
    }

    config = model._sampling_config(num_inference_steps=5, shift=3.0)

    assert config['scheduler_type'] == 'unipc'
    assert config['num_steps'] == 5
    assert config['shift'] == 3.0
    assert model.rectified_flow_inference_config['num_steps'] == 30
    assert model.rectified_flow_inference_config['shift'] == 10.0


def test_generate_joint_forwards_negative_cfg_tokens(cosmos_modules):
    inference, _ = cosmos_modules

    class JointStub(inference.Cosmos3InferenceMixin):
        ori_action_dim = 8

        def _prepare_flow_request(self, **kwargs):
            self.flow_kwargs = kwargs
            return object()

        def _generate_flow(self, flow):
            del flow
            return {
                'actions': torch.zeros(1, 4, 64),
                'vision_latents': torch.zeros(1),
            }

    model = JointStub()
    negative = torch.tensor([[5, 6]])
    model.generate_joint(
        images=torch.zeros(1, 3, 1, 2, 2),
        text_token_ids=torch.tensor([[1, 2]]),
        negative_text_token_ids=negative,
        embodiment_id=8,
        raw_action_dim=8,
        sequence_plan=_SequencePlan(),
        num_frames=5,
        action_horizon=4,
        guidance=3.0,
    )

    torch.testing.assert_close(model.flow_kwargs['negative_text_token_ids'],
                               negative)
    assert model.flow_kwargs['guidance'] == 3.0


def test_cosmos_metadata_accepts_private_inference_model_path(cosmos_modules):
    _, transforms = cosmos_modules
    transform = transforms.SetCosmos3ActionMetadata(
        conditioning_fps=15.0,
        action_fps=15.0,
        raw_action_dim=8,
        prepend_state_to_action=True,
        model_path='/unused/checkpoint',
    )

    result = transform({})

    assert result['conditioning_fps'].item() == 15.0
    assert result['action_fps'].item() == 15.0
    assert result['raw_action_dim'].item() == 8
    assert result['prepend_state_to_action'] is True


def test_sequence_plan_rejects_incompatible_action_length(cosmos_modules):
    _, transforms = cosmos_modules
    with pytest.raises(ValueError, match='non-history action length'):
        transforms.build_sequence_plan_from_mode(
            mode='policy',
            video_length=5,
            action_length=2,
        )

    plan = transforms.build_sequence_plan_from_mode(
        mode='policy',
        video_length=5,
        action_length=5,
    )
    assert plan.condition_frame_indexes_action == [0]
    assert plan.action_start_frame_offset == 0


def test_sequence_builder_aligns_state_and_validity_masks(cosmos_modules):
    _, transforms = cosmos_modules
    transform = transforms.BuildCosmos3Sequence(
        mode='policy',
        frame_window_size=5,
        raw_action_dim=8,
        prepend_state_to_action=True,
    )
    sample = {
        'actions': torch.zeros(4, 64).numpy(),
        'states': torch.zeros(64).numpy(),
        'action_masks': torch.tensor([1, 1, 0, 0]).numpy(),
        'frame_masks': torch.ones(5).numpy(),
        'embodiment_ids': torch.tensor(8).numpy(),
    }

    result = transform(sample)

    assert result['actions'].shape == (5, 64)
    assert result['action_masks'].dtype.name == 'bool'
    assert result['action_masks'].tolist() == [True, True, True, False, False]
    assert result['frame_masks'].dtype.name == 'bool'


def test_prompt_emits_matching_empty_chat_tokens_for_cfg(cosmos_modules):
    _, transforms = cosmos_modules

    class Tokenizer:

        def apply_chat_template(self, conversations, **kwargs):
            del kwargs
            content = conversations[-1]['content']
            return [10, 11] if content == '' else [20, 21, 22]

    transform = transforms.ProcessCosmos3Prompt(
        tokenizer={'type': 'unused'},
        output_key='lang_tokens',
        negative_output_key='negative_text_token_ids',
    )
    transform._tokenizer = Tokenizer()

    result = transform({'task_description': 'pick up the cup'})

    assert result['lang_tokens'].tolist() == [20, 21, 22]
    assert result['negative_text_token_ids'].tolist() == [10, 11]


def test_prompt_tokenizer_contract_fails_on_revision_drift(cosmos_modules):
    _, transforms = cosmos_modules

    class Tokenizer:
        vocab_size = 131072
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 11

        @staticmethod
        def convert_tokens_to_ids(token):
            return {'<|vision_start|>': 20, '<|vision_end|>': 99}[token]

    transform = transforms.ProcessCosmos3Prompt(
        tokenizer={'type': 'unused'},
        expected_vocab_size=131072,
        expected_special_token_ids={
            'pad_token_id': 0,
            'bos_token_id': 1,
            'eos_token_id': 11,
            'start_of_generation': 20,
            'end_of_generation': 21,
        },
    )
    with pytest.raises(ValueError, match='end_of_generation mismatch'):
        transform._validate_tokenizer_contract(Tokenizer())
