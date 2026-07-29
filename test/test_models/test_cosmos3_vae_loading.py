import pytest
import torch

from fluxvla.tokenizers.cosmos3_wan22_vae import Cosmos3Wan22VAE


def _tiny_vae(loaded=False):
    vae = Cosmos3Wan22VAE.__new__(Cosmos3Wan22VAE)
    torch.nn.Module.__init__(vae)
    vae.vae = torch.nn.Linear(2, 2, bias=True)
    vae._weights_loaded = loaded
    vae.requested_pretrained_path = '/missing/Wan2.2_VAE.pth'
    return vae


def test_unloaded_embedded_vae_cannot_encode():
    vae = _tiny_vae(loaded=False)

    with pytest.raises(RuntimeError, match='weights are not loaded'):
        vae.encode(torch.zeros(1))


def test_exact_standalone_state_dict_load_marks_embedded_vae_ready():
    vae = _tiny_vae(loaded=False)
    weight = torch.ones_like(vae.vae.weight)
    bias = torch.ones_like(vae.vae.bias)

    vae.load_state_dict({
        'vae.weight': weight,
        'vae.bias': bias,
    }, strict=True)

    assert vae._weights_loaded is True
    torch.testing.assert_close(vae.vae.weight, weight)
    torch.testing.assert_close(vae.vae.bias, bias)


def test_partial_non_strict_state_dict_does_not_mark_embedded_vae_ready():
    vae = _tiny_vae(loaded=False)

    result = vae.load_state_dict(
        {'vae.weight': torch.ones_like(vae.vae.weight)}, strict=False)

    assert result.missing_keys == ['vae.bias']
    assert vae._weights_loaded is False


def test_recursive_parent_load_waits_for_strict_checkpoint_gate():
    vae = _tiny_vae(loaded=False)
    parent = torch.nn.Module()
    parent.vision_vae = vae

    parent.load_state_dict({
        'vision_vae.vae.weight': torch.ones_like(vae.vae.weight),
        'vision_vae.vae.bias': torch.ones_like(vae.vae.bias),
    }, strict=True)

    # A generic nn.Module recursively calls the child's _load_from_state_dict,
    # not its public load_state_dict. Cosmos3's strict parent loader owns the
    # readiness transition; arbitrary parents cannot enable it.
    assert vae._weights_loaded is False
    vae.mark_weights_loaded()
    assert vae._weights_loaded is True


def _tiny_cosmos_parent(loaded=False):
    from fluxvla.models.vlas.cosmos3_flowmatching import Cosmos3FlowMatching

    model = Cosmos3FlowMatching.__new__(Cosmos3FlowMatching)
    torch.nn.Module.__init__(model)
    model.vision_vae = _tiny_vae(loaded=loaded)
    model.probe = torch.nn.Linear(2, 2)
    model.register_load_state_dict_post_hook(
        model._mark_complete_embedded_vae_after_recursive_load)
    return model


def test_exact_cosmos_parent_load_marks_embedded_vae_ready():
    model = _tiny_cosmos_parent(loaded=False)

    result = model.load_state_dict(model.state_dict(), strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert model.vision_vae._weights_loaded is True


def test_partial_cosmos_parent_load_cannot_mark_embedded_vae_ready():
    model = _tiny_cosmos_parent(loaded=False)

    result = model.load_state_dict(
        {'probe.weight': torch.ones_like(model.probe.weight)}, strict=False)

    assert result.missing_keys
    assert model.vision_vae._weights_loaded is False


def test_non_strict_parent_resume_marks_only_complete_vae_subtree():
    model = _tiny_cosmos_parent(loaded=False)
    state = {
        key: value
        for key, value in model.state_dict().items()
        if key != 'probe.bias'
    }

    result = model.load_state_dict(state, strict=False)

    assert result.missing_keys == ['probe.bias']
    assert model.vision_vae._weights_loaded is True


def test_outer_wrapper_recursive_load_marks_complete_cosmos_vae():
    model = _tiny_cosmos_parent(loaded=False)
    outer = torch.nn.Module()
    outer.wrapped = model

    outer.load_state_dict(outer.state_dict(), strict=True)

    assert model.vision_vae._weights_loaded is True
