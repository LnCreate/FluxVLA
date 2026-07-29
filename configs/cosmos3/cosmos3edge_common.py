# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Shared, pure-Python config builders for Cosmos3-Edge experiments."""

from copy import deepcopy
from pathlib import Path
from runpy import run_path


EDGE_BASE_CHECKPOINT = './checkpoints/Cosmos3-Edge'
WAN22_VAE = './checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'

EDGE_TEXT_CONFIG = dict(
    model_type='nemotron_3_dense_vl_text',
    vocab_size=131072,
    tie_word_embeddings=False,
    hidden_size=2048,
    intermediate_size=9216,
    num_hidden_layers=28,
    num_attention_heads=16,
    head_dim=128,
    num_key_value_heads=8,
    mlp_hidden_act='relu2',
    attention_bias=False,
    mlp_bias=False,
    layer_norm_epsilon=1e-5,
    pad_token_id=11,
    bos_token_id=1,
    eos_token_id=11,
    max_position_embeddings=131072,
    rope_theta=100000000.0,
    enable_mrope=True,
    mrope_section=[24, 20, 20],
    use_und_k_norm_for_gen=True,
)

EDGE_SPECIAL_TOKENS = dict(
    eos_token_id=11,
    start_of_generation=20,
    end_of_generation=21,
)


def _checkpoint_action_layout(checkpoint):
    """Read Dmax/domain count from the checkpoint header when available.

    Configs remain inspectable before weights are downloaded, using the
    published VFM layout as an expected fallback.  A real checkpoint present
    at launch is authoritative and any absent/ambiguous action head is fatal.
    """
    path = Path(checkpoint).expanduser()
    expected = (64, 32)
    if not path.exists():
        return expected

    source = (Path(__file__).resolve().parents[2] /
              'fluxvla/models/vlas/cosmos3/checkpoint_mixin.py')
    infer = run_path(str(source))['infer_cosmos3_action_layout']
    return infer(path)


def _tokenizer(checkpoint):
    return dict(
        type='PretrainedTokenizer',
        model_path=checkpoint,
        model_max_length=4096,
        padding_side='right',
        trust_remote_code=True,
    )


def _model(
    checkpoint,
    raw_action_dim,
    horizon,
    tuning='partial',
    enable_vision_loss=True,
    action_layout=None,
    action_init='checkpoint',
    fresh_action_domain_ids=None,
):
    if tuning not in {'action', 'lora', 'partial', 'full'}:
        raise ValueError(f'Unsupported Cosmos3-Edge tuning mode: {tuning}')
    hidden_size = EDGE_TEXT_CONFIG['hidden_size']
    max_action_dim, num_domains = (
        _checkpoint_action_layout(checkpoint)
        if action_layout is None else action_layout)
    if max_action_dim < raw_action_dim:
        raise ValueError(
            f'Checkpoint Dmax={max_action_dim} is smaller than raw action '
            f'dim {raw_action_dim}.')
    model = dict(
        type='Cosmos3FlowMatching',
        vlm_backbone=dict(
            type='Cosmos3EdgeBackbone',
            vlm_config=deepcopy(EDGE_TEXT_CONFIG),
            include_visual=False,
            skip_init_weights=True,
        ),
        vision_latent_dim=48,
        latent_patch_size=2,
        max_action_dim=max_action_dim,
        num_embodiment_domains=num_domains,
        vision_in_proj=dict(
            type='LinearProjector',
            in_dim=48 * 2 * 2,
            out_dim=hidden_size,
        ),
        vision_out_proj=dict(
            type='LinearProjector',
            in_dim=hidden_size,
            out_dim=48 * 2 * 2,
        ),
        action_in_proj=dict(
            type='DomainAwareLinear',
            input_size=max_action_dim,
            output_size=hidden_size,
            num_domains=num_domains,
        ),
        action_out_proj=dict(
            type='DomainAwareLinear',
            input_size=hidden_size,
            output_size=max_action_dim,
            num_domains=num_domains,
        ),
        rectified_flow_training_config=dict(
            shift={'256': 3, '480': 5, '720': 10},
            use_dynamic_shift=False,
            train_time_image_distribution='logitnormal',
            train_time_video_distribution='waver',
            train_time_action_distribution='logitnormal',
            train_time_weight='uniform',
            vision_loss_weight=1.0,
            independent_action_schedule=False,
            shift_action=None,
            use_high_sigma_strategy=False,
            use_high_sigma_strategy_action=False,
            use_discrete_rf=False,
            normalize_loss_by_active=True,
            action_loss_weight=10.0,
        ),
        rectified_flow_inference_config=dict(
            num_train_timesteps=1000,
            scheduler_type='unipc',
            num_steps=10,
            shift=10.0,
            use_dynamic_shifting=False,
            use_karras_sigmas=False,
        ),
        timestep_scale=0.001,
        packed_attention_backend='flash2',
        position_embedding_type='unified_3d_mrope',
        unified_3d_mrope_reset_spatial_ids=True,
        unified_3d_mrope_temporal_modality_margin=15000,
        enable_fps_modulation=True,
        base_fps=24.0,
        special_tokens=deepcopy(EDGE_SPECIAL_TOKENS),
        pretrained_name_or_path=checkpoint,
        strict_mapping=True,
        checkpoint_min_coverage=1.0,
        action_init=action_init,
        vision_vae=dict(
            type='Cosmos3Wan22VAE',
            pretrained_name_or_path=WAN22_VAE,
            encode_exact_durations=[horizon + 1],
        ),
        ori_action_dim=raw_action_dim,
        action_horizon=horizon,
        freeze_vlm_backbone=tuning in {'action', 'lora'},
        freeze_non_moe_vlm_backbone=tuning == 'partial',
        freeze_non_action_components=tuning == 'action',
        enable_vision_loss=enable_vision_loss,
    )
    if fresh_action_domain_ids is not None:
        model['fresh_action_domain_ids'] = list(fresh_action_domain_ids)
    if tuning == 'lora':
        model.update(
            use_lora=True,
            lora_rank=32,
            lora_alpha=64,
            lora_dropout=0.0,
            lora_target_modules=[
                'self_attn.q_proj_moe_gen',
                'self_attn.k_proj_moe_gen',
                'self_attn.v_proj_moe_gen',
                'self_attn.o_proj_moe_gen',
                'mlp_moe_gen.up_proj',
                'mlp_moe_gen.down_proj',
            ],
            modules_to_save=[
                'action_in_proj',
                'action_out_proj',
                'action_modality_embed',
            ],
        )
    return model


def _collator(pad_id=11):
    return dict(
        type='Cosmos3Collator',
        tensor_keys=[
            'images',
            'actions',
            'states',
            'embodiment_ids',
            'raw_action_dim',
            'conditioning_fps',
            'action_fps',
            'action_masks',
            'frame_masks',
        ],
        sequence_keys=['text_token_ids'],
        list_keys=['sequence_plan'],
        meta_keys=['task_description', 'stats', 'info', 'timestamp'],
        pad_id=pad_id,
    )


def _runner(tokenizer, tuning, max_steps, save_interval=500):
    action_lr = 2e-4
    base_lr = action_lr if tuning == 'action' else 4e-5
    common = dict(
        type='DDPTrainRunner' if tuning == 'lora' else 'FSDPTrainRunner',
        max_steps=max_steps,
        max_epochs=None,
        optimizer=dict(
            lr=base_lr,
            type='AdamW',
            weight_decay=0.05,
            betas=(0.9, 0.99),
            eps=1e-8,
            fused=True,
            paramwise_learning_rate={
                'action_in_proj.': action_lr,
                'action_out_proj.': action_lr,
                'action_modality_embed': action_lr,
                # PEFT wraps modules-to-save below base_model.model.*.
                'base_model.model.action_in_proj.': action_lr,
                'base_model.model.action_out_proj.': action_lr,
                'base_model.model.action_modality_embed.': action_lr,
            },
        ),
        max_grad_norm=1.0,
        save_iter_interval=save_interval,
        max_keep_ckpts=4,
        tokenizer=tokenizer,
        collator=_collator(),
        sampler=None,
        grad_accumulation_steps=8,
        metric=dict(
            type='VLAMetric',
            active_trackers=('jsonl', 'wandb'),
            run_dir='work_dirs',
            grad_accumulation_steps=8,
            window_size=1,
        ),
        lr_scheduler=dict(
            type='linear-warmup+cosine-decay',
            warmup_ratio=0.03,
        ),
        enable_gradient_checkpointing=True,
        enable_mixed_precision_training=True,
        mixed_precision_dtype='bf16',
    )
    if tuning == 'lora':
        common['static_graph'] = False
    else:
        common.update(sharding_strategy='full-shard', change_key_name=False)
    return common


def build_libero_config(
    suite='libero_spatial',
    tuning='partial',
    max_steps=2000,
    save_interval=500,
    task_indices=None,
    max_windows=None,
    eval_task_ids=None,
    eval_trials=50,
    seed=7,
    cfg_dropout_rate=0.1,
    action_init='checkpoint',
    fresh_action_domain_ids=None,
    action_layout=None,
    data_root_path=None,
):
    suite_to_data = {
        'libero_spatial': './datasets/libero_spatial_no_noops_lerobotv2.1',
        'libero_object': './datasets/libero_object_no_noops_lerobotv2.1',
        'libero_goal': './datasets/libero_goal_no_noops_lerobotv2.1',
        'libero_10': './datasets/libero_10_no_noops_lerobotv2.1',
    }
    if suite not in suite_to_data:
        raise ValueError(f'Unsupported LIBERO suite: {suite}')
    if data_root_path is None:
        data_root_path = suite_to_data[suite]
    elif isinstance(data_root_path, (list, tuple)):
        data_root_path = [
            str(Path(path).expanduser()) for path in data_root_path
        ]
    else:
        data_root_path = str(Path(data_root_path).expanduser())
    if action_init not in {'checkpoint', 'fresh_domain', 'fresh_all'}:
        raise ValueError(f'Unsupported action_init: {action_init!r}')
    if action_init == 'fresh_domain' and fresh_action_domain_ids is None:
        fresh_action_domain_ids = [5]
    if action_init != 'fresh_domain' and fresh_action_domain_ids is not None:
        raise ValueError(
            'fresh_action_domain_ids is only valid with '
            'action_init="fresh_domain".')
    checkpoint = EDGE_BASE_CHECKPOINT
    action_layout = (_checkpoint_action_layout(checkpoint)
                     if action_layout is None else tuple(action_layout))
    if (len(action_layout) != 2 or int(action_layout[0]) < 7
            or int(action_layout[1]) <= 5):
        raise ValueError(
            'LIBERO action_layout must be (Dmax>=7, num_domains>5), got '
            f'{action_layout}.')
    action_layout = (int(action_layout[0]), int(action_layout[1]))
    max_action_dim, _ = action_layout
    if action_layout[1] <= 5:
        raise ValueError(
            f'Checkpoint has {action_layout[1]} domains; LIBERO domain 5 '
            'is unavailable.')
    tokenizer = _tokenizer(checkpoint)
    horizon = 16
    frame_window = horizon + 1
    raw_dim = 7
    statistic_name = f'{suite}_no_noops'
    image_height, image_width = 128, 256
    transforms = [
        dict(
            type='ProcessParquetInputs',
            parquet_keys=[
                'observation.state', 'timestamp', 'actions', 'info', 'stats',
                'action_masks', 'frame_masks'
            ],
            video_keys=[
                'observation.images.image',
                'observation.images.wrist_image',
            ],
            name_mappings={
                'observation.state': ['states'],
                'actions': ['actions'],
            },
            embodiment_id=5,
        ),
        dict(
            type='ProcessCosmos3Prompt',
            tokenizer=tokenizer,
            expected_vocab_size=EDGE_TEXT_CONFIG['vocab_size'],
            expected_special_token_ids=dict(
                pad_token_id=EDGE_TEXT_CONFIG['pad_token_id'],
                bos_token_id=EDGE_TEXT_CONFIG['bos_token_id'],
                **EDGE_SPECIAL_TOKENS,
            ),
            max_len=512,
            cfg_dropout_rate=cfg_dropout_rate,
            action_metadata=dict(
                append_viewpoint=False,
                frame_window_size=frame_window,
                conditioning_fps=20.0,
                video_height=256,
                video_width=256,
            ),
        ),
        dict(
            type='ResizeImagesWithPad',
            height=image_height,
            width=image_width,
            pad_value=127,
            pad_direction='center',
        ),
        dict(type='SimpleNormalizeImages'),
        dict(
            type='NormalizeStatesAndActions',
            action_dim=max_action_dim,
            state_dim=max_action_dim,
            state_key='proprio',
            action_key='action',
            norm_type='mean_std',
        ),
        dict(
            type='BuildCosmos3Sequence',
            raw_action_dim=raw_dim,
            mode='policy',
            frame_window_size=frame_window,
            prepend_state_to_action=False,
            conditioning_fps=20.0,
        ),
        dict(type='PrepareVideo', num_views=2,
             frame_window_size=frame_window),
    ]
    model = _model(
        checkpoint,
        raw_action_dim=raw_dim,
        horizon=horizon,
        tuning=tuning,
        enable_vision_loss=tuning not in {'action', 'lora'},
        action_layout=action_layout,
        action_init=action_init,
        fresh_action_domain_ids=fresh_action_domain_ids,
    )
    train_dataloader = dict(
        per_device_batch_size=1,
        per_device_num_workers=2,
        dataset=dict(
            type='DistributedRepeatingDataset',
            name_mappings={
                'observation.state': ['proprio'],
                'action': ['action'],
            },
            statistic_keys=['observation.state', 'timestamp', 'action'],
            statistic_name=statistic_name,
            datasets=dict(
                type='ParquetDataset',
                data_root_path=data_root_path,
                transforms=transforms,
                action_window_size=horizon,
                action_key='action',
                use_delta=False,
                statistic_name=statistic_name,
                window_start_idx=0,
                frame_window_size=frame_window,
                require_full_window=True,
                task_indices=task_indices,
                max_windows=max_windows,
            ),
        ),
    )
    eval_cfg = dict(
        type='LiberoEvalRunner',
        task_suite_name=suite,
        model_family='cosmos3',
        norm_stats_key=statistic_name,
        eval_chunk_size=10,
        resize_size=256,
        num_trials_per_task=eval_trials,
        task_ids=eval_task_ids,
        num_steps_wait=10,
        seed=seed,
        inference_seed=seed,
        num_inference_steps=10,
        guidance=1.0,
        shift=10.0,
        enable_mixed_precision_training=True,
        mixed_precision_dtype='bf16',
        dataset=dict(
            type='LiberoParquetEvalDataset',
            img_buffer_len=1,
            extra_tensor_keys=[
                'conditioning_fps', 'action_fps', 'raw_action_dim',
                'prepend_state_to_action', 'negative_text_token_ids'
            ],
            transforms=[
                dict(
                    type='ProcessLiberoEvalInputs',
                    img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
                    embodiment_id=5,
                ),
                dict(
                    type='SetCosmos3ActionMetadata',
                    conditioning_fps=20.0,
                    action_fps=20.0,
                    raw_action_dim=raw_dim,
                    prepend_state_to_action=False,
                ),
                dict(
                    type='TransformImage',
                    image_resize_strategy='letterbox',
                    input_sizes=[[3, image_height, image_width]] * 2,
                    means=[[127.5, 127.5, 127.5]] * 2,
                    stds=[[127.5, 127.5, 127.5]] * 2,
                ),
                dict(
                    type='ProcessCosmos3Prompt',
                    tokenizer=tokenizer,
                    expected_vocab_size=EDGE_TEXT_CONFIG['vocab_size'],
                    expected_special_token_ids=dict(
                        pad_token_id=EDGE_TEXT_CONFIG['pad_token_id'],
                        bos_token_id=EDGE_TEXT_CONFIG['bos_token_id'],
                        **EDGE_SPECIAL_TOKENS,
                    ),
                    max_len=512,
                    cfg_dropout_rate=0.0,
                    action_metadata=dict(
                        append_viewpoint=False,
                        frame_window_size=frame_window,
                        conditioning_fps=20.0,
                        video_height=256,
                        video_width=256,
                    ),
                    output_key='lang_tokens',
                    output_attention_mask_key='lang_masks',
                    negative_output_key='negative_text_token_ids',
                ),
                dict(
                    type='LiberoProprioFromInputs',
                    norm_type='mean_std',
                    state_dim=max_action_dim,
                    pos_key='robot0_eef_pos',
                    quat_key='robot0_eef_quat',
                    gripper_key='robot0_gripper_qpos',
                    out_key='states',
                ),
                dict(type='PrepareVideo', num_views=2, frame_window_size=1),
            ],
        ),
        denormalize_action=dict(
            type='DenormalizeLiberoAction',
            norm_type='mean_std',
            action_dim=raw_dim,
        ),
    )
    return dict(
        seed=seed,
        model=model,
        inference_model=deepcopy(model),
        train_dataloader=train_dataloader,
        runner=_runner(
            tokenizer, tuning, max_steps, save_interval=save_interval),
        eval=eval_cfg,
    )


JIKUN_LIBERO_ROOT = (
    '/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/libero_mujoco3.3.2'
)


def build_jikun_libero_single_config(suite='libero_spatial'):
    """Build one JiKun-aligned, full-schedule Edge LIBERO experiment.

    This keeps the data, image layout, joint video/action objective, optimizer,
    epoch budget, and evaluation settings of the Nano single-suite recipe. The
    Edge/Nemotron backbone and its partial generation-tower tuning are the
    intentional algorithm-specific differences.
    """
    suite_to_dir = {
        'libero_spatial': 'libero_spatial_no_noops_lerobot',
        'libero_object': 'libero_object_no_noops_lerobot',
        'libero_goal': 'libero_goal_no_noops_lerobot',
        'libero_10': 'libero_10_no_noops_lerobot',
    }
    if suite not in suite_to_dir:
        raise ValueError(f'Unsupported LIBERO suite: {suite}')

    statistic_name = f'{suite}_no_noops'
    data_root_path = str(Path(JIKUN_LIBERO_ROOT) / suite_to_dir[suite])
    config = build_libero_config(
        suite=suite,
        tuning='partial',
        max_steps=1,
        eval_trials=50,
        data_root_path=data_root_path,
        action_init='checkpoint',
    )

    for model_key in ('model', 'inference_model'):
        edge_model = config[model_key]
        edge_model['rectified_flow_training_config'].update(
            action_loss_weight=10.0,
            vision_loss_weight=1.0,
            normalize_loss_by_active=False,
        )
        edge_model['rectified_flow_inference_config']['num_steps'] = 30

    train = config['train_dataloader']
    train.update(per_device_batch_size=8, per_device_num_workers=4)
    wrapper = train['dataset']
    wrapper['statistic_name'] = statistic_name
    dataset = wrapper['datasets']
    dataset['statistic_name'] = statistic_name
    # JiKun's implementation reports every source row in the epoch length and
    # resamples invalid episode-tail starts. The hardened dataset prevalidates
    # starts, then repeats them to retain exactly that exposure budget.
    dataset['repeat_to_full_length'] = True

    for index, transform in enumerate(dataset['transforms']):
        if transform['type'] == 'ProcessCosmos3Prompt':
            transform['action_metadata'].update(
                video_height=256,
                video_width=128,
            )
        elif transform['type'] == 'ResizeImagesWithPad':
            dataset['transforms'][index] = dict(
                type='ResizeImages', height=128, width=128)
        elif transform['type'] == 'BuildCosmos3Sequence':
            transform['mode'] = 'joint'

    runner = config['runner']
    runner.update(
        max_steps=None,
        max_epochs=12,
        save_epoch_interval=1,
        save_iter_interval=10_000,
        max_keep_ckpts=2,
        grad_accumulation_steps=1,
    )
    # JiKun's single-suite recipe applies one learning rate to all trainable
    # parameters, unlike the mixed recipe's 5x action-head multiplier.
    runner['optimizer'].update(lr=1e-4, paramwise_learning_rate={})
    runner['metric'].update(
        active_trackers=('jsonl', 'wandb'),
        grad_accumulation_steps=1,
    )
    runner['lr_scheduler'] = dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.0,
    )
    runner['collator'] = dict(
        type='Cosmos3Collator',
        tensor_keys=[
            'images',
            'actions',
            'embodiment_ids',
            'raw_action_dim',
            'conditioning_fps',
            'action_fps',
        ],
        sequence_keys=['text_token_ids'],
        list_keys=['sequence_plan'],
        meta_keys=['task_description', 'stats', 'info', 'timestamp'],
        pad_id=11,
    )

    eval_cfg = config['eval']
    eval_cfg.update(
        task_suite_name=suite,
        norm_stats_key=statistic_name,
        resize_size=128,
        num_trials_per_task=50,
        task_ids=None,
        num_inference_steps=30,
    )
    for transform in eval_cfg['dataset']['transforms']:
        if transform['type'] == 'TransformImage':
            transform.update(
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 128, 128], [3, 128, 128]],
            )
        elif transform['type'] == 'ProcessCosmos3Prompt':
            transform['action_metadata'].update(
                video_height=256,
                video_width=128,
            )

    return config
