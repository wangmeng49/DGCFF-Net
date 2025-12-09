net = 'stade_cdnet'

######################## base_config #########################
epoch = 200
gpus = [0]
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
logging_interval = 'epoch'
resume_ckpt_path = None
monitor_val = 'val_change_f1'
monitor_test = ['test_change_f1']
argmax = True

test_ckpt_path = r''

exp_name = 'STADENet_LEVIRCD_BS4_epoch200/{}'.format(net)

######################## dataset_config ######################
_base_ = [
    './_base_/LEVIRCD_config.py',
]
num_class = 2
ignore_index = 255

######################## model_config ########################
model_config = dict(
    backbone=dict(
        type='Base',
        name='Resnet18',
    ),
    decoderhead=dict(
        type='BASE_Transformer',  # 要与你实现的模型类名称一致
input_nc=3,
    output_nc=2,
    with_pos='learned',
    resnet_stages_num=4,
    token_len=4,
    token_trans=True,
    enc_depth=1,
    dec_depth=1,
    dim_head=64,
    decoder_dim_head=64,
    tokenizer=True,
    if_upsample_2x=True,
    pool_mode='max',
    pool_size=2,
    backbone='resnet18',
    decoder_softmax=True,
    with_decoder_pos='learned',
    with_decoder=True

    )
)

######################## loss_config #########################
loss_config = dict(
    type='myLoss',
    loss_name=['CELoss', 'dice_loss'],
    loss_weight=[0.7, 0.3],
    param=dict(
        CELoss=dict(
            ignore_index=255,
            reduction='mean',
            weight=[0.6, 1.4]  # 背景类、变化类
        ),
        dice_loss=dict(
            eps=1e-7
        )
    )
)

##################### optimizer_config ########################
optimizer_config = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=1e-5,
        lr_mode='multistep'
    ),
    scheduler=dict(
        type='multistep',
        milestones=[50, 100, 150],
        gamma=0.9
    )
)

########################## metrics ############################
metric_cfg1 = dict(
    task='multiclass',
    average='micro',
    num_classes=num_class,
)

metric_cfg2 = dict(
    task='multiclass',
    average='none',
    num_classes=num_class,
)
