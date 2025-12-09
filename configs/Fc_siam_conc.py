net = 'fc_siam_conc'

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

exp_name = 'FC_siam_conc_WHUCD_BS4_epoch200/{}'.format(net)

######################## dataset_config ######################
_base_ = [
    './_base_/WHUCD_config.py',
]
num_class = 2
ignore_index = 255

######################## model_config ########################
model_config = dict(

    decoderhead=dict(
        type='SiamUnet_conc',  # 要与你实现的模型类名称一致
        input_nbr=3,           # RGB 输入通道
        label_nbr=2            # 输出类别数（变化/不变）

    )
)

######################## loss_config #########################
loss_config = dict(
    type='myLoss',
    loss_name=['CELoss', 'dice_loss'],
    loss_weight=[0.4, 0.6],
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
