net = 'dgcffnet'
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

exp_name = 'LEVIRCD_BS4_epoch200/{}'.format(net)

######################## dataset_config ######################
_base_ = [
    './_base_/LEVIRCD_config.py',
]
num_class = 2
ignore_index = 255

######################### model_config #########################
model_config = dict(
    backbone = dict(
        type = 'Base',
        name = 'Resnet18'
    ),
    decoderhead = dict(
        type = 'DGCFFNet',
        num_class = 2,
        channel_list = [64, 128, 256, 512],
        transform_feat = 128,
        layer_num = 4
    )
)
######################## loss_config #########################

loss_config = dict(
    type = 'myLoss',
    loss_name = ['CELoss', 'dice_loss'],
    loss_weight = [0.4, 0.6],
    param = dict(
        CELoss=dict(
            ignore_index=255,
            reduction='mean',
            weight=[0.6, 1.4]
        ),
        dice_loss = dict(
            eps=1e-7
        )
    )
)


######################## optimizer_config ######################
optimizer_config = dict(
    optimizer = dict(
        type = 'AdamW',
        lr =1e-4,
        #momentum=0.9,
        weight_decay=1e-5,
        lr_mode = "multistep"
    ),
    scheduler = dict(
        type = 'multistep',
        milestones = [50, 100,150],
        gamma = 0.9
    )
)

metric_cfg1 = dict(
    task = 'multiclass',
    average='micro',
    num_classes = num_class, 
)

metric_cfg2 = dict(
    task = 'multiclass',
    average='none',
    num_classes = num_class, 
)