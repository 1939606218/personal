datasets = [
    {
        'current_dataset': 'farmland',
        'current_band': 155,
        'num_head': 5
    },
    # {
    #     'current_dataset': 'hermiston',
    #     'current_band': 242,
    #     'num_head': 2
    # },
    # {
    #     'current_dataset': 'river',
    #     'current_band': 198,
    #     'num_head': 6
    # },
    {
        'current_dataset': 'Barbara',
        'current_band': 224,
        'num_head': 4
    },
    {
        'current_dataset': 'BayArea',
        'current_band': 224,
        'num_head': 4
    }
]

current_model = '_AIWSEN'
patch_size = 7
lr = 0.0005
bs_number = 64
epoch_number = 200

phase = ['train', 'test', 'no_gt']
train_set_num = 0.01

def get_config(dataset_info):
    current_dataset = dataset_info['current_dataset']
    current_band = dataset_info['current_band']
    num_head = dataset_info['num_head']

    current = current_dataset + current_model

    data = dict(
        current_dataset=current_dataset,
        train_set_num=train_set_num,
        patch_size=patch_size,
        train_data=dict(
            phase=phase[0]
        ),
        test_data=dict(
            phase=phase[1]
        ),
    )

    # 2. model
    model = dict(
        in_fea_num=155,
    )

    # 3. train
    train = dict(
        optimizer=dict(
            typename='SGD',
            lr=lr,
            momentum=0.9,
            weight_decay=5e-3
        ),
        train_model=dict(
            gpu_train=True,
            gpu_num=1,
            workers_num=12,
            epoch=epoch_number,
            batch_size=bs_number,
            lr=lr,
            lr_adjust=True,
            lr_gamma=0.1,
            lr_step=[35, 70],
            save_folder='./weights/' + current_dataset + '/',
            save_name=current,
            reuse_model=False,
            reuse_file='./weights/' + current + '_Final.pth',
        )
    )

    # 4. test
    test = dict(
        batch_size=1000,
        gpu_train=True,
        gpu_num=0,
        workers_num=8,
        model_weights='./weights/' + current_dataset + '/' + current + '_Final.pth',
        save_name=current,
        save_folder='./result' + '/' + current_dataset
    )

    return data, model, train, test, current_band, num_head