from predata import *
from evaluation import two_cls_access
import scipy.io as sio

mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patches)
mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patches)
x1_true = np.zeros((height*width, args.patches, args.patches, band), dtype=float)
x2_true = np.zeros((height*width, args.patches, args.patches, band), dtype=float)
for i in range(height):
    for j in range(width):
        x1_true[i*width+j, :, :, :] = mirror_image_t1[i:(i+args.patches), j:(j+args.patches), :]
        x2_true[i*width+j, :, :, :] = mirror_image_t2[i:(i+args.patches), j:(j+args.patches), :]
x1_true_band = gain_neighborhood_band(x1_true, band, args.band_patches, args.patches)
x2_true_band = gain_neighborhood_band(x2_true, band, args.band_patches, args.patches)
x1_true_band = torch.from_numpy(x1_true_band.transpose(0, 2, 1)).type(torch.FloatTensor)
x2_true_band = torch.from_numpy(x2_true_band.transpose(0, 2, 1)).type(torch.FloatTensor)
Label_true = Data.TensorDataset(x1_true_band, x2_true_band)
label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)
print('------测试数据加载完毕------')

model.load_state_dict(torch.load("log/{}.pth".format(args.dataset)))
model.eval()

pre_u = test_epoch(model, label_true_loader)
prediction_matrix = np.zeros((height, width), dtype=float)
for i in range(height):
    for j in range(width):
        prediction_matrix[i, j] = pre_u[i * width + j] + 1

label = loadmat('Datasets/farm450/label.mat')['label']  # 修改
prediction_matrix[prediction_matrix == 2] = 0
two_cls_access(label, prediction_matrix)

plt.subplot(1, 1, 1)
plt.imshow(prediction_matrix, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.savefig('{}predict.png'.format(args.dataset),  dpi=1200, pad_inches=0.0)

print('-------------Valar Dohaeris-------------')

