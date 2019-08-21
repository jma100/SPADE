import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import numpy as np

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)
#mu_collector = np.zeros((len(dataloader.dataset), opt.z_dim))
#var_collector = np.zeros((len(dataloader.dataset), opt.z_dim))
#z_collector = np.zeros((len(dataloader.dataset), opt.z_dim))
mu_collector = []
var_collector = []
z_collector = []
path_collector = []

model = Pix2PixModel(opt)
model.eval()

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    z, mu, logvar = model(data_i, mode='encode_only')
    num, _ = mu.size()
    start = i*opt.batchSize
    end = start+num
    mu_collector.append(mu.cpu().detach().numpy())
    var_collector.append(logvar.cpu().detach().numpy())
    z_collector.append(z.cpu().detach().numpy())
#    mu_collector[start:end, :] = mu.cpu().detach().numpy()
#    var_collector[start:end, :] = logvar.cpu().detach().numpy()
#    z_collector[start:end, :] = z.cpu().detach().numpy()
    path_collector += data_i['path']


from sklearn.cluster import KMeans
import shutil
n_clusters = 10
mu_collector = np.concatenate(mu_collector, axis=0)
var_collector = np.concatenate(var_collector, axis=0)
z_collector = np.concatenate(z_collector, axis=0)
new = np.concatenate((mu_collector, var_collector), axis=1)
print('starts!')
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(new)
print('done')
outdir = '/data/vision/torralba/virtualhome/realvirtualhome/SPADE/checkpoints/%s/cluster_%s' % (opt.name, opt.which_epoch)
labels = kmeans.labels_
if not os.path.isdir(outdir):
    os.makedirs(outdir)

if os.path.isdir(outdir):
    shutil.rmtree(outdir)
for i in range(n_clusters):
    paths = [path_collector[idx] for idx, n in enumerate(labels) if n == i]
    folder = os.path.join(outdir, 'folder%02d' % i)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(os.path.join(outdir,'folder%02d.txt' % i) , 'w') as f:
        for path in paths:
            f.write(path+'\n')
            shutil.copy(path, folder)

centers = kmeans.cluster_centers_
np.save(os.path.join(outdir, "cluster_%s.npy" % n_clusters), centers)
#import pdb; pdb.set_trace()
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
epoch = int(opt.which_epoch)
reduced_data = PCA(n_components=2).fit_transform(mu_collector[:300])
plt.clf()
new_kmeans = KMeans(n_clusters=3, random_state=0)
new_kmeans.fit(reduced_data)
y_kmeans = new_kmeans.predict(reduced_data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_kmeans, s=50, cmap='viridis')
new_centers = new_kmeans.cluster_centers_
plt.scatter(new_centers[:, 0], new_centers[:, 1], c='black', s=200, alpha=0.5); 
plt.savefig(os.path.join(outdir, "cluster_mu_%03d_3.png" % epoch))

reduced_data = PCA(n_components=2).fit_transform(var_collector[:300])
plt.clf()
new_kmeans = KMeans(n_clusters=3, random_state=0)
new_kmeans.fit(reduced_data)
y_kmeans = new_kmeans.predict(reduced_data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_kmeans, s=50, cmap='viridis')
new_centers = new_kmeans.cluster_centers_
plt.scatter(new_centers[:, 0], new_centers[:, 1], c='black', s=200, alpha=0.5); 
plt.savefig(os.path.join(outdir, "cluster_var_%03d_3.png" % epoch))

reduced_data = PCA(n_components=2).fit_transform(new[:300])
plt.clf()
new_kmeans = KMeans(n_clusters=3, random_state=0)
new_kmeans.fit(reduced_data)
y_kmeans = new_kmeans.predict(reduced_data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_kmeans, s=50, cmap='viridis')
new_centers = new_kmeans.cluster_centers_
plt.scatter(new_centers[:, 0], new_centers[:, 1], c='black', s=200, alpha=0.5); 
plt.savefig(os.path.join(outdir, "cluster_together_%03d_3.png" % epoch))
#import pdb; pdb.set_trace()

reduced_data = PCA(n_components=2).fit_transform(mu_collector[:300])
plt.clf()
new_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
new_kmeans.fit(reduced_data)
y_kmeans = new_kmeans.predict(reduced_data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_kmeans, s=50, cmap='viridis')
new_centers = new_kmeans.cluster_centers_
plt.scatter(new_centers[:, 0], new_centers[:, 1], c='black', s=200, alpha=0.5); 
plt.savefig(os.path.join(outdir, "cluster_mu_%03d_%02d.png" % (epoch, n_clusters)))

reduced_data = PCA(n_components=2).fit_transform(var_collector[:300])
plt.clf()
new_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
new_kmeans.fit(reduced_data)
y_kmeans = new_kmeans.predict(reduced_data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_kmeans, s=50, cmap='viridis')
new_centers = new_kmeans.cluster_centers_
plt.scatter(new_centers[:, 0], new_centers[:, 1], c='black', s=200, alpha=0.5); 
plt.savefig(os.path.join(outdir, "cluster_var_%03d_%02d.png" % (epoch, n_clusters)))

reduced_data = PCA(n_components=2).fit_transform(new[:300])
plt.clf()
new_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
new_kmeans.fit(reduced_data)
y_kmeans = new_kmeans.predict(reduced_data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_kmeans, s=50, cmap='viridis')
new_centers = new_kmeans.cluster_centers_
plt.scatter(new_centers[:, 0], new_centers[:, 1], c='black', s=200, alpha=0.5);
plt.savefig(os.path.join(outdir, "cluster_together_%03d_%02d.png" % (epoch, n_clusters)))

