import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.metrics
import matplotlib.pyplot as plot
import pickle

def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding ='latin1')
    return dict

def readData():
    batch_1_dict=unpickle("data_batch_1")
    batch_2_dict=unpickle("data_batch_2")
    batch_3_dict=unpickle("data_batch_3")
    batch_4_dict=unpickle("data_batch_4")
    batch_5_dict=unpickle("data_batch_5")
    test_batch_dict=unpickle("test_batch")
    batches_meta_dict=unpickle("batches.meta")
    #combine test and train sets
    images_X = np.concatenate((batch_1_dict['data'],batch_2_dict['data'],batch_3_dict['data'],batch_4_dict['data'],
                                  batch_5_dict['data'],test_batch_dict['data']))
    labels_Y = np.concatenate((batch_1_dict['labels'], batch_2_dict['labels'], batch_3_dict['labels'],
                                     batch_4_dict['labels'], batch_5_dict['labels'], test_batch_dict['labels']))
    label_names=batches_meta_dict['label_names']
    images_X=np.transpose(images_X.reshape(60000,3,1024),axes=(0,2,1)).reshape(60000,3072)
    return label_names, images_X, labels_Y

def calcMean(label_names, images_X, labels_Y):
    #calculate mean image and first 20 components
    category =[]
    images_mean=[]
    for i in range(len(label_names)):
        category.append(images_X[labels_Y == i])
        images_mean.append(np.mean(category[i],axis=0))
    return category,images_mean

def meanImage(label_names,images_mean):
    #plot of mean image for each class
    for i in range(len(label_names)):
        plot.subplot(2,5,i+1)
        plot.imshow(np.array(images_mean[i]/255).reshape(32,32,3))
        plot.title(label_names[i])
    plot.show()

def bargraph(category,images_mean, label_names):
    error=[]

#    part A :calculate error and plot
    for i in range(len(label_names)):
        pca = sklearn.decomposition.PCA(n_components=20)
        pca.fit(category[i])
        cifar_images_Xhat=np.add(np.dot(pca.transform(category[i],),pca.components_),images_mean[i])
        error.append((np.sum(np.square(category[i]-cifar_images_Xhat)))/len(category[i]))

    plot.bar(np.arange(len(label_names)),np.asarray(error))
    plot.xticks(np.arange(len(label_names)),label_names)
    plot.ylabel("Average sum of squared diff of Orig and reconstructed version")
    plot.title("Error resulting images using PCA 20 bar graph")
    plot.show()

def euclDist(images_mean, label_names):
#    part B:compute Euclidean distance and write csv file
    distMatrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(i+1,10):
            distMatrix[i,j] =np.linalg.norm((images_mean[i]-images_mean[j]))
            distMatrix[j,i] =distMatrix[i,j]
    distMatrix= np.square(distMatrix)
    pd.DataFrame(distMatrix).to_csv("partb_distances.csv",header=None,index=None)

    #part B:  MDS
    A = np.identity(10)-np.ones((10,10))/10
    W=-0.5*A.dot(distMatrix).dot(np.transpose(A))
    eigenvalues,eigenvectors=np.linalg.eigh(W)
    idx=eigenvalues.argsort()[::-1]
    eigenvalues=eigenvalues[idx]
    eigenvectors=eigenvectors[:,idx]
    scalePoint= eigenvectors[:,:2].dot(np.diag(np.sqrt(eigenvalues[:2])))

    fig,ax = plot.subplots()
    ax.scatter(scalePoint[:,0],scalePoint[:,1])

    for i, txt in enumerate (label_names):
        ax.annotate(txt,(scalePoint[i][0],scalePoint[i][1]))

    plot.show()

def similarity(label_names, category):
#    part C
#    E(A---->B) and E(B---->A)
    erroratob=np.zeros((10,10))
    distancespartc=np.zeros((10,10))

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            pca = sklearn.decomposition.PCA(n_components=20).fit(category[j])
            erroratob[i, j] = np.mean(np.sum((pca.inverse_transform(pca.transform(category[i]))-category[i]) ** 2, axis=1))

    for i in range(10):
        for j in range(10):
            distancespartc[i][j] = 0.5 * (erroratob[i][j] + erroratob [j][i])

    pd.DataFrame(distancespartc).to_csv("partc_distances.csv",header=None,index=None)
    # part C:  MDS
    A = np.identity(10) - np.ones((10, 10)) / 10
    W = -0.5 * A.dot(distancespartc).dot(np.transpose(A))
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    scalePoint = eigenvectors[:, :2].dot(np.diag(np.sqrt(eigenvalues[:2])))

    fig, ax = plot.subplots()
    ax.scatter(scalePoint[:, 0], scalePoint[:, 1])

    for i, txt in enumerate(label_names):
        ax.annotate(txt, (scalePoint[i][0], scalePoint[i][1]))

    plot.show()

def main():
    label_names, images_X, labels_Y = readData()
    category, images_mean = calcMean(label_names, images_X, labels_Y)
    meanImage(label_names,images_mean)
    bargraph(category,images_mean, label_names)
    euclDist(images_mean, label_names)
    similarity(label_names, category)
main()
