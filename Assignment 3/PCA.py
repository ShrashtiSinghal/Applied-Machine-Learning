import csv
import numpy as np

dims = 4
samples = 150

def readData(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        list = []
        for row in reader:
            list.append(row)
        file.close()

        array = np.array(list[1:]).astype(np.float)
        return array

def getMean(data_in):
    return np.mean(data_in, axis=0)

def reconstructData(data_in, data_mean, iris_cov, useNoiseless):
    # NORMALIZATION
    data_mean_repeat = np.tile(data_mean, [samples, 1])
    data_norm = (data_in - data_mean_repeat)

    # COVARIANCE
    if(useNoiseless == False):      # c
        data_cov = np.cov(data_norm, rowvar=0)
    else:                           # n
        data_cov = iris_cov

    # EIGENVALUE, EIGENVECTOR
    data_eigval, data_eigvec = np.linalg.eig(data_cov)
    index = data_eigval.argsort()[::-1]
    data_eigval = data_eigval[index]
    data_eigvec = data_eigvec[:, index]

    # REDUCE DIMENSION AND RECONSTRUCTION
    data_recons = []
    for dim in range(dims + 1):
        data_feature = data_eigvec[:, :dim].reshape(dims, dim)
        data_reduce = np.dot(data_norm, data_feature).reshape(samples, dim)
        data_reconstruction = np.dot(data_reduce, data_feature.T) + data_mean_repeat
        data_recons.append(data_reconstruction)

    return data_recons

def MSE(x1, x2):
    mse = []
    for dim in range(dims+1):
        square = pow((x1[dim] - x2), 2)
        mean = np.sum(square)/samples
        mse.append(mean)
    return mse

def Data1(iris_in,iris_mean,iris_cov):
    # dataI
    mse = []
    mse_I = []
    dataI_in = readData('dataI.csv')
    dataI_mean = getMean(dataI_in)
    # dataI + n
    dataI_n_reconstructions = reconstructData(dataI_in, iris_mean, iris_cov, True)
    dataI_n_mse = MSE(dataI_n_reconstructions, iris_in)
    mse_I.append(dataI_n_mse)
    # dataI + c
    dataI_c_reconstructions = reconstructData(dataI_in, dataI_mean, iris_cov, False)
    dataI_c_mse = MSE(dataI_c_reconstructions, iris_in)
    mse_I.append(dataI_c_mse)
    # mse
    mse.append(mse_I)
    return mse, dataI_c_reconstructions

def Data2(mse, iris_in,iris_mean,iris_cov):
    # dataII
    mse_II = []
    dataII_in = readData('dataII.csv')
    dataII_mean = getMean(dataII_in)
    # dataII + n
    dataII_n_reconstructions = reconstructData(dataII_in, iris_mean, iris_cov, True)
    dataII_n_mse = MSE(dataII_n_reconstructions, iris_in)
    mse_II.append(dataII_n_mse)
    # dataII + c
    dataII_c_reconstructions = reconstructData(dataII_in, dataII_mean, iris_cov, False)
    dataII_c_mse = MSE(dataII_c_reconstructions, iris_in)
    mse_II.append(dataII_c_mse)
    # mse
    mse.append(mse_II)
    return mse

def Data3(mse, iris_in,iris_mean,iris_cov):
    # dataIII
    mse_III = []
    dataIII_in = readData('dataIII.csv')
    dataIII_mean = getMean(dataIII_in)
    # dataIII + n
    dataIII_n_reconstructions = reconstructData(dataIII_in, iris_mean, iris_cov, True)
    dataIII_n_mse = MSE(dataIII_n_reconstructions, iris_in)
    mse_III.append(dataIII_n_mse)
    # dataIII + c
    dataIII_c_reconstructions = reconstructData(dataIII_in, dataIII_mean, iris_cov, False)
    dataIII_c_mse = MSE(dataIII_c_reconstructions, iris_in)
    mse_III.append(dataIII_c_mse)
    # mse
    mse.append(mse_III)
    return mse

def Data4(mse, iris_in,iris_mean,iris_cov):
    # dataIV
    mse_IV = []
    dataIV_in = readData('dataIV.csv')
    dataIV_mean = getMean(dataIV_in)
    # dataIV + n
    dataIV_n_reconstructions = reconstructData(dataIV_in, iris_mean, iris_cov, True)
    dataIV_n_mse = MSE(dataIV_n_reconstructions, iris_in)
    mse_IV.append(dataIV_n_mse)
    # dataIV + c
    dataIV_c_reconstructions = reconstructData(dataIV_in, dataIV_mean, iris_cov, False)
    dataIV_c_mse = MSE(dataIV_c_reconstructions, iris_in)
    mse_IV.append(dataIV_c_mse)
    # mse
    mse.append(mse_IV)
    return mse


def Data5(mse,iris_in,iris_mean,iris_cov):
    # dataV
    mse_V = []
    dataV_in = readData('dataV.csv')
    dataV_mean = getMean(dataV_in)
    # dataV + n
    dataV_n_reconstructions = reconstructData(dataV_in, iris_mean, iris_cov, True)
    dataV_n_mse = MSE(dataV_n_reconstructions, iris_in)
    mse_V.append(dataV_n_mse)
    # dataV + c
    dataV_c_reconstructions = reconstructData(dataV_in, dataV_mean, iris_cov, False)
    dataV_c_mse = MSE(dataV_c_reconstructions, iris_in)
    mse_V.append(dataV_c_mse)
    # mse
    mse.append(mse_V)
    return mse

def WriteData(mse, dataI_c_reconstructions):
    # WRITE dataI
    with open("ankushs2-recon.csv", "w") as output_file:
        output_writer = csv.writer(output_file)
        # write header
        fileHeading = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
        output_writer.writerow(fileHeading)
        # write content
        output_writer.writerows(dataI_c_reconstructions[2])

    # WRITE MSE
    with open("ankushs2-numbers.csv", "w") as output_file:
        output_writer = csv.writer(output_file)
        # write header
        fileHeading = ["0N", "1N", "2N", "3N", "4N", "0c", "1c", "2c", "3c", "4c"]
        output_writer.writerow(fileHeading)
        # write content
        content = []
        for dim in range(dims+1):
            temp = []
            for iter in range(dims+1):
                temp.append(mse[dim][0][iter])
            for iter in range(dims+1):
                temp.append(mse[dim][1][iter])
            content.append(temp)
        output_writer.writerows(content)

def main():

    # iris
    iris_in = readData('iris.csv')
    iris_mean = getMean(iris_in)
    iris_mean_repeat = np.tile(iris_mean, [samples, 1])
    iris_norm = (iris_in - iris_mean_repeat)
    iris_cov = np.cov(iris_norm, rowvar=0)
    mse,dataI_c_reconstructions = Data1(iris_in,iris_mean,iris_cov)
    mse = Data2(mse,iris_in,iris_mean,iris_cov)
    mse = Data3(mse,iris_in,iris_mean,iris_cov)
    mse = Data4(mse,iris_in,iris_mean,iris_cov)
    mse = Data5(mse,iris_in,iris_mean,iris_cov)
    WriteData(mse, dataI_c_reconstructions)

main()