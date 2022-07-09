import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
"""
class G:
    def __init__(self):
        None

    def get(self):
        with tf.variable_scope("conv1"):
            self.en = tf.Variable(66, dtype=tf.float32, name="Y")
        with tf.variable_scope("conv2"):
            self.de = tf.Variable(99, dtype=tf.float32, name="Z")
        return self.en

graph = tf.Graph()
with graph.as_default():
    g = G()
    en = g.get()
    saver = tf.train.Saver()
    reader = tf.train.NewCheckpointReader('./test/test3')

    var_to_shape_map = reader.get_variable_to_shape_map()
    for var_name in var_to_shape_map.keys():
            # 用reader获取变量值
        var_value = reader.get_tensor(var_name)

        print("var_name", var_name)
        print("var_value", var_value)

    for var in tf.global_variables():
        print(var)
        if var.name.startswith('conv1'):
            print(reader.get_tensor(var.name[:-2]))


    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver.restore(sess, tf.train.latest_checkpoint('./test'))
        var = tf.global_variables()
        print(var)
        print(var[0].name[:-2])
        print(var[1].name[:-2])
        #print(sess.run(en))

        for var in tf.global_variables():
            if var.name.startswith('conv1'):
                sess.run(var.assign(reader.get_tensor(var.name[:-2])))
        #print(sess.run(en))
        saver.save(sess, './test/test3')

# music_encoder init
reader = tf.train.NewCheckpointReader('./Check_point1/model_epoch_80_gs_125120')
var_to_shape_map = reader.get_variable_to_shape_map()
for var_name in var_to_shape_map.keys():
    # 用reader获取变量值
    var_value = reader.get_tensor(var_name)
    print("var_name", var_name)
    print("var_value", var_value)
for var in tf.global_variables():
    if var.name.startswith('music_encoder') and (var.name.find("Adam") == -1):
        var_reader = var
        sess.run(var.assign(reader.get_tensor(var_reader.name.replace("music_encoder", "encoder_MLM", 1)[:-2])))

def pcatf(x,dim = 2):
    with tf.name_scope("PCA"):
        m,n= tf.to_float(x.get_shape()[0]),tf.to_int32(x.get_shape()[1])
        #assert not tf.assert_less(dim,n)
        mean = tf.reduce_mean(x,axis=1)
        x_new = x - tf.reshape(mean,(-1,1))
        cov = tf.matmul(x_new,x_new,transpose_a=True)/(m - 1)
        e,v = tf.linalg.eigh(cov,name="eigh")
        e_index_sort = tf.math.top_k(e,sorted=True,k=dim)[1]
        v_new = tf.gather(v,indices=e_index_sort)
        pca = tf.matmul(x_new,v_new,transpose_b=True)
    return pca

d = tf.constant([[1,1,1,1],[2,2,2,2],[10,19.9,10,19]])
newdata = pcatf(d)
sess = tf.Session()
newdata = sess.run(newdata)
plt.scatter(newdata[:, 0], newdata[:, 1])
plt.show()
"""

class PCAtest():
    def __init__(self, k):
        self.k = k

    def loadIris(self):
        data = np.load('en_array.npy')
        data = np.mean(data,1)
        print(data)
        return data

    def stand_data(self, data):
        mean_vector = np.mean(data, axis=0)
        return mean_vector, data - mean_vector

    def getCovMat(self, standData):
        return np.cov(standData, rowvar=0)

    def getFValueAndFVector(self, covMat):
        fValue, fVector = np.linalg.eig(covMat)
        return fValue, fVector

    def getVectorMatrix(self, fValue, fVector):
        fValueSort = np.argsort(-fValue)
        # print(fValueSort)
        fValueTopN = fValueSort[:self.k]
        # print(fValueTopN)
        return fVector[:, fValueTopN]

    def getResult(self, data, vectorMat):
        return np.dot(data, vectorMat)

if __name__ == "__main__":
    pca = PCAtest(2)
    data = pca.loadIris()
    print(1)
    (mean_vector, standdata) = pca.stand_data(data)
    print(2)
    cov_mat = pca.getCovMat(standdata)
    print(3)
    fvalue, fvector = pca.getFValueAndFVector(cov_mat)
    print(4)
    fvectormat = pca.getVectorMatrix(fvalue, fvector)
    #print("最终需要的特征向量：\n%s" % fvector)
    newdata = pca.getResult(standdata, fvectormat)
    #print(newdata)
    #print("最终重构结果为:\n{}".format(np.mat(newdata) * fvectormat.T + mean_vector))
    plt.scatter(newdata[:, 0], newdata[:, 1])
    plt.axis([-20, 20, -15, 15])
    plt.show()
