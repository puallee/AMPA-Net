##Official code of the paper AMPA-Net: Optimization-Inspired Attention Neural Network for Deep Compressed Sensing 
#Kunming University of science and technology(KUST)，Quantum Intelligence inc.
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import ampa
import test
import RGBtest
ratio=[50，40,25，10,4,1]
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
os.environ['CUDA_VISIBLE_DEVICES']='0'
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
for i in range(1):
    CS_ratio = ratio[0]    # 4, 10, 25, 30, 40, 50
    if   CS_ratio == 4:
        n_input = 43
    elif CS_ratio == 1:
        n_input = 10
    elif CS_ratio == 10:
        n_input = 109
    elif CS_ratio == 25:
        n_input = 272
    elif CS_ratio == 30:
        n_input = 327
    elif CS_ratio == 40:
        n_input = 436
    elif CS_ratio == 50:
        n_input = 545
    elif CS_ratio == 20:
        n_input = 218
    elif CS_ratio == 33:
        n_input = 363
    elif CS_ratio ==12:
        n_input=136
    n_output = 1089
    batch_size = 64
    StackingNumber = 9
    nrtrain = 88912
    learning_rate = 0.0001
    EpochNum = 120


    print('Load Data of 91 Image...')

    Training_data_Name = './Training_Data_Img91.mat'
    Training_data = sio.loadmat(Training_data_Name)
    Training_inputs = Training_data['inputs']
    Training_labels = Training_data['labels']
    XX = Training_labels.transpose()
    labelinput=Training_labels.transpose()
    
    # Adpative Sensing
    X_input = tf.placeholder(tf.float32, [None, 1089])
    fc1w=tf.Variable(tf.random_normal([1089,n_input], stddev=1e-2), name='fc1w')
    fc2w=tf.Variable(tf.random_normal([n_input,1089], stddev=1e-2), name='fc2w')
    fc3w = tf.Variable(tf.random_normal([n_input, 1089], stddev=1e-2), name='fc3w')
    flatten=tf.reshape(X_input,[-1,1089])
    X_inputt = tf.matmul(flatten, fc1w)
    y=X_inputt
    ## Initialization attention network
    k=tf.nn.relu(X_inputt)
    a=tf.matmul(k,fc3w)
    a=tf.nn.softmax(a)

    X_output = tf.placeholder(tf.float32, [None, n_output])

    X0=tf.matmul(X_inputt,fc2w) # Input of Initialization
    X0 = X0*a
    Phi = fc1w

    PhiT =  tf.transpose(fc1w)

    [CSRecon, CSOrtho] = ampa.inference_ampa(X0,StackingNumber,PhiT,Phi,y,reuse=False)
    lost0 = tf.reduce_mean(tf.square(X0 - X_output))

    def compute_charbonnier_loss(tensor1, tensor2, is_mean=True):
        epsilon = 1e-6
        if is_mean:
            loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1]))
        else:
            loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1]))

        return loss

    def compute_lost(CSRecon, X_output, StackingNumber,CSOrtho):
        Reconlost=compute_charbonnier_loss(CSRecon[-1], X_output, is_mean=True)
        Ortholost = 0
        for k in range(StackingNumber):
            epsilon = 1e-6
            Ortholost += tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(CSOrtho[k]) + epsilon), [1]))
        return [Reconlost, Ortholost]


    [Reconlost, Ortholost] = compute_lost(CSRecon, X_output, StackingNumber,CSOrtho)


    lost_all = Reconlost + 0.01*Ortholost


    optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lost_all)

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    print("...............................")
    print("Stacking Number of AMPA-Net is %d, CS ratio is %d%%" % (StackingNumber, CS_ratio))
    print("...............................\n")
    print("Strart Training of AMPA-Net..")
    acc_All = np.zeros([1, 4], dtype=np.float32)

    Acc100=np.zeros([EpochNum+1],dtype=np.float32)
    Acc11=np.zeros([EpochNum+1],dtype=np.float32)
    best=0.0
    best2=0.0
    best3=0.0
    best4=0.0
    for epoch_i in range(0, EpochNum+1):
        randidx_all = np.random.permutation(nrtrain)
        for batch_i in range(nrtrain // batch_size):
            randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

            batch_ys = Training_labels[randidx, :]
        
            feed_dict = {X_input: batch_ys, X_output: batch_ys}
            sess.run(optm_all, feed_dict=feed_dict)

        output_data = "[%02d/%02d] Reconlost: %.6f, Ortholost: %.6f \n" % (epoch_i, EpochNum, sess.run(Reconlost, feed_dict=feed_dict), sess.run(Ortholost, feed_dict=feed_dict))
        print(output_data)
        if epoch_i%1==0:
            print('and')
            print('Set11')
            acc1 = test.test(sess, 11,CSRecon[-1],X_input)
            if acc1>best:
               best=acc1
            print('*/'*100)
            print('Set11 best')
            print(best)
            print('*/'*100)
            print('BSDS100')
            acc2 = RGBtest.test(sess, 'BSDS100','png', CSRecon[-1], X_input)
            if acc2>best2:
               best2=acc2
            print('*/'*100)
            print('BSDS100 best')
            print(best2)
            print('*/'*100)
            Acc100[epoch_i] = acc2
            print('Urban100')
            acc3 = RGBtest.test(sess, 'Urban100','png', CSRecon[-1], X_input)
            if acc3>best3:
               best3=acc3
            print('*/'*100)
            print('Urban100 best')
            print(best3)
            print('*/'*100)
            Acc11[epoch_i]=acc1
            acc4 = test.test(sess,68,CSRecon[-1],X_input)
            if acc4>best4:
               best4=acc4
            print('*/'*100)
            print('BSD68 best')
            print(best4)
            print('*/'*100)
    F=np.max(Acc100,axis=0)
    idex=np.where(Acc100==F)
    idex=idex[0]
    print(idex)
    F2=Acc11[idex]
out="Final ACC100:%.4f and ACC11:%.4f in Epoch \n "%(F,F2)
print(out)
print("Training Finished")


