"""
@Date: April 3rd, 2011
@Author: Di, Wei (vanessa.wdi@gmail.com)
This file is to test the modified Libsvm
Added functions to output the corresponding value:
    |w|^2	
    rho		
    nSV		
    probA	
    probB
With |w|^2 and the predict-value, 
we can then compute the distance of each sample to the hyperplane.
We only test the C-SVM part.
"""

# Import the alternated lib-path file
import sys
sys.path.append('./python')

from svm import *
from svmutil import *


def run_test(plot_tag = False, data_path = None, tr_name = None, ts_name = None):
    # Import libsvm-formate data
    if data_path == None:
        data_path = './data/letter'
        tr_name = 'letter.scale'
        ts_name = 'letter.scale.t'
    
    train_y,train_x = svm_read_problem(data_path + '\\' + tr_name)
    test_y,test_x = svm_read_problem(data_path + '\\' + ts_name)

    para = dict(c=512, g=0.0039)  # this is arbitrary value
    libsvm_options = ' '.join(['-'+ k + ' '+str(v) for (k,v) in para.items()])

    no_tr = min( int(len(train_y) * 0.3), 1000)
    no_ts = min( int(len(test_y) * 0.3), 100)
    svm_m = svm_train(train_y[:no_tr], train_x[:no_tr], libsvm_options)
    p_label, p_acc, p_val = svm_predict(test_y[:no_ts], test_x[:no_ts], svm_m)

    disH = distance_to_hyperplane(p_val, svm_m, signed=False)

    import pprint
    print "== dir(svm_m) ==="
    pprint.pprint(dir(svm_m))
    
    return svm_m, disH

 
if __name__ == '__main__':
    m, disH = run_test(plot_tag = False)
    import matplotlib.pyplot as plt
    plt.imshow(disH)
    plt.show()

    check = ['get_w2_vector', 'get_rho_vector', 'get_nSV', 'get_probA_vector','get_probB_vector']
    for i in check:
        print "svm-model has attr:" + i + ' -- ' + str(hasattr(m, 'get_nSV'))
