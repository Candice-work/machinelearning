import pickle
import matplotlib.pyplot as plt
import numpy
import sklearn.linear_model as lin
import bonnerlib2D as bl2d
import numpy as np
from numpy import random
import sklearn.utils as ut
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

with open('cluster_data.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)
Xtrain,Ttrain = dataTrain
Xtest,Ttest = dataTest

clf = lin.LogisticRegression(multi_class='multinomial',solver='lbfgs')
clf.fit(Xtrain,Ttrain)

print('Question 1')
print('-------------')
scoreTrain = clf.score(Xtrain, Ttrain) #store accuracy on training data
scoreTest = clf.score(Xtest, Ttest) #store accuracy on testing data
print('\nQuestion 1(a):')
print('The accuracy of the classifier on the training data: {}'.format(scoreTrain)) #print out the accuracy
print('The accuracy of the classifier on the testing data: {}'.format(scoreTest)) #print out the accuracy

bl2d.plot_data(Xtrain, Ttrain) # make training data into a graph
bl2d.boundaries(clf) # create boundaries to classify three clusters
plt.title('Question 1(b): decision boundaries for linear classification') # make a title
plt.show() # show the plot

W = clf.coef_ # assign weight
b = clf.intercept_ # assign bias

def predict(X, W, b):
    z = np.dot(W, X.T).T + b # sub X, W, b into the vectorized equation
    z_n = np.exp(z - np.max(z)) # get nominator of softmax
    sol = z_n / z_n.sum() # softmax of z
    return np.argmax(sol, axis = 1)

Y_1 = clf.predict(Xtest)
Y_2 = predict(Xtest, W, b)

print('\nQuestion 1(e):')
print('The squareed magnitude of Y1 and Y2 is {}'.format(np.linalg.norm(Y_1 - Y_2)))

def one_hot(Tint): #converts integer target values to one-hot encodings
    # print(np.unique(Tint))
    no_class = np.max(Tint) + 1 # 4 columns
    no_entry = len(Tint) # 8 rows
    C = np.arange(no_entry)
    Tint = np.array(Tint)
    Thot = numpy.zeros((no_entry, no_class))
    Thot[C, Tint] = 1
    return Thot

#print(one_hot([0,1,2,3,0,1,2,3]))
print('\nQuestion 1(f):')
print(one_hot([4,3,2,3,2,1,2,1,0]))

print('\nQuestion 2')
print('-------------')

print('\nQuestion 2(a):')

def pred_XW(W, X):
    z = X @ W
    z_n = np.exp(z - np.max(z))  # get nominator of softmax
    sol = z_n / z_n.sum()  # softmax of z
    return np.argmax(sol, axis=1)

def grad(W, X, T):  # help function fr calculating the gradient descent
    # print(W.shape, X.shape, T.shape)
    z = X @ W
    z_n = np.exp(z - np.max(z, axis=1, keepdims=True))  # get nominator of softmax
    y = z_n / z_n.sum(axis=1).reshape(-1, 1)
    return ((X.T @ (y - T))) / X.shape[0]

def GDlinear(I,lrate): # batch gradient descent for multi-class linear classification
    np.random.seed(7)
    print('learning rate =', lrate)
    Train_b = np.column_stack((np.ones(Xtrain.shape[0]), Xtrain))
    Test_b = np.column_stack((np.ones(Xtest.shape[0]), Xtest))
    Train_hot = one_hot(Ttrain)
    Test_hot = one_hot(Ttest)
    w = random.rand(Train_hot.shape[1], Train_b.shape[1])/10000 # initialize the weights

    # print(Train_b)
    ent_train = [] #average cross entropy on training data
    ent_test = [] #average cross entropy on test data
    accu_train = [] #accuracy on training data
    accu_test = [] #accuracy on test data

    def pred_XW(W, X):
        z = X @ W
        z_n = np.exp(z - np.max(z, axis=1, keepdims=True))  # get nominator of softmax
        sol = z_n / z_n.sum(axis=1).reshape(-1, 1)
        return sol

    for i in range(I):
        # print(grad(w, Train_b, Train_hot))
        w -= lrate * grad(w, Train_b, Train_hot)
        y_train = pred_XW(w, Train_b)
        ent_train.append(np.mean(-np.sum((Train_hot * np.log(y_train + 1e-9)), axis=1)))

        y_test = pred_XW(w, Test_b)
        ent_test.append(np.mean(-np.sum((Test_hot * np.log(y_test + 1e-9)), axis=1)))

        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        train_acc = np.sum(y_train == Ttrain) / y_train.shape[0]
        accu_train.append(train_acc)

        test_acc = np.sum(y_test == Ttest) / y_test.shape[0]
        accu_test.append(test_acc)


    return(ent_train, ent_test, accu_train, accu_test, w)

ent_train, ent_test, accu_train, accu_test, w = GDlinear(10000, 0.1)

#Q2(a)(vii) Loss entropy graph
plt.semilogx(ent_train)
plt.semilogx(ent_test)
plt.title('Question 2(a): Training and test loss v.s. iterations')
plt.xlabel('Iteration number')
plt.ylabel('Cross entropy')
plt.legend(('training data', 'test data'))
plt.show()

#Q2(a)(viii) Accuracy graph
plt.semilogx(accu_train)
plt.semilogx(accu_test)
plt.title('Question 2(a): Training and test accuracy v.s. iterations')
plt.xlabel('Iteration number')
plt.ylabel('Accuracy')
plt.legend(('training data', 'test data'))
plt.show()

#Q2(a)(ix) Loss entropy graph for test data 50 on
plt.semilogx(ent_test[50:], "r-")
plt.title('Question 2(a): test loss from iteration 50 on')
plt.xlabel('Iteration number')
plt.ylabel('Cross entropy')
plt.show()

#Q2(a)(ix) Loss entropy graph for training data 50 on
plt.semilogx(ent_train[50:], "b-")
plt.title('Question 2(a): training loss from iteration 50 on')
plt.xlabel('Iteration number')
plt.ylabel('Cross entropy')
plt.show()

#Q2(a)(xi) Training accuracy
print('\nQuestion 2(a)(xi):')
print('Training accuracy: {}'.format(accu_train[-1]))
print('Difference between the training accuracy: {}'.format(accu_train[-1] - scoreTrain))

#Q2(a)(xii) Test accuracy
print('\nQuestion 2(a)(xii):')
print('Test accuracy: {}'.format(accu_test[-1]))
print('Difference between the test accuracy: {}'.format(accu_test[-1] - scoreTest))

#Q2(a)(xiii) Decision boundaries
print('\nQuestion 2(a)(xiii):')
bl2d.plot_data(Xtrain, Ttrain)
weight = w[:, 1:]
bias = w[:, 0]
bl2d.boundaries2(weight, bias, predict)
plt.title('Question 2(a): decision boundaries for linear classification') # make a title
plt.show() # show the plot

# Q2(a)(xiii) Loss entropy graph for 5 different learning rates
# ent_train_10, ent_test_10, accu_train_10, accu_test_10, w_10 = GDlinear(10000, 10) # with learning rate 10
# ent_train_1, ent_test_1, accu_train_1, accu_test_1, w_1 = GDlinear(10000, 1) # with learning rate 1
# ent_train_01, ent_test_01, accu_train_01, accu_test_01, w_01 = GDlinear(10000, 0.1) # with learning rate 0.1
# ent_train_001, ent_test_001, accu_train_001, accu_test_001, w_001 = GDlinear(10000, 0.001) # with learning rate 0.001
# ent_train_00001, ent_test_00001, accu_train_00001, accu_test_00001, w_00001 = GDlinear(10000, 0.00001) # with learning rate 0.00001
#
# plt.semilogx(ent_train_10)
# plt.semilogx(ent_train_1)
# plt.semilogx(ent_train_01)
# plt.semilogx(ent_train_001)
# plt.semilogx(ent_train_00001)
# plt.title('Question 2(a): Training loss on 5 different learning rates v.s. iterations')
# plt.xlabel('Iteration number')
# plt.ylabel('Cross entropy')
# plt.legend(('lrate = 10', 'lrate = 1', 'lrate = 0.1', 'lrate = 0.001', 'lrate = 0.00001'))
# plt.show()

ent_train_1, ent_test_1, accu_train_1, accu_test_1, w_1 = GDlinear(10000, 0.1) # smooth curve
print('avg entropy of training: {}'.format(np.sum(ent_train_1)/len(ent_train_1)))
print('avg entropy of test: {}'.format(np.sum(ent_test_1)/len(ent_test_1)))
print('avg accuracy of training: {}'.format(np.sum(accu_train_1)/len(accu_train_1)))
print('avg accuracy of test: {}'.format(np.sum(accu_test_1)/len(accu_test_1)))
print('weight: {}'.format(w_1))
plt.semilogx(ent_train_1)
plt.semilogx(ent_test_1)
plt.title('Question 2(a): Training and test loss on lrate = 0.1')
plt.xlabel('Iteration number')
plt.ylabel('Cross entropy')
plt.legend(('training data', 'test data'))
plt.show()

print('\nQuestion 2(d):')

def SGDlinear(I, batch_size, lrate0, alpha, kappa):
    print(batch_size, lrate0, alpha, kappa)
    np.random.seed(7)
    lrate = lrate0
    Train_b = np.column_stack((np.ones(Xtrain.shape[0]), Xtrain))
    Test_b = np.column_stack((np.ones(Xtest.shape[0]), Xtest))
    Train_hot = one_hot(Ttrain)
    Test_hot = one_hot(Ttest)
    w = random.rand(Train_hot.shape[1], Train_b.shape[1]) / 10000  # initialize the weights

    ent_train = []  # average cross entropy on training data
    ent_test = []  # average cross entropy on test data
    accu_train = []  # accuracy on training data
    accu_test = []  # accuracy on test data

    def pred_XW(W, X):
        z = X @ W
        z_n = np.exp(z - np.max(z, axis=1, keepdims=True))  # get nominator of softmax
        sol = z_n / z_n.sum(axis=1).reshape(-1, 1)
        # sol = softmax(z)
        # return np.argmax(sol, axis=1)
        return sol

    for i in range(I):
        if i >= kappa:
            lrate = lrate0 / (1 + alpha * (i - kappa))
        all_shuf = ut.shuffle(np.column_stack([Train_b, Train_hot]), random_state=7)
        X_shuf, T_shuf = all_shuf[:, :3], all_shuf[:, 3:]
        for mini_batch_num in range(Train_b.shape[0] // batch_size + 1):
            batch_X = X_shuf[mini_batch_num * batch_size: (mini_batch_num + 1) * batch_size, :]
            batch_T = T_shuf[mini_batch_num * batch_size: (mini_batch_num + 1) * batch_size, :]
            # print(batch_X.shape)
            w -= lrate * grad(w, batch_X, batch_T)
        y_train = pred_XW(w, Train_b)
        ent_train.append(np.mean(-np.sum((Train_hot * np.log(y_train + 1e-9)), axis=1)))

        y_test = pred_XW(w, Test_b)
        ent_test.append(np.mean(-np.sum((Test_hot * np.log(y_test + 1e-9)), axis=1)))

        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        train_acc = np.sum(y_train == Ttrain) / y_train.shape[0]
        accu_train.append(train_acc)

        test_acc = np.sum(y_test == Ttest) / y_test.shape[0]
        accu_test.append(test_acc)

    return (ent_train, ent_test, accu_train, accu_test, w)


ent_train, ent_test, accu_train, accu_test, w = SGDlinear(500, 30, 1e-2, 1e-5, 390)

# Q2(d)(vii) Loss entropy graph
plt.semilogx(ent_train)
plt.semilogx(ent_test)
plt.title('Question 2(d): Training and test loss v.s. iterations')
plt.xlabel('Iteration number')
plt.ylabel('Cross entropy')
plt.legend(('training data', 'test data'))
plt.show()

# Q2(d)(viii) Accuracy graph
# x = np.logspace(0, np.log10(I), num=len(ent_train))
# plt.semilogx(x, accu_train, x, accu_test)
plt.semilogx(accu_train)
plt.semilogx(accu_test)
plt.title('Question 2(d): Training and test accuracy v.s. iterations')
plt.xlabel('Iteration number')
plt.ylabel('Accuracy')
plt.legend(('training data', 'test data'))
plt.show()

# Q2(d)(xi) Training accuracy
print('\nQuestion 2(d)(xi):')
print('Training accuracy: {}'.format(accu_train[-1]))
print('Difference between the training accuracy: {}'.format(accu_train[-1] - scoreTrain))

# Q2(d)(xii) Test accuracy
print('\nQuestion 2(d)(xii):')
print('Test accuracy: {}'.format(accu_test[-1]))
print('Difference between the test accuracy: {}'.format(accu_test[-1] - scoreTest))

print('\nQuestion 3')
print('-------------')

print('\nQuestion 3(a)')
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(Xtrain, Ttrain)

scoreTrain_qda = qda.score(Xtrain, Ttrain) #store accuracy on training data
scoreTest_qda = qda.score(Xtest, Ttest) #store accuracy on testing data
print('The accuracy of the classifier on the training data: {}'.format(scoreTrain_qda)) #print out the accuracy
print('The accuracy of the classifier on the testing data: {}'.format(scoreTest_qda)) #print out the accuracy

bl2d.plot_data(Xtrain, Ttrain) # make training data into a graph
bl2d.boundaries(qda) # create boundaries to classify three clusters
plt.title('Question 3(a): decision boundaries on Quadratic Discriminant Analysis') # make a title
plt.show() # show the plot

print('\nQuestion 3(b)')
gnb = GaussianNB()
gnb.fit(Xtrain, Ttrain)

scoreTrain_gnb = gnb.score(Xtrain, Ttrain) #store accuracy on training data
scoreTest_gnb = gnb.score(Xtest, Ttest) #store accuracy on testing data
print('The accuracy of the classifier on the training data: {}'.format(scoreTrain_gnb)) #print out the accuracy
print('The accuracy of the classifier on the testing data: {}'.format(scoreTest_gnb)) #print out the accuracy

bl2d.plot_data(Xtrain, Ttrain) # make training data into a graph
bl2d.boundaries(gnb) # create boundaries to classify three clusters
plt.title('Question 3(b): decision boundaries on GaussianNB') # make a title
plt.show() # show the plot

print('\nQuestion 3(f)')
def EstMean(X,T):
    N = np.sum(T, axis = 0)
    mu = (np.dot(X.T, T)/N).T
    return mu

Mean = EstMean(Xtrain, one_hot(Ttrain))
diff_mean = np.sum((Mean - qda.means_)**2)
print('The total squared difference of mean matrices: {}'. format(diff_mean))

print('\nQuestion 3(g)')
def EstCov(X,T):
    mean = np.reshape(EstMean(X, T), [1, EstMean(X, T).shape[0], EstMean(X,T).shape[1]])
    N = np.sum(T, axis = 0)
    X = np.reshape(X, [X.shape[0], 1, X.shape[1]])
    A = X - mean
    A_i = np.reshape(A, [A.shape[0], A.shape[1], A.shape[2], 1])
    A_j = np.reshape(A, [A.shape[0], A.shape[1], 1, A.shape[2]])
    B = A_i * A_j
    T = np.reshape(T, [T.shape[0], T.shape[1], 1, 1])
    C = T * B
    D = np.sum(C, axis = 0)
    N = np.reshape(N, [N.shape[0], 1, 1])
    sigma = D / (N - 1)
    return sigma

Cov = EstCov(Xtrain, one_hot(Ttrain))
diff_cov = np.sum((Cov - qda.covariance_)**2)
print('The total squared difference of covariance matrices: {}'. format(diff_cov))

print('\nQuestion 3(h)')
def EstPrior(T):
    N = T.shape[0]
    T = np.sum(T, axis=0)
    prior = T/N
    return prior

Prior = EstPrior(one_hot(Ttrain))
diff_prior = np.sum((Prior - qda.priors_)**2)
print('The total squared difference of prior probability vectors: {}'. format(diff_prior))

print('\nQuestion 3(i)')
print('I dont know')
# def EstPost(mean,cov,prior,X):
#     mn = multivariate_normal.pdf(X, mean, cov)
#     print(mn)
#     return(mn)
#
# # P = EstPost(EstMean(Xtest, one_hot(Ttest)), EstCov(Xtest, one_hot(Ttest)), EstPrior(one_hot(Ttest)), Xtest)
# P_sk = MultinomialNB()
# # print('Difference in vector of posterior probabilities: {}'.format(np.sum((P - P_sk)**2)))
print('\nQuestion 3(j)')
print('I dont know')
print('\nQuestion 3(k)')
print('I dont know')

print('\nQuestion 4')
print('-------------')

np.random.seed(7)
nn = MLPClassifier(activation='logistic', hidden_layer_sizes=5)
nn.fit(Xtrain, Ttrain)
score_train_nn = nn.score(Xtrain, Ttrain)
score_test_nn = nn.score(Xtest, Ttest)
print('\nQuestion 4(a):')
print('The accuracy of the MPLclassifier on the training data: {}'.format(score_train_nn)) #print out the accuracy
print('The accuracy of the MPLclassifier on the testing data: {}'.format(score_test_nn)) #print out the accuracy

bl2d.plot_data(Xtrain, Ttrain) # make training data into a graph
bl2d.boundaries(nn) # create boundaries to classify three clusters
plt.title('Question4(a): neural net with 5 hidden units') # make a title
plt.show() # show the plot

print('\nQuestion 4(b):')
#hidden layer = 1
np.random.seed(7)
nn_1 = MLPClassifier(activation='logistic', hidden_layer_sizes=1)
nn_1.fit(Xtrain, Ttrain)
#hidden layer = 2
np.random.seed(7)
nn_2 = MLPClassifier(activation='logistic', hidden_layer_sizes=2)
nn_2.fit(Xtrain, Ttrain)
#hidden layer = 4
np.random.seed(7)
nn_4 = MLPClassifier(activation='logistic', hidden_layer_sizes=4)
nn_4.fit(Xtrain, Ttrain)
#hidden layer = 10
np.random.seed(7)
nn_10 = MLPClassifier(activation='logistic', hidden_layer_sizes=10)
nn_10.fit(Xtrain, Ttrain)


fig = plt.figure()
plt.subplot(2, 2, 1)
bl2d.plot_data(Xtrain, Ttrain) # make training data into a graph
bl2d.boundaries(nn_1) # create boundaries to classify three clusters
plt.title("hidden unit = 1")
plt.subplot(2, 2, 2)
bl2d.plot_data(Xtrain, Ttrain) # make training data into a graph
bl2d.boundaries(nn_2) # create boundaries to classify three clusters
plt.title("hidden unit = 2")
plt.subplot(2, 2, 3)
bl2d.plot_data(Xtrain, Ttrain) # make training data into a graph
bl2d.boundaries(nn_4) # create boundaries to classify three clusters
plt.title("hidden unit = 4")
plt.subplot(2, 2, 4)
bl2d.plot_data(Xtrain, Ttrain) # make training data into a graph
bl2d.boundaries(nn_10) # create boundaries to classify three clusters
plt.title("hidden unit = 10")
plt.suptitle("Question 4(b): Neural net decsion boundaries.")
plt.show()