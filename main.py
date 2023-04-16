import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC, LinearSVC
# from skimage import feature
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class Hog_discriptor():
    def __init__(self,img,cell_size = 8, bins = 9, block_size = 2):
        self.img = img
        self.h = self.img.shape[0]
        self.w = self.img.shape[1]
        self.cell_size = cell_size
        self.bins = bins
        self.block_size = block_size
        self.orient_per_hist = 180//self.bins
        self.num_cell_x = self.w // self.cell_size  # Number of cell in x-axis
        self.num_cell_y = self.h // self.cell_size  # Number of cell in y-axis

    def global_gradient(self):
        xkernel = np.array([[-1, 0, 1]])
        ykernel = np.array([[-1], [0], [1]])
        dx = cv2.filter2D(self.img, cv2.CV_32F, xkernel)
        dy = cv2.filter2D(self.img, cv2.CV_32F, ykernel)

        # Sobel Filter
        # dx = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=3)
        # dy = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=3)

        magnitude = np.sqrt(np.square(dx) + np.square(dy)) #calculate magnitude
        orientation = (np.rad2deg(np.arctan2(dy,dx)))%180  #calculate orientation

        return magnitude,orientation

    def histogram(self):
        hist_tensor = np.zeros((self.num_cell_y, self.num_cell_x, self.bins))  # Historam tensor for an image
        magnitude, orientation = self.global_gradient()
        for cx in range(self.num_cell_x):
            for cy in range(self.num_cell_y):
                orient = orientation[cy*self.cell_size:self.cell_size*(cy+1),cx*self.cell_size:self.cell_size*(cx+1)]
                mag = magnitude[cy * self.cell_size:self.cell_size*(cy+1),cx * self.cell_size:self.cell_size*(cx+1)]

                hist = np.zeros((1,self.bins))

                for hist_index in range(self.bins):

                    start_orient = hist_index*self.orient_per_hist
                    end_orient = (hist_index+1)*self.orient_per_hist
                    hist[:,hist_index] = np.sum(mag[(orient>=start_orient) & (orient<end_orient)])

                hist_tensor[cy, cx, :] = hist
        return hist_tensor

    def compute_feature(self):
        redundant_cell = self.block_size - 1
        feature_tensor = np.zeros((self.num_cell_y - redundant_cell, self.num_cell_x - redundant_cell,
                                   self.block_size*self.block_size*self.bins))
        hist_tensor = self.histogram()
        eps = 1e-5
        for bx in range(self.num_cell_x - redundant_cell):
            for by in range(self.num_cell_y - redundant_cell):
                tmp = hist_tensor[by:by+self.block_size,bx:bx+self.block_size,:].flatten()
                norm = np.sqrt(np.sum(tmp ** 2) + eps ** 2)
                feature_tensor[by, bx, :] = tmp / norm

        return feature_tensor.flatten()

class SVM_Classifier():
    def __init__(self,learning_rate = 0.01,lambda_parameter = 0.01,epoches = 100):
        self.learning_rate = learning_rate
        self.lambda_parameter = lambda_parameter
        self.epoches = epoches

    def transform_label(self,y,classes):
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)

        interger_encoded =label_encoder.transform(y)# Interger encode such as 0,1,2,...
        svm_encoded = np.where(interger_encoded == 0, -1,1)# -1, 1
        return svm_encoded

    def cost(self,X,y):
        u = np.dot(X,self.w) + self.b
        return (np.sum(np.maximum(0,1 - u@y))) + self.lambda_parameter/2 * np.sum(self.w*self.w)

    def fit(self,X,y):
        self.classes = list(set(y))
        y = self.transform_label(y,self.classes)
        row,collomn = X.shape #each row is a data point
        self.w = np.random.randn(collomn)
        self.b = np.random.randn(1)
        count = 0
        Loss = []
        Acc = []
        for epoch in range(self.epoches):
            for i,xi in enumerate(X):
                if(y[i] * (np.dot(xi,self.w) + self.b) >= 1):
                    dw = self.lambda_parameter * self.w
                    db = 0
                else:
                    dw = self.lambda_parameter * self.w - np.dot(xi,y[i])
                    db = - y[i]

                count +=1
                if (count%1000 == 0):
                    Loss.append([self.cost(X,y),count])
                    y_pred = self.predict(X)
                    Acc.append([accuracy_score(y, y_pred),count])
                    print(f"Iteration: {count}, Loss = {self.cost(X, y)}, Accuracy = {accuracy_score(y, y_pred)*100} %")

                self.w = self.w - self.learning_rate * dw
                self.b = self.b - self.learning_rate * db

            self.learning_rate = self.learning_rate/(0.001*epoch+1) #update Learning rate
        self.plot_result(Loss,Acc)

    def plot_result(self,Loss,Accuracy):
        Loss = np.array(Loss)
        Accuracy = np.array(Accuracy)
        plt.figure()
        plt.plot(Loss[:,1],Loss[:,0])
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.figure()
        plt.plot(Accuracy[:,1],Accuracy[:,0])
        plt.title("Training Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.show()

    def predict(self,X):
        output = np.dot(X,self.w) + self.b
        y_pred = np.sign(output)
        y_pred = np.asarray(y_pred, dtype='int')
        return y_pred

    def score(self,y_test,y_predict):
        y_test = self.transform_label(y_test,self.classes)
        return(accuracy_score(y_test, y_predict))

    def plot_wrong_image(self,X,hog,y):
        y_pred = self.predict(hog)
        y_transformed = self.transform_label(y, self.classes)
        X = np.array(X)
        X_wrong = X[y_pred !=  y_transformed]
        y_wrong = y_pred[y_pred !=  y_transformed]
        no_pic = 4
        plt.figure(figsize= (8,4))

        for i in range(no_pic):
            plt.subplot(1,no_pic,i+1)
            img = X_wrong[i,:,:]
            if(y_wrong[i] == -1):
                plt.title("Fresh")
            else:
                plt.title("Rotten")
            plt.axis('off')
            plt.imshow(img, cmap = 'gray')
        plt.show()

def LoadData(Data_dir):
    data = []
    label = []
    for folder in os.listdir(Data_dir):
        for file in os.listdir(os.path.join(Data_dir,folder)):
            file_path = os.path.join(Data_dir,folder,file)
            img = cv2.imread(file_path)
            img = cv2.resize(img,(64,128))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(gray)
            label.append(folder)
    return data,label

def train(data_dir):
    data,label = LoadData(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42, stratify=label)
    hog_train = []
    hog_test = []

    print("--------------HOG phase-------------")

    for img in X_train:

        hog = Hog_discriptor(img, cell_size = 8, block_size = 2 ,bins = 9)
        feat = hog.compute_feature()
        #Using library for HOG below
        # feat = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        #                   cells_per_block=(2, 2), transform_sqrt=False, block_norm="L2")
        hog_train.append(feat)
    hog_train = np.array(hog_train)
    for img in X_test:
        hog = Hog_discriptor(img, cell_size=8, block_size=2, bins=9)
        feat = hog.compute_feature()
        # Using library for HOG below
        # feat = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        #                   cells_per_block=(2, 2), transform_sqrt=False, block_norm="L2")
        hog_test.append(feat)
    hog_test = np.array(hog_test)


    print("--------------SVM Phase-------------")
    model_svm = SVM_Classifier(learning_rate=0.01, lambda_parameter=0.01, epoches=100)
    model_svm.fit(hog_train, y_train)
    y_predict = model_svm.predict(hog_test)
    print('Độ chính xác: ', model_svm.score(y_test, y_predict))
    model_svm.plot_wrong_image(X_test, hog_test, y_test)

    # Using library for SVM below
    # model_svm = SVC(kernel="linear", C=100)
    # model_svm.fit(hog_train, y_train)
    # y_predict = model_svm.predict(hog_test)
    # print('Độ chính xác: ', model_svm.score(hog_test, y_test))
    test = cv2.imread('vertical_flip_Screen Shot 2018-06-08 at 2.31.33 PM.png')
    hog = Hog_discriptor(test, cell_size=8, block_size=2, bins=9)
    feat = hog.compute_feature()
    print(model_svm.predict(feat))

if __name__ == "__main__":
    data_dir = 'FruitData'
    train(data_dir)





