import numpy as np
import util

from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.
    
    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
  
    # *** START CODE HERE ***
    x_train, y_train= util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid= util.load_dataset(eval_path, add_intercept=True)
    logreg= LogisticRegression()
    logreg.fit(x_train,y_train)
    
    print("Theta is: ", logreg.theta)
    print("The accuracy of Logistic Regression on training set is: ", np.mean(logreg.predict(x_train) == y_train)) 
    
    util.plot(x_valid, y_valid, logreg.theta, pred_path)
    print("The accuracy of Logistic Regression on validation set is: ", np.mean(logreg.predict(x_valid) == y_valid))
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def loaddataset(self, n ,train_path ):
        x= util.load_dataset(train_path, label_col='y', add_intercept= True)
        if (n==1 or n==2):
            y= np.zeros(len(x[0]))
            for i in range(len(x[0])):
                y[i]= x[0][i][n-1]
            return y
        else:
            y= np.zeros(len(x[1]))
            for i in range(len(x[1])):
                y[i]= x[1][i]
            return y


    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        def sigmoid( x, theta):
          
            return (1/(1+np.exp(-np.dot(x , theta))))
        
        def gradient( x, y,theta):
            
            return np.dot(x.T, (y - sigmoid(x,theta)))
        
        def hessian(x,theta ):
            
            h_theta_x = np.reshape(sigmoid(x,theta), (-1, 1))
            return np.dot(x.T, h_theta_x * (1 - h_theta_x) * x)
        
        def next_theta(x,y,theta):
            return theta + np.dot(np.linalg.inv(hessian(x,theta)), gradient(x,y,theta))
        
        m,n= x.shape
        if self.theta==None:
            self.theta=np.zeros(n)
        
        old_theta= self.theta
        new_theta= next_theta(x,y,old_theta)
        
        j=0
        while np.linalg.norm(new_theta - old_theta, 1) >= self.eps and j<self.max_iter:
            old_theta= new_theta
            new_theta= next_theta(x,y,old_theta)
        
        self.theta= new_theta
  
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        :param x: Inputs of shape (m, n).
        :return:  Outputs of shape (m,).
        """
        return x @ self.theta >= 0

ds1_training_set_path = '/Users/nimitjain/Desktop/acads/Choding/Machine Learning/PS1/ds1_train.csv'
ds1_valid_set_path = '/Users/nimitjain/Desktop/acads/Choding/Machine Learning/PS1/ds1_valid.csv'
ds2_training_set_path = '/Users/nimitjain/Desktop/acads/Choding/Machine Learning/PS1/ds2_train.csv'
ds2_valid_set_path = '/Users/nimitjain/Desktop/acads/Choding/Machine Learning/PS1/ds2_valid.csv'
pred_path= '/Users/nimitjain/Desktop/acads/Choding/Machine Learning/PS1/LogregPred'

main(ds2_training_set_path,ds2_valid_set_path,pred_path)
