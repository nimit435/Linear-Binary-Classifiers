import numpy as np
import util

from linear_model import LinearModel

ds1_training_set_path = '/Users/nimitjain/Desktop/acads/Choding/Machine Learning/PS1/ds1_train.csv'
ds1_valid_set_path = '/Users/nimitjain/Desktop/acads/Choding/Machine Learning/PS1/ds1_valid.csv'
ds2_training_set_path = '/Users/nimitjain/Desktop/acads/Choding/Machine Learning/PS1/ds2_train.csv'
ds2_valid_set_path = '/Users/nimitjain/Desktop/acads/Choding/Machine Learning/PS1/ds2_valid.csv'


x_train, y_train = util.load_dataset(ds1_training_set_path, add_intercept=False)

def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    
    gda= GDA()
    gda.fit(x_train, y_train)
    
    util.plot(x_valid,y_valid,gda.theta,pred_path)
    print("Theta is: ", gda.theta)
    print("The accuracy of GDA on training set is: ", np.mean(gda.predict(x_train) == y_train))
    print("The accuracy of GDA on validation set is: ", np.mean(gda.predict(x_valid) == y_valid))
    

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.
        
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n= x.shape
        
        phi= np.sum(y)/m
        nu_0= np.dot(x.T,1-y)/np.sum(1-y)
        nu_1= np.dot(x.T, y)/np.sum(y)
        res= np.zeros(((n,n)))
        # nu_y= (1-y[1])*nu_0 + y[1]*nu_1
        for i in range(m):
            nu_yi= (1-y[i])*nu_0 + y[i]*nu_1
            a=np.reshape((x[i]- nu_yi),(n,-1))
            b=np.reshape((x[i]- nu_yi),(n,-1)).T
            
            res= res+ np.dot(a,b)
        sigma =res/m
        # print(sigma)
        inv = np.linalg.inv(sigma)
        a= np.reshape((nu_1-nu_0),(n,-1))
        b= np.reshape((nu_1+nu_0),(n,-1)).T
        theta= np.dot(inv,a)
        theta_0= (-(np.dot(b,theta)))/2-np.log((1-phi)/phi)
        final_theta= np.insert(theta,0,theta_0)
        
        self.theta= final_theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***f
        return util.add_intercept(x) @ self.theta >= 0
        # *** END CODE HERE
        
main(ds2_training_set_path, ds2_training_set_path, '/Users/nimitjain/Desktop/acads/Choding/Machine Learning/PS1/GDA')
