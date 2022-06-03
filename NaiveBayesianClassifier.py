import numpy as np
from requests import post







# features = (x1,x2,x3,x4,x5)
# Labels = Y

class NaiveBayesianClassifier():
    def __init__(self):
        pass
    

    def calculate_prior(self, df, Y):
        unique_values = sorted(list(df[Y].unique()))
        priors = []
        for i in unique_values:
            priors.append(len(df[df[Y] == i]) / len(df))

        return priors


    def calculate_likelihood_gaussian(self, df, feat_name, feat_val, Y, label):
        feat = list(df.columns)
        df = df[df[Y] == label]
        mean, std = df[feat_name].mean(), df[feat_name].std()
        prob_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val - mean)**2 / (2 * std**2)))
        return prob_x_given_y

    def naive_bayes_guassian(self, df, X, Y):
        # feature names
        features = list(df.columns)[:-1]

        #calculate priors
        priors = self.calculate_prior(df, Y)

        Y_pred = []

        for x in X:
            labels = sorted(list(df[Y].unique()))
            likelihoods = [1] * len(labels)
            for j in range(len(labels)):
                for i in range(len(features)):
                    likelihoods[j] *= self.calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

            
            post_probs = [1] * len(labels)
            for i in range(len(labels)):
                post_probs[i] = likelihoods[i] * priors[i]
            
            Y_pred.append(np.argmax(post_probs))
        return np.array(Y_pred)
    