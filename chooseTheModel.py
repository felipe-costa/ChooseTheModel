"""ChooseTheModel
Author: Felipe Soares Costa
E-mail: felipesibh@gmail.com
Date: 24/11/2017
"""
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import datetime
class Chooser(object):
    """Chooser: Class used to compare and choose the best model to fit the data
      Example:
    """
    def __init__(self, models, x, y):
        """Constructor
        Args:
            models: List of models to compare
            x: Input variables (features)
            y: Ground truth values
        """
        self.__models = models
        self.__predicted = []
        self.__errors = []
        self.__scores = []
        self.__sdt = []
        self.__x = x
        self.__y = y
        self.__names = []
        self.__choosen = None
        self.__df = None

    def get_models(self):
        """Get the models array
        """
        return self.__models

    def get_predicted(self):
        """Get the predicted array
        """
        return self.__predicted

    def get_erros(self):
        """Get the errors array
        """
        return self.__errors

    def get_socores(self):
        """Get the scores array
        """
        return self.__scores

    def get_sdt(self):
        """Get the std array
        """
        return self.__sdt

    def get_names(self):
        """Get the names array
        """
        return self.__names

    def get_choosen(self):
        """Get the choosen model
        """
        return self.__choosen


    Models = property(fget=get_models, fset=None)
    Predicted = property(fget=get_predicted, fset=None)
    Errors = property(fget=get_erros, fset=None)
    Scores = property(fget=get_socores, fset=None)
    Std = property(fget=get_sdt,fset=None)
    Names = property(fget=get_names,fset=None)
    Choosen = property(fget=get_choosen,fset=None)

    def choose(self):
        """Chose the best model
        """
        scores = []
        results = []
        sdt = []
        names = []
        r2 = []
        x_names = []
        inicio = []
        termino = []
        seed = 7
        for model in self.__models:
            for xname,xvalues in self.__x:
                inicio.append(datetime.datetime.now())
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, xvalues, self.__y, cv=kfold, scoring='neg_mean_squared_error')
                termino.append(datetime.datetime.now())
                Y_pred = model_selection.cross_val_predict(model, xvalues, self.__y, cv=kfold)
                scores.append(np.sqrt(abs(cv_results.mean())))
                sdt.append(abs(cv_results.std()))
                results.append(abs(cv_results))
                r2.append(r2_score(self.__y, Y_pred, multioutput='variance_weighted'))
                names.append(model.__class__.__name__)
                x_names.append(xname)

        col={'Grupo': x_names, 'RMSE':scores, 'Standard Deviation': sdt, 'R2 Score': r2, 'Início': inicio,'Término': termino }
        df=pd.DataFrame(data=col,index=names)
        df = df.sort_values(by='RMSE',ascending=True)
        model_name = df.iloc[0].name
        x_name = df.iloc[0]['Grupo']

        self.__scores = r2
        self.__errors = scores
        self.__sdt = sdt
        self.__names = names
        self.__df = df
        return self.__df

    def chart(self,names=None):
        if(names == None):
            names = self.__names
        df = self.__df[self.__df['Grupo'].isin(names)]
        df = df.sort_values(by='RMSE',ascending=True)
        df.plot(kind='bar',title="Model Compare")


    def predict(self,model, x):
        """Use the choosen model to predict new values
        """
        seed = 7
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        return model_selection.cross_val_predict(model, x, self.__y, cv=kfold)

    def test(self,grupo, model):
        """Simple test the choosen model
        """
        x = []
        for n,values in self.__x:
            if grupo == n:
                x = values

        seed = 7
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        y_pred = model_selection.cross_val_predict(model, x, self.__y, cv=kfold)

        fig, ax = plt.subplots()
        ax.scatter(self.__y, y_pred, edgecolors=(0, 0, 0))
        ax.plot([self.__y.min(), self.__y.max()], [self.__y.min(), self.__y.max()], 'k-', lw=4)
        ax.set_xlabel('Real')
        ax.set_ylabel('Predicted')
        plt.show()

