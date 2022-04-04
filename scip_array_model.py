#!/usr/bin/env python
# encoding: utf-8

"""
@author: zn—zhangmazi
@desc: Efficient array operation class with pyscipopt based on scip native api
"""

import numpy as np
from pyscipopt import Model


class ScipArrayModel(Model):
    """
    array operation class for scip native api

    """
    class VarArray(np.ndarray):
        """
        scip array var, derive from ndarray

        """
        def add_model(self, model):
            """
            add scip model for getVal method

            Parameters
            ----------
            model: scip model
            """
            self.Model = model

        def __eq__(self, other):
            return self, other, '=='

        def __ne__(self, other):
            return self, other, '!='

        def __le__(self, other):
            return self, other, '<='

        def __ge__(self, other):
            return self, other, '>='

        @property
        def value(self):
            """
            make scip var element of VarArray object to numerical type of ndarray

            Returns
            -------
            var : ndarray
                the values of scip variable solved
            """
            var_copy = self.copy()
            for i, v in enumerate(var_copy.ravel()):
                var_copy.ravel()[i] = self.Model.getVal(v)
            var = np.array(var_copy)
            return var

    def array(self, array):
        """
        Just like numpy np.array function, make a iterable
        object to VarArray object for scip constrains

        Parameters
        ----------
        array : iterable
            list or ndarray of scip variable

        Returns
        -------
        arr : VarArray
        """
        arr = np.array(array)
        arr = self.VarArray(shape=arr.shape, dtype=arr.dtype, buffer=arr)
        arr.add_model(self)
        return arr

    def add_var(self, shape, v_type='C', lb=None, ub=None):
        """
        creat VarArray variables

        Parameters
        ----------
        shape : tuple
            variable shape
        v_type : str
            variable type
        lb : lower bound of the variable, use None for -infinity (Default value = -infinity)
        ub : upper bound of the variable, use None for +infinity (Default value = +infinity)

        Returns
        -------
        var : VarArray object
            VarArray variable for matrix multiplication
        """
        var = np.zeros(shape, object)
        var_ravel = var.ravel()
        if v_type == 'B':
            for i in range(len(var_ravel)):
                var_ravel[i] = self.addVar(vtype='B')

        elif v_type == 'C':
            for i in range(len(var_ravel)):
                var_ravel[i] = self.addVar(vtype='C', lb=lb, ub=ub)

        else:
            for i in range(len(var_ravel)):
                var_ravel[i] = self.addVar(vtype=v_type, lb=lb, ub=ub)

        var = self.VarArray(shape=shape, dtype=object, buffer=var)
        var.add_model(self)
        return var

    def add_cons(self, cons):
        """
        add the array constraints to scip model

        Parameters
        ----------
        cons : list
            constraints of ndarray object
        """
        if type(cons) is not list:
            raise TypeError('the constraints type must be list')
        for c in cons:
            var, other, operator = c
            shape = var.shape
            try:
                other = np.zeros(shape=shape) + other
            except:
                raise ValueError('constraints could not be broadcast together with shapes %s %s' % (shape, other.shape))
            for i, v in enumerate(var.ravel()):
                if operator == '==':
                    self.addCons(v == other.ravel()[i])
                elif operator == '!=':
                    self.addCons(v != other.ravel()[i])
                elif operator == '<=':
                    self.addCons(v <= other.ravel()[i])
                elif operator == '>=':
                    self.addCons(v >= other.ravel()[i])

    def setObjective(self, obj, *args, **kwargs):
        """
        the VarArray to the Expr type

        Parameters
        ----------
        obj : VarArray or Expr
            the objective

        """
        if type(obj) == ScipArrayModel.VarArray:
            if obj.shape == () or obj.shape == (1, ):
                obj = obj.reshape((1,))[0]
            else:
                raise AssertionError(
                    'given coefficients shape are neither () or (1, ) but %s' % obj.shape)
        return super().setObjective(obj, *args, **kwargs)
