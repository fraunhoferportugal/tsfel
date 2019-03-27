import json
import numpy as np


def compute_dictionary(features_json, default):
    """
    This function computes the dictionary
    :param features_json: (json file)
           list of features
    :param default: (json file)
           default features
    :return: dictionary: (json file)
             complete dictionary
    """

    data = json.load(open(features_json))

    domain = data.keys()
    for atype in domain:
        domain_feats = data[atype].keys()
        for feat in domain_feats:
            # Concatenate two dictionaries
            data[atype][feat] = dict(list(default.items()) + list(data[atype][feat].items()))

    return data


def feat_extract(dictionary, signal_window, signal_label, FS=100, iteration=None):
    """
    This function computes features matrix for one window.
    :param dictionary: (json file)
           list of features
    :param signal_window: (narray-like)
           input from which features are computed, window.
    :param signal_label: (narray-like)
           one of axes of acelerometer.
    # :param : (int)
    #        sampling frequency
    :return: res: (narray-like)
             values of each features for signal.
             nam: (narray-like)
             names of the features
    """
    domain = dictionary.keys()

    #Create global arrays
    func_total = []
    func_names=[]
    imports_total=[]
    parameters_total=[]
    free_total=[]

    for atype in domain:

       domain_feats = dictionary[atype].keys()
       #domain_feats.sort()

       for feat in domain_feats:
            #Only returns used functions
            if dictionary[atype][feat]['use'] != 'no':

                #Read Function Name (generic name)
                func_name=feat
                func_names+=[func_name]

                #Read Function (real name of function)
                function = dictionary[atype][feat]['function']
                func_total += [function]

                #Read Imports
                imports = str(dictionary[atype][feat]['imports'])
                imports_total += [imports]

                #Read Parameters
                parameters = dictionary[atype][feat]['parameters']
                parameters_total += [parameters]

                #Read Free Parameters
                free_parameters = dictionary[atype][feat]['free parameters']
                free_total += [free_parameters]

    #Execute imports
    for imp in set(imports_total):
        exec(imp)

    nfuncs = len(func_total)
    func_results = []
    name=[]
    names=[]

    for i in range(nfuncs):

        if func_total[i]!='correlation': #Correlation receives 2 arguments, separate implementation

            if iteration==None:

                execf =  func_total[i] + '(signal_window'

                if parameters_total[i] != '':
                    execf += ', '+ parameters_total[i]

                if free_total[i] != '' :
                    for n, v in free_total[i].items():
                        #TODO: conversion may loose precision (str)
                        execf += ', ' + n + '=' + str(v[0])

                execf += ')'

            else:

                execf =  func_total[i] + '(signal_window'

                if parameters_total[i] != '':
                    execf += ', '+ parameters_total[i]

                if free_total[i] != '' :

                    for n,v in iteration:
                        #TODO: conversion may loose precision (str)
                        execf += ', ' + n + '=' + str(iteration)

                execf += ')'

            eval_result = eval( execf, locals())


            #Function returns more than one element
            if type(eval_result) == tuple:
                for rr in range(len(eval_result)):
                    if np.isnan(eval_result[0]):
                        eval_result=np.zeros(len(eval_result))
                    func_results += [eval_result[rr]]
                    name += [signal_label + '_' + func_names[i] + '_' + str(rr)]

                #Low g sum, for total acceleration
                if func_total[i] == 'hist' and signal_label=='tot':

                    nbins = dictionary['statistical']['histogram']['free parameters']['nbins'][0]
                    factor= int(0.10*nbins)
                    low = func_results[:factor]
                    low_g = sum(low)
                    func_results+=[low_g]
                    name += [signal_label + '_' + 'low_g_sum']

            else:

                func_results += [eval_result]
                name += [signal_label + '_' + func_names[i]]


    #To Decoded function_names
    for a in range(len(name)):
        names+=[str(name[a])]

    #To sort names and func_results (to the same order)
    ordered=list(zip(names,func_results))
    ordered.sort()
    ordered=tuple(ordered)

    #To separate names and func_results in two different vectors
    nam=[]
    for q in range(len(ordered)):
        s = ordered[q]
        n= s[0]
        nam+=[s[0]]

    res=[]
    for q in range(len(ordered)):
        s = ordered[q]
        n= s[1]
        res+=[s[1]]

    return res, nam


def one_extract(feat_dict, signal_window, FS=100, iteration=None):
    """
    This function computes features matrix for one window.
    :param dictionary: (json file)
           list of features
    :param signal_window: (narray-like)
           input from which features are computed, window.
    :param signal_label: (narray-like)
           one of axes of acelerometer.
    # :param : (int)
    #        sampling frequency
    :return: res: (narray-like)
             values of each features for signal.
             nam: (narray-like)
             names of the features
    """

    #Create global arrays
    func_total = []
    func_names=[]
    imports_total=[]
    parameters_total=[]
    free_total=[]


    #Read Function (real name of function)
    function = feat_dict['function']
    #Read Imports
    imports = str(feat_dict['imports'])

    #Read Parameters
    parameters = feat_dict['parameters']

    #Read Free Parameters
    free_parameters = feat_dict['free parameters']

    #Execute imports
    exec(imports)

    nfuncs = len(func_total)
    func_results = []
    name=[]
    names=[]

    if iteration==None:

        execf = function + '(signal_window'

        if parameters != '':
            execf += ', '+ parameters

        if free_parameters != '' :
            for n, v in free_parameters.items():
                #TODO: conversion may loose precision (str)
                execf += ', ' + n + '=' + str(v[0])

        execf += ')'

    eval_result = eval( execf, locals())

    #Function returns more than one element
    if type(eval_result) == tuple:
        for rr in range(len(eval_result)):
            if np.isnan(eval_result[0]):
                eval_result=np.zeros(len(eval_result))
            func_results += [eval_result[rr]]

        #Low g sum, for total acceleration
        if func_total == 'hist' and signal_label=='tot':

            nbins = dictionary['statistical']['histogram']['free parameters']['nbins'][0]
            factor= int(0.10*nbins)
            low = func_results[:factor]
            low_g = sum(low)
            func_results+=[low_g]

    else:

        func_results += [eval_result]


    #To sort names and func_results (to the same order)

    return func_results
