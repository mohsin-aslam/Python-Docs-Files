#!/usr/bin/python
# -*- coding: utf-8 -*-

# Package imports
from hive_utilities.hai.expertise_algorithm_comb import ExpertiseAlgorithmComb
from hive_utilities.hai.comb_output_description import CombOutputDescription
from hive_utilities.hai.analytic_result import AnalyticResult
from hive_utilities import static_uris as uris

# Code author(s) that implemented the comb
__author__ = "Karl-Heinz Fiebig"
# Copyright for the comb implementation
__copyright__ = "Copyright 2018, idatase GmbH"
# Credit all people that contributed in some way
__credits__ = ["Karl-Heinz Fiebig"]
# State the license of the comb (in accordance with the idatase license)
__license__ = "All rights reserved"
# Version of the comb
__version__ = "0.0.1"
# Person that will maintain the comb
__maintainer__ = "Karl-Heinz Fiebig"
# Contact address regarding issues with the comb
__email__ = "karl-heinz.fiebig@idatase.de"
# Status as one of "Development", "Prototype", or "Production"
__status__ = "Prototype"


class ThresholdsComb(ExpertiseAlgorithmComb):
    """Expertise analytics comb for value thresholds on observations implementing the Hive
    Analytics Interface (HAI).

    Attributes
    ----------
    lst_observations : list of str
        A list of observations to which the threshold boundaries are applied.
    th_max : float
        The upper threshold for observation values.
    th_min : float
        The lower threshold for observation values.
    """
    
    def __init__(self, **kwargs):
        """Creates an analytic comb to impose threshold boundaries on observation values.

        This comb implements expertise analytics, which means that the results produced by this
        comb are usually based on domain knowledge of an expert. In this case, the domain knowledge
        is incorporated in form of known minimum and maximum values for observations. Whenever an
        observation value violates a threshold boundary, the comb yields an analytic result that is
        flagged as abnormal/anomaly. A common use case for a domain expert is to monitor
        observations when they exceed or fall below critical values.

        While this comb supports thresholds for multiple observations, all observations will get
        the same threshold boundaries. In order to use different boundaries for different
        observations, you will have to start multiple combs.
        """
        super(ThresholdsComb, self).__init__(**kwargs)
        self.lst_observations = None
        self.th_max = None
        self.th_min = None
        self.cache_file =[]
        self.valueList=[]


    ######################
    # HAI Implementation #
    ######################

    def _initialize(self, params, lst_features, lst_targets):
        """Sets up the lower and upper thresholds of this comb up and generates all output
        descriptions.

        This method implements the ExpertiseAlgorithmComb._initialize interface. The comb generates
        one output for each observation in the list of features. The output is the distance to the
        threshold boundaries (see `_analyze` for details).

        Parameters
        ----------
        params : dict
            The parameter values according to the parameter file associated with this comb.
        lst_features : list of str
            The list of feature/observation variable names to set up this comb for.
        lst_targets : None
            This parameter is here for compatibility reasons and not used.

        Returns
        -------
        lst_outputs : list of CombOutputDescription
            A list with one output descriptions for each features that was provided. The list
            contains the description of threshold distances.
        """
        # Setup parameters
        self.th_max = params['maximum']
        self.th_min = params['minimum']
        self.lst_observations = lst_features
        # Print basic settings
        self.pyout('Comb settings:')
        self.pyout('* Analyzed observations: {}'.format(self.lst_observations))
        self.pyout('* Minimum: {}'.format(self.th_min))
        self.pyout('* Maximum: {}'.format(self.th_max))
        lst_outputs = []
        for obs_uri in self.lst_observations:
            # Setup output description
            obs_label = self.get_label_of_feature(obs_uri)
            output_descr = CombOutputDescription(
                label='{}: Distance'.format(self.get_comb_label()),
                unit=self.get_unit_of_feature(obs_uri),
                description='A value of zero states that {} value lies within the upper '
                            'and lower limit boundaries. A positive value states that {} exceeds the upper'
                            'limit. A negative value states that {} deceeds the lower limit. Values that'
                            'violate the boundaries are flagged'
                            'as anomalies.'.format(obs_label, obs_label, obs_label),
                lst_associations=[obs_uri]
            )
            # English translation
            obs_label = self.get_label_of_feature(obs_uri, lang='en')
            output_descr.set_label('{}: Distance'.format(self.get_comb_label(lang='en')), lang='en')
            output_descr.set_description('A value of zero states that {} value lies within the upper '
                                        'and lower limit boundaries. A positive value states that {} exceeds the upper'
                                        'limit. A negative value states that {} deceeds the lower limit. Values that'
                                         'violate the boundaries are flagged'
                                         'as anomalies.'.format(obs_label, obs_label, obs_label), lang='en')
            # German translation
            obs_label = self.get_label_of_feature(obs_uri, lang='de')
            output_descr.set_label('{}: Abstand'.format(self.get_comb_label(lang='de')), lang='de')
            output_descr.set_description('Ein Wert von Null zeigt das {} innerhalb der definierten Schranken liegt.'
                                         'Ein positiver Wert zeigt eine Anomalie bei der {} die obere Schranke'
                                         'Ã¼berschritten hat. Ein negativer Wert zeigt eine Anomalie bei der {} die '
                                         'untere Schranke Ã¼berschritten hat. Schrankenverletzungen werden als '
                                         'Anomalie gekennzeichnet.'.format(obs_label, obs_label, obs_label), lang='de')
            # Add final output description to the list of outputs of this comb
            lst_outputs.append(output_descr)
        return lst_outputs

    def _analyze(self, feature_name, value, ts, is_abnormal):
        """Analyzes a new data point to lie within the threshold boundaries of this comb.

        This method implements the ExpertiseAlgorithmComb._analyze interface.

        Parameters
        ----------
        feature_name : str
            The feature name to which the data point belongs to.
        value : bool, int or float
            The actual value of the data point.
        ts : datetime
            The timestamp of the data point.
        is_abnormal : bool or None
            `True` if the data point was flagged as abnormal/anomaly, `False` if the data point
            is stated to be normal and `None` if no statement about the abnormality of the data
            point was made.
            Note: This comb does not use this parameter.

        Returns
        -------
        lst_res : list of {AnalyticResult, None}
            An analytic result for the corresponding output of the feature containing 0 and
            no anomaly flag if the value lies within the threshold interval. If the value violates
            the lower threshold, the result will contain a negative distance to the lower
            threshold boundary and mark the result as abnormal. If the value violates
            the upper threshold, the result will contain a positive distance to the upper
            threshold boundary and mark the result as abnormal.
        """
        self.pyout('Analyzing "{}": {} ({}), is abnormal: {}'.format(
            feature_name, value, ts, is_abnormal))
        # Init as normal state
        dist = 0
        is_abn = False
        self.valueList.Add(value)
        if (len(self.valueList) % 2 == 0 and len(self.valueList)!=0 ):
            dist = 77.0
        else:
            
            if value > self.th_max:
        #if (value > self.th_max and feature_name == 'New Thresholds: Distance') and (value > self.th_max and feature_name == 'New Thresholds: Distance') :
            # Value exceeds the upper threshold
                dist = value - self.th_max
                is_abn = True
            elif value < self.th_min:
            # Value deceeds the lower threshold
                dist = value - self.th_min
                is_abn = True
        # Return flagged result value
        output_idx = self.lst_observations.index(feature_name)
        lst_res = [None] * len(self.lst_observations)
        lst_res[output_idx] = AnalyticResult(dist, ts, is_abn)
        self.pyout('-> Distance result: {} ({}), is abnormal: {}'.format(dist, ts, is_abn))
        return lst_res

    def _reset_comb(self):
        """Resets the internal comb state.

        This comb does not keep track of an internal state
        """
        return

    def _get_output_date_range_for(self, from_input_dt, to_input_dt):
        """Retrieves the desired output date range of the comb for a given input range.

        The comb produces results for the same timestamps given by inputs. This means that the timestamps for
        the produced outputs are in the same range as the timestamps of the input data.
        """
        return from_input_dt, to_input_dt

    def _get_input_date_range_for(self, from_output_dt, to_output_dt):
        """Retrieves the required input date range of the comb for a given output range.


        The comb requires data with the same timestamps as desired for the given outputs. This means that the
        timestamps for the necessary inputs are in the same range as the timestamps of the output data.
        """
        return from_output_dt, to_output_dt