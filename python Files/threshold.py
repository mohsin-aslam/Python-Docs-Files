#!/usr/bin/python
# -*- coding: utf-8 -*-

# Package imports
from hive_utilities.hive_mixin import HiveMixin
from hive_utilities.hai.expertise_algorithm_comb import ExpertiseAlgorithmComb
from hive_utilities.hai.comb_output_description import CombOutputDescription
from hive_utilities.hai.analytic_result import AnalyticResult
from hive_utilities import static_uris as uris
import numpy as np
from hive_utilities.hive_data_buffer import HiveDataBuffer
from datetime import timedelta
from sklearn.exceptions import NotFittedError

# from hive_utilities.hai.analytic_utilities import ValueVectorAggregator

#  Code author(s) that implemented the comb
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
        self.time_window = None
        self.lst_features = None
        self.target_var = None
        #self.pub_mean = None
        self.mean = None
        self.n_outputs = None
        self.history_buffer = None
        self.cache_file = []
        # self.feature_aggregator = None
        # self.target_aggregator = None


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
        self.time_window = params['time_window']
        self.pub_mean = params['mean']
        self.lst_observations = lst_features
        # Print basic settings
        self.pyout('Comb settings:')
        self.pyout('* Analyzed observations: {}'.format(self.lst_observations))
        self.pyout('* Minimum: {}'.format(self.th_min))
        self.pyout('* Maximum: {}'.format(self.th_max))
        self.pyout('* Time window (days): {}'.format(self.time_window))
        self.pyout('* Feature variables: {}'.format(self.lst_features))
        self.pyout('* Target variable: {}'.format(self.target_var))
        #self.pyout('* Real-time adaptation: {}'.format(self.is_adaptive))
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
                                         # Append final output description
        lst_outputs.append(output_descr)
        # if self.pub_mean:
        #     # Create output description for mean
        #     obs_label = self.get_label_of_feature(obs_uri)
        #     output_descr = CombOutputDescription(
        #         label='{}: Mean'.format(self.get_comb_label()),
        #         unit=self.get_unit_of_feature(obs_uri),
        #         description='The proportion of variation in the training data explained by the '
        #                     'ridge regression model. The Mean measures of how well the observations for '
        #                     '{} are replicated by the model in the training data. A score of 1.0 states a '
        #                     'perfect replication. A value of zero states a bad replication as the '
        #                     'model does not explain variations. If the value is negative, a '
        #                     'constant model yielding the mean is better than the trained model '
        #                     'and you may start worrying as something probably went wrong.'.format(obs_label),
        #         lst_associations=[obs_uri])
        #     # English translation
        #     obs_label = self.get_label_of_feature(obs_uri, lang='en')
        #     output_descr.set_label('{}: Mean'.format(self.get_comb_label(lang='en')), lang='en')
        #     output_descr.set_description('The proportion of variation in the training data explained by the '
        #                                  'ridge regression model. The Mean of how well the observations '
        #                                  'for {} are replicated by the model in the training data. A score of '
        #                                  '1.0 states a perfect replication. A value of zero states a bad '
        #                                  'replication as the model does not explain variations. If the value '
        #                                  'is negative, a constant model yielding the mean is better than the '
        #                                  'trained model and you may start worrying as something probably went '
        #                                  'wrong.'.format(obs_label), lang='en')
        #     # Add final output description to the list of outputs of this comb
            #lst_outputs.append(output_descr)
            # self.feature_aggregator = ValueVectorAggregator(sorted(self.lst_features))
            # self.target_aggregator = ValueVectorAggregator(self.lst_features + [self.target_var])
        self.history_buffer = HiveDataBuffer(self.lst_features)
        self.n_outputs = len(lst_outputs)
        self.pyout('The comb is ready.')
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
        self.history_buffer.add_data(feature_name, value, ts)
        self.pyout('* Added updated value to history buffer.', lvl=HiveMixin.DEBUG)
                # filter to new time window
        self.history_buffer.filter_recent_time_window(days=self.time_window)
        from_dt, to_dt = self.history_buffer.get_datetime_range_from(feature_name)
        self.pyout('* \t\tFiltered recent time window: {} samples from {} to {}'.format(
            self.history_buffer.num_samples_for(feature_name), from_dt, to_dt), lvl=HiveMixin.DEBUG)
        self.pyout('Analyzing "{}": {} ({}), is abnormal: {}'.format(
            feature_name, value, ts, is_abnormal))
        # Init as normal state
        #self.history_buffer = data_buffer.clone()
        
        list1=np.array([[]], np.int32)
        list1 = self.history_buffer.get_data_from(self.history_buffer,feature_name)
        
        self.mean = np.mean(list1)
        dist = 0
        is_abn = False
        if value > self.th_max:
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
        lst_res[output_idx] = AnalyticResult(self.mean, ts, is_abn)
        self.pyout('-> Distance result: {} ({}), is abnormal: {}'.format(self.mean, ts, is_abn))
        #return lst_res

        #For analyzing other feature 
        # self.pyout('Analyzing "{}": {} ({}), is abnormal: {}'.format(
        #     feature_name, value, ts, is_abnormal), lvl=HiveMixin.DEBUG)
        # # Pre-compile result list and update prediction buffer
        # lst_results = [None]*self.n_outputs
        # # if feature_name in self.lst_features:
        #     self.pyout('* Adding value to current feature vector.', lvl=HiveMixin.DEBUG)
        #     X, f_ts = self.feature_aggregator.aggregate(feature_name, value, ts)
        #     if X is not None:
        #         # Predict new output from aggregated feature vector
        #         self.pyout('* Predicting outcome for target {}...'.format(self.target_var),lvl=HiveMixin.DEBUG)
        #         try:
        #             X_pred = self.scaler.transform(X)
        #             # Set prediction output
        #             y_pred = self.model.predict(X_pred)[0]
        #             lst_results[0] = AnalyticResult(y_pred, f_ts)
        #             self.pyout('* \t\tPredicted regression outcome: {} (Mean: {})'.format(y_pred, self.mean), lvl=HiveMixin.DEBUG)
        #             # Set mean output
                #     if self.pub_mean:
                #         lst_results[1] = AnalyticResult(self.mean, f_ts)
                #         except NotFittedError:
                #             self.pyerr('* \t\tModel not fitted.', lvl=HiveMixin.DEBUG)
                # # # Update history buffer and adapt model
        
        return lst_res
    


    # def _train(self, data_buffer):
    #     self.pyout('#####')
    #     self.pyout('Received training data with {} variables.'.format(
    #         len(data_buffer.get_variables())))
    #     ###
    #     # Process training data to comb settings
    #     self.history_buffer = data_buffer.clone()
    #     for var_name in self.history_buffer.get_variables():
    #         # Print data information of variable
    #         from_dt, to_dt = self.history_buffer.get_datetime_range_from(var_name)
    #         n_samples = self.history_buffer.num_samples_for(var_name)
    #         self.pyout('* Data for {}: {} samples from {} to {}'.format(
    #             var_name, n_samples, from_dt, to_dt))
    #         # # Check if enough samples are available to work with
    #         # if n_samples == 0:
    #         #     if var_name in self.lst_features:
    #         #         self.pywarn('* Not enough data for a feature variable! Returning.')
    #         #         return
    #         #     else:
    #         #         self.pywarn('* Not enough data for a target variable! Skipping target.')
    #         #         continue
    #         # Filter to recent time window if necessary
    #         if self.time_window is not None:
    #             self.history_buffer.filter_recent_time_window(days=self.time_window)
    #             from_dt, to_dt = self.history_buffer.get_datetime_range_from(var_name)
    #             self.pyout('*\t\tFiltered recent time window: {} samples from {} to {}'.format(
    #                 self.history_buffer.num_samples_for(var_name), from_dt, to_dt))
    #         # Remove all anomalies
    #         self.history_buffer.filter_anomalies()
    #     ###
    #     # Train regression model from the features
    #     self._train_model()
    #     if not self.is_adaptive:
    #         self.history_buffer.clear()
            
    def _reset_comb(self):
        self.history_buffer.clear()
        # self.feature_aggregator.clear()
        # self.target_aggregator.clear()
        return

    def _get_output_date_range_for(self, from_input_dt, to_input_dt):
        """Retrieves the desired output date range of the comb for a given input range.

        The comb produces results for the same timestamps given by inputs. This means that the timestamps for
        the produced outputs are in the same range as the timestamps of the input data.
        """
        return from_input_dt, to_input_dt

    def _get_input_date_range_for(self, from_output_dt, to_output_dt):
        if self.is_adaptive:
            if from_output_dt is None:
                return from_output_dt, to_output_dt
            return from_output_dt-timedelta(days=self.time_window), to_output_dt
        # Otherwise, the comb is already trained and does not need to fill up the time window first
        return from_output_dt, to_output_dt

    # def _train_model(self, verbose_lvl=HiveMixin.INFO):
    #     self.mean = 0.5
    #     #self.pyout('* -> Mean:{}'.format(0.5,self.mean), lvl=verbose_lvl)