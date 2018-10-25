#!/usr/bin/python
# -*- coding: utf-8 -*-

# Package imports
from hive_utilities.hai.learning_algorithm_comb import LearningAlgorithmComb
from hive_utilities.hai.comb_output_description import CombOutputDescription
from hive_utilities.hai.analytic_result import AnalyticResult
from hive_utilities.hive_data_buffer import HiveDataBuffer
from hive_utilities.hai.analytic_utilities import ValueVectorAggregator
from hive_utilities import static_uris as uris
import numpy as np
from datetime import timedelta
import scipy.stats as stats

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


class GaussianReferenceComb(LearningAlgorithmComb):
    """Data-driven analytics comb implementing the Hive Analytics Interface (HAI) for statistics
    based on univariate normal distributions with a reference series as mean to detect abnormal
    deviations from the reference.

    Attributes
    ----------
    model : UnivariateGaussian
        The gaussian model with known mean for the feature variable.
    aggregator : ValueVectorAggregator
        The real-time feature vector aggregator to get multivariate states for feature and target series.
    time_window : float
        The size of a historic moving time window (in days) from which the distributions are
        estimated.
    confidence_interval : tuple
        The lower and upper bound of the statistical confidence interval that is used to detect
        unlikely observations.
    history_buffer : HiveDataBuffer
        The data buffer gathering historic data within a certain time window.
    n_outputs : int
        The total number of outputs produced by this comb.
    target_var : str
        The target variable (i.e. observation URI) acting as reference for this comb.
    feature_var : str
        The feature variable (i.e. observation URI) which is compared against the reference (target_var).
    """

    def __init__(self, **kwargs):
        """Creates an analytic comb to estimate abnormal deviations of a feature observation from a reference
        observation.

        This comb implements learning analytics, which means that the results produced by this
        comb are usually based on historic data. In this case, the historic data is used to train
        a univariate Gaussian distribution with known mean for a feature variable. The known mean
        is composed of the time series given by the target variable (i.e. the reference). This comb
        implements multivariate statistics (2 dimensional), which renders the algorithm subject to time
        synchronization. Hence, the resulting analytic output of this comb may have gaps and interpolated
        timestamps.

        Analytic results published by this comb consist of a data likelihood for a pair of feature and target
        variable. The likelihood measures how well an observed deviation of the feature variable from the
        reference (target variable) is explained by a trained normal distribution at each time point. The
        likelihood is normalized to [0, 1] where 0 states that the deviation is very unlikely and 1 is very likely.
        This comb computes an additional confidence interval at a specified level to flag new data points as
        anomalies if they fall outside the interval.
        """
        super(GaussianReferenceComb, self).__init__(**kwargs)
        self.model = None
        self.aggregator = None
        self.time_window = None
        self.confidence_interval = None
        self.history_buffer = None
        self.n_outputs = None
        self.target_var = None
        self.feature_var = None


    ######################
    # HAI Implementation #
    ######################

    def _initialize(self, params, lst_features, lst_targets):
        """Sets up the historic buffer and confidence interval of this comb up and generates the
        output description for the likelihood of a deviation of the feature series from the target
        series.

        This method implements the LearningAlgorithmComb._initialize interface. The comb generates
        one output. The outputs consist of the likelihood for new data samples of the feature observation
        to deviate from the target observation (see `_analyze` for details). Furthermore, a statistical
        confidence interval for the standard normal distribution (i.e. zero mean and unit variance) is
        precomputed for the specified level.

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
            A list with one output description for the deviation likelihood between feature and target
            variable.
        """
        self.pyout('Initializing Gaussian Reference comb...')
        if len(lst_features) != 1:
            raise ValueError('Expected exactly one feature variable')
        if len(lst_targets) != 1:
            raise ValueError('Expected exactly one target variable')
        self.feature_var = lst_features[0]
        self.target_var = lst_targets[0]
        if self.feature_var == self.target_var:
            raise ValueError('Feature can not be the same as the target variable')
        # Retrieve parameters
        self.time_window = params['time_window']
        alpha = params['conf_level']
        # Compute confidence interval at confidence level alpha for standard normal distribution
        self.confidence_interval = stats.norm.interval(alpha/100., loc=0, scale=1)
        # Print basic settings
        self.pyout('Comb settings:')
        self.pyout('* Time window (days): {}'.format(self.time_window))
        self.pyout('* {}% Confidence interval: {}'.format(alpha, self.confidence_interval))
        # Initialize Gaussian models for each feature and compile output description
        lst_outputs = []
        self.pyout('Initializing Gaussian reference model and generating output description...')
        self.model = GaussianReference()
        self.aggregator = ValueVectorAggregator(lst_features+lst_targets)
        # Default output description setup for likelihood, mean and standard deviation
        comb_label = self.get_comb_label()
        feature_label = self.get_label_of_feature(self.feature_var)
        target_label = self.get_label_of_target(self.target_var)
        output_descr_lik = CombOutputDescription(
            label='{}: Likelihood'.format(comb_label),
            unit=uris.I_UNIT_PERCENTAGE,
            description='The normalized data likelihood for the deviation of {} from {} assuming Gaussian noise. '
                        'A likelihood of 100% states that the observed deviation is very likely. A value of 0% '
                        'indicates that the observed value is very unlikely. The likelihood is flagged as '
                        'anomaly, if the deviation falls outside the {}% confidence interval.'.format(
                feature_label, target_label, alpha),
            lst_associations=[self.feature_var, self.target_var])
        # English translations
        comb_label = self.get_comb_label(lang='en')
        feature_label = self.get_label_of_feature(self.feature_var, lang='en')
        target_label = self.get_label_of_target(self.target_var, lang='en')
        output_descr_lik.set_label('{}: Likelihood'.format(comb_label), lang='en')
        output_descr_lik.set_description('The normalized data likelihood for the deviation of {} from {} assuming '
                                         'Gaussian noise. A likelihood of 100% states that the observed deviation is'
                                         'very likely. A value of 0% indicates that the observed value is very '
                                         'unlikely. The likelihood is flagged as anomaly, if the deviation falls '
                                         'outside the {}% confidence interval.'.format(
            feature_label, target_label, alpha), lang='en')
        # German translations
        comb_label = self.get_comb_label(lang='de')
        feature_label = self.get_label_of_feature(self.feature_var, lang='de')
        target_label = self.get_label_of_target(self.target_var, lang='de')
        output_descr_lik.set_label('{}: Wahrscheinlichkeit'.format(comb_label), lang='de')
        output_descr_lik.set_description('Die normalisierte Wahrscheinlichkeit der Abweichung fÃ¼r {} von {} unter '
                                         'Annahme einer Normalverteilung. Eine Wahrscheinlichkeit von 100% zeigt an '
                                         'das die beobachtete Abweichung Ã¤uÃŸerst Wahrscheinlich ist. Eine '
                                         'Wahrscheinlichkeit von 0% entspricht einer sehr Unwahrscheinlichen '
                                         'Abweichung. Die Wahrscheinlichkeit wird als Anomalie markiert, falls '
                                         'die beobachtete Abweichung nicht im {}% Konfidenzintervall liegt.'.format(
            feature_label, target_label, alpha), lang='de')
        # Add final output descriptions to the list of outputs of this comb
        lst_outputs.append(output_descr_lik)
        # Initialize history buffer nad auxiliary attributes
        self.history_buffer = HiveDataBuffer(lst_features+lst_targets)
        self.n_outputs = len(lst_outputs)
        self.pyout('Comb ready.')
        return lst_outputs

    def _analyze(self, obs_name, value, ts, is_abnormal):
        """Analyzes the likelihood of the deviation of a new data point to the reference and updates the history
        and Gaussian model.

        This method implements the LearningAlgorithmComb._analyze interface. The comb computes the
        normalized likelihood for the deviation of the new data point in the feature observation from
        the target observation (reference) assuming Gaussian noise. Afterwards, if the given data point
        is not flagged as abnormal, the historic data buffer is updated (i.e. the time window is further
        filled or moved). In case that the new data point was added to the history window, the Gaussian model
        is retrained with the new data window.

        Parameters
        ----------
        obs_name : str
            The feature name to which the data point belongs to.
        value : bool, int or float
            The actual value of the data point.
        ts : datetime
            The timestamp of the data point.
        is_abnormal : bool or None
            `True` if the data point was flagged as abnormal/anomaly, `False` if the data point
            is stated to be normal and `None` if no statement about the abnormality of the data
            point was made. This comb does not update the history of the feature variable with
            abnormal values.

        Returns
        -------
        lst_res : list of {AnalyticResult, None}
            If a synchronized data point from the feature and target observation was aggregated with this update,
            the list will contain an analytic result for the normalized Likelihood for the deviation of the feature
            data point from the target data point. If the computed likelihood lies outside the pre-computed
            confidence intervall, the result is additionally flagged as anomaly.
        """
        self.pyout('Analyzing "{}": {} ({}), is abnormal: {}'.format(
            obs_name, value, ts, is_abnormal))
        # Pre-compile result list, find base output index of feature and retrieve its model
        lst_results = [None]*self.n_outputs
        # Handle update of target value
        # Retrieve observation model and compute likelihood of data sample
        ###
        # Determine result for likelihood output
        X, f_ts = self.aggregator.aggregate(obs_name, value, ts)
        if X is not None:
            x_f = X[0, 0]
            x_t = X[0, 1]
            lh = self.model.norm_likelihood(x_f, x_t)
            self.pyout('* Normalized likelihood: {} ({})'.format(lh, f_ts))
            if lh is not None:
                # Check if value is within the confidence interval
                std_x = self.model.standardize(x_f, x_t)
                is_abn = None
                if std_x < self.confidence_interval[0] or std_x > self.confidence_interval[1]:
                    is_abn = True
                self.pyout('* Standardized value {} outside {}: {}'.format(
                    std_x, self.confidence_interval, bool(is_abn)))
                # Compile analytic results
                lst_results[0] = AnalyticResult(lh*100, f_ts, is_abn)
        ###
        # Update history buffer and retrain observation model
        if self.time_window is not None and not is_abnormal:
            # Add data point to history buffer and re-frame buffer back to time window
            self.pyout('* Updating history to new time window...')
            try:
                self.history_buffer.add_data(obs_name, value, ts, is_abnormal)
            except ValueError:
                pass  # might be historic run and the data point is already added
            self.history_buffer.filter_recent_time_window_for(obs_name, days=self.time_window)
            start_dt, end_dt = self.history_buffer.get_datetime_range_from(obs_name)
            n_samples = self.history_buffer.num_samples_for(obs_name)
            self.pyout('* \t\tCurrent samples: {}'.format(n_samples))
            self.pyout('* \t\tFrom: {}'.format(start_dt))
            self.pyout('* \t\tTo  : {}'.format(end_dt))
            # Re-train observation model if vector is complete
            if X is not None:
                self.pyout('* Retraining reference model...')
                self._train_model()
        return lst_results

    def _train(self, data_buffer):
        """Trains the Gaussian model for the deviation between feature and target variable from the given
        historic data.

        This method implements the LearningAlgorithmComb._train interface.  For each feature
        variable, the historic buffer is filled with all samples within the most recent time window
        as specified by self.time_window, skipping all samples flagged to be abnormal. The historic data used to
        train the model is synchronized with a common 2-dimensional time vector (and therefore artificially
        interpolates values with modified time stamps).The model is only trained if the data set does not have
        empty series.

        Parameters
        ----------
        data_buffer : HiveDataBuffer
            The data buffer containing all available historic data points for the feature and target
            variable of this comb.
        """
        self.pyout('#####')
        self.pyout('Received training data with {} features:'.format(
            len(data_buffer.get_variables())))
        # Copy training data to history buffer and fit models
        self.history_buffer = data_buffer.clone()
        # Check if there are enough samples to train
        if self.history_buffer.exists_empty_series():
            self.pyout('* -> Not enough samples in training buffer, cancelling training.')
            return
        # Gather and check original data situation
        n_sample = self.history_buffer.num_samples_for(self.feature_var)
        start_dt, end_dt = self.history_buffer.get_datetime_range_from(self.feature_var)
        self.pyout('* \t\tAvailable time window with {} samples.'.format(n_sample))
        self.pyout('* \t\t\tFrom: {}'.format(start_dt))
        self.pyout('* \t\t\tTo  : {}'.format(end_dt))
        # Filter all data to the most recent time window and remove anomalies
        if self.time_window is not None:
            self.history_buffer.filter_recent_time_window(days=self.time_window)
        self.history_buffer.filter_anomalies()
        # Gather and check filtered data situation
        if self.history_buffer.exists_empty_series():
            self.pyout('* -> Not enough samples in training after filtering, cancelling training.')
            return
        start_dt, end_dt = self.history_buffer.get_datetime_range_from(self.feature_var)
        n_sample = self.history_buffer.num_samples_for(self.feature_var)
        self.pyout('* \t\tExtracted time window of {} day(s) with {} samples.'.format(
            self.time_window, n_sample))
        self.pyout('* \t\t\tFrom: {}'.format(start_dt))
        self.pyout('* \t\t\tTo  : {}'.format(end_dt))
        # Fit gaussian if enough samples for the empirical estimates are available
        self.pyout('* Training Gaussian Reference model...')
        if n_sample == 0:
            self.pyout('* -> Not enough feature samples, cancelling training.')
            return
        self._train_model()
        self.pyout('Training finished.')
        return

    def _train_model(self):
        self.pyout('* Training Gaussian Reference model between {} and {}.'.format(self.feature_var, self.target_var))
        # Extract synchronized matrix
        feature_idx = self.history_buffer.get_variables().index(self.feature_var)
        target_idx = self.history_buffer.get_variables().index(self.target_var)
        X, ts, _ = self.history_buffer.to_aligned_matrix()
        self.pyout('* \t\tTraining data shape X: {} '.format(X.shape))
        # Fit Gaussian Reference model
        self.model.fit(X[:, feature_idx], X[:, target_idx])
        self.pyout('* \t\t-> Done, standard deviation: {}'.format(self.model.std_dev))

    def _reset_comb(self):
        """Resets the internal comb state.

        This method resets all internal historic buffers and aggregators. If the comb is adaptive and trains
        with real-time updates in its sliding window fashion, the distribution model is also reset to its
        initial state. If the comb is not adaptive and the distribution model wastrained manually through the
        train models task, the trained model is left untouched and only the current state aggregators are
        cleared from data.
        """
        self.history_buffer.clear()
        self.aggregator.clear()
        if self.time_window is not None:
            self.model = GaussianReference()

    def _get_output_date_range_for(self, from_input_dt, to_input_dt):
        """Retrieves the desired output date range of the comb for a given input range.

        The comb produces estimates for the same timestamps given by inputs. This means that the timestamps for
        the produced outputs are in the same range as the timestamps of the input data.
        """
        return from_input_dt, to_input_dt

    def _get_input_date_range_for(self, from_output_dt, to_output_dt):
        """Retrieves the required input date range of the comb for a given output range.

        The required inputs for this comb depend on the mode it is activated. If the comb is not adaptive,
        the internal regression model can be directly applied to data and the required inputs to produce results
        in the given output stay in the same date range. However, if the comb is adaptive, the comb needs to fill
        up a recent history for a sliding window to retrain the internal regression model. In this case, the
        required input range dates back according to the size of the time window with which this comb is configured.
        """
        # If comb is adaptive, the required input date range needs to account for the time window
        if self.time_window is not None:
            if from_output_dt is None:
                return from_output_dt, to_output_dt
            return from_output_dt-timedelta(days=self.time_window), to_output_dt
        # Otherwise, the comb is already trained and does not need to fill up the time window first
        return from_output_dt, to_output_dt


class GaussianReference(object):
    """A general univariate Gaussian distribution model parameterized by a mean and standard
    deviation, assuming that the mean is known.

    Attributes
    ----------
    std_dev : float
        The standard deviation of this distribution (i.e. the square root of the variance).
    _std_norm_scale : float
        The normalization factor for the likelihood.
    """

    def __init__(self):
        """Creates a new Gaussian distribution model.

        The model has to be trained with data using `fit` before it can be used.
        """
        self.std_dev = 0.0
        self._std_norm_scale = stats.norm.pdf(0, loc=0, scale=1)

    def fit(self, data, mus):
        """Fits a univariate Gaussian distribution to data.

        This method uses a given series as mean and a likelihood estimate 
        for the standard deviation to estimate the Gaussian distribution
        parameters from the data.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples,)
            The training data points to estimate the model parameters from.
        mus : np.ndarray of shape (n_samples,)
            The reference series which is used as known mean for the model.
        """
        assert len(data) == len(mus)
        diff = data - mus
        self.std_dev = np.sqrt(1.0/len(data)*np.sum(diff*diff))

    def norm_likelihood(self, x, mu):
        """Computes the normalized likelihood of a data point under this distribution model.

        The normalized likelihood is obtained by normalizing the probability density function value
        for the given data point to [0, 1]. For Gaussian distributions, the scaling factor is
        the mode (i.e. the pdf value at the mean). The likelihood measures how likely a given
        data point was drawn from a model distribution. In this case 1 is most likely and 0 least
        likely.

        Parameters
        ----------
        x : float
            The data point to compute the data likelihood for.
        mu : float
            The reference value which is used as mean for the Gaussian model.

        Returns
        -------
            The likelihood in [0, 1]. A likelihood of 1 indicates high probability that the data
            point is explained by this distribution. A value of 0 states that the data point is
            not explained by this model.
        """
        # Compute normalized likelihood of observing the data point under the distribution
        return stats.norm.pdf(self.standardize(x, mu), loc=0, scale=1) / self._std_norm_scale

    def standardize(self, x, mu):
        """Projects a data point under this distribution model to the standard normal distribution.

        Standardization is achieved by subtracting the mean (in this case the mean is known
        and given as reference value) from the data point and dividing the
        shifted data point by the standard deviation. This procedure transform the data point to
        the standard normal distribution with zero mean and unit variance.

        Parameters
        ----------
        x : float
            The data point to standardize.
        mu : float
            The reference value which is used as mean for the Gaussian model.

        Returns
        -------
            The standardized data point in a normal distribution with zero mean and unit variance.
        """
        return (x - mu) / max(1e-99, self.std_dev)