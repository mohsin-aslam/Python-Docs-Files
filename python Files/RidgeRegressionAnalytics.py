#!/usr/bin/python
# -*- coding: utf-8 -*-

# Package imports
from hive_utilities.hive_mixin import HiveMixin
from hive_utilities.hai.learning_algorithm_comb import LearningAlgorithmComb
from hive_utilities.hai.comb_output_description import CombOutputDescription
from hive_utilities.hai.analytic_result import AnalyticResult
from hive_utilities.hive_data_buffer import HiveDataBuffer
from hive_utilities.hai.analytic_utilities import ValueVectorAggregator
from hive_utilities import static_uris as uris
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.exceptions import NotFittedError
from sklearn import base
from datetime import timedelta

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


class RidgeRegressionComb(LearningAlgorithmComb):
    """Data-driven analytics comb implementing the Hive Analytics Interface (HAI) to perform linear regression
    with regularization.

    Attributes
    ----------
    time_window : float
        The size of a historic sliding time window (in days) from which the regression is (re)-trained.
    alphas : lst of float or float
        The regularization parameter(s). If a list is given, the regularization parameters are cross-
        validated.
    lst_features : lst of str
        The list of features (i.e. observation URIs) for the regression model.
    target_var : str
        The name of the target variable (i.e. observation URI) for the regression model.
    n_outputs : int
        The number of outputs this comb generates.
    history_buffer : HiveDataBuffer
        The historic values gathered from training and/or from real-time updates.
    feature_aggregator : ValueVectorAggregator
        The aggregator to gather synchronized feature vectors from teal-time updates.
    target_aggregator : float
        The aggregator to gather a labeled synchronized feature vector from real-time updates.
    model : Ridge or RidgeCV
        The linear regression model from scikit-learn.
    is_adaptive : bool
        A boolean flag that indicates if this comb is adaptive (i.e. retrains from real-time updates).
    pub_feature_rel : bool
        A boolean flag that indicates if the feature relevance should be published by this comb.
    pub_r2 : bool
        A boolean flag that indicates if the coefficient of determination should be published by this comb.
    scaler : float
        The feature scaler used to transform the feature space for the regression model.
    r2 : float
        The current coefficient of determination of the regression model fitted to data.
    """

    def __init__(self, **kwargs):
        """Creates an analytic comb to predict a target variable from feature variables.

        This comb implements learning analytics, which means that the results produced by this
        comb are usually based on historic data. In this case, the historic data is used to train
        linear regression model with regularization (Ridge Regression) to predict a target variable
        from different feature/observation variables. This comb implements multivariate statistics,
        which renders the algorithm subject to time synchronization. Hence, resulting analytic outputs
        of this comb may have gaps and interpolated timestamps.

        Analytic results published by this comb consist mainly of a prediction of the target variable
        from an aggregated feature vector at a time point. If the corresponding parameter flags are
        set, this comb additionally outputs the normalized feature relevance on the regression mode
        for each feature variable and the R2 (coefficient of determination). This comb does not
        flag output values as anomalies.
        """
        super(RidgeRegressionComb, self).__init__(**kwargs)
        self.time_window = None
        self.alphas = None
        self.lst_features = None
        self.target_var = None
        self.n_outputs = None
        self.history_buffer = None
        self.feature_aggregator = None
        self.target_aggregator = None
        self.model = None
        self.is_adaptive = None
        self.pub_feature_rel = None
        self.pub_r2 = None
        # Feature space scaling parameters
        self.scaler = None
        self.r2 = None


    ######################
    # HAI Implementation #
    ######################

    def _initialize(self, params, lst_features, lst_targets):
        """Sets up the regression model, aggregators and historic buffer of this comb and generates all
        output descriptions.

        This method implements the LearningAlgorithmComb._initialize interface. The comb generates
        at least one output: the predicted value of the target variable from all other features at a
        time point. If the corresponding flags are set, the comb also generates outputs for the feature
        relevance and coefficient of determination (see `_analyze` for details).

        Parameters
        ----------
        params : dict
            The parameter values according to the parameter file associated with this comb.
        lst_features : list of str
            The list of feature/observation variable names to set up this comb for.
        lst_targets : list of str
            This parameter is here for compatibility reasons and not used.

        Returns
        -------
        lst_outputs : list of CombOutputDescription
            A list with one or more output descriptions. The first entry is the output description for the
            prediction of the target variable. The second may contain the coefficient of determination if
            the flag was set by the user. The remaining descriptions will contain the feature relevance on
            the model for each feature variable if the corresponding flag was set..
        """
        # Perform sanity checks
        self.pyout('Initializing Ridge Regression comb...')
        if len(lst_features) == 0:
            raise ValueError('Expected at least one feature variable')
        if len(lst_targets) != 1:
            raise ValueError('Expected exactly one target variable')
        # Retrieve parameters
        self.time_window = params['time_window']
        self.alphas = params['alphas']
        self.pub_feature_rel = params['feature_rel']
        self.pub_r2 = params['r2']
        self.lst_features = lst_features
        self.target_var = lst_targets[0]
        self.is_adaptive = self.time_window is not None
        use_cv = isinstance(self.alphas, list)
        cv = params['cv_folds']
        if self.target_var in self.lst_features:
            raise ValueError('Target variable can not also be a feature of this model')
        # Print basic settings
        self.pyout('Comb settings:')
        self.pyout('* Time window (days): {}'.format(self.time_window))
        self.pyout('* Alpha value(s): {}'.format(self.alphas))
        self.pyout('* Feature variables: {}'.format(self.lst_features))
        self.pyout('* Target variable: {}'.format(self.target_var))
        self.pyout('* Publish feature relevance vector: {}'.format(self.pub_feature_rel))
        self.pyout('* Cross-validation: {}'.format(use_cv))
        self.pyout('* Real-time adaptation: {}'.format(self.is_adaptive))
        # Initialize regression models for each target and compile output descriptions
        self.pyout('Initializing ridge regression model and generating output descriptions...')
        lst_outputs = []
        # Create scaler and regression model
        self.scaler = StandardScaler()
        self.r2 = 0.0
        if use_cv:
            self.model = RidgeCV(alphas=self.alphas, fit_intercept=True, cv=cv)
        else:
            self.model = Ridge(alpha=self.alphas, fit_intercept=True)
        # Create output description for prediction
        comb_label = self.get_comb_label()
        target_label = self.get_label_of_target(self.target_var)
        output_descr = CombOutputDescription(
            label='{}: {} Prediction'.format(comb_label, target_label),
            unit=self.get_unit_of_target(self.target_var),
            description='Continuous prediction of the target variable {} based on a linear Ridge '
                        'Regression model.'.format(target_label),
            lst_associations=[self.target_var])
        # English translation
        comb_label = self.get_comb_label(lang='en')
        target_label = self.get_label_of_target(self.target_var, lang='en')
        output_descr.set_label('{}: {} Prediction'.format(comb_label, target_label), lang='en')
        output_descr.set_description('Continuous prediction of the target variable {} based on a linear Ridge '
                               'Regression model.'.format(target_label), lang='en')
        # German translation
        comb_label = self.get_comb_label(lang='de')
        target_label = self.get_label_of_target(self.target_var, lang='de')
        output_descr.set_label('{}: {} Vorhersage'.format(comb_label, target_label), lang='de')
        output_descr.set_description('Stetige Vorhersage der Zielvariable {} mithilfe eines Ridge Regression '
                               'Modells'.format(target_label), lang='de')
        # Append final output description
        lst_outputs.append(output_descr)
        if self.pub_r2:
            # Create output description for R2
            comb_label = self.get_comb_label()
            target_label = self.get_label_of_target(self.target_var)
            output_descr = CombOutputDescription(
                label='{}: Coefficient of Determination'.format(comb_label),
                unit=uris.I_NO_UNIT,
                description='The proportion of variation in the training data explained by the '
                            'ridge regression model. The R^2 measures of how well the observations for '
                            '{} are replicated by the model in the training data. A score of 1.0 states a '
                            'perfect replication. A value of zero states a bad replication as the '
                            'model does not explain variations. If the value is negative, a '
                            'constant model yielding the mean is better than the trained model '
                            'and you may start worrying as something probably went wrong.'.format(target_label),
                lst_associations=[self.target_var])
            # English translation
            comb_label = self.get_comb_label(lang='en')
            target_label = self.get_label_of_target(self.target_var, lang='en')
            output_descr.set_label('{}: Coefficient of Determination'.format(comb_label), lang='en')
            output_descr.set_description('The proportion of variation in the training data explained by the '
                                         'ridge regression model. The R^2 measures of how well the observations '
                                         'for {} are replicated by the model in the training data. A score of '
                                         '1.0 states a perfect replication. A value of zero states a bad '
                                         'replication as the model does not explain variations. If the value '
                                         'is negative, a constant model yielding the mean is better than the '
                                         'trained model and you may start worrying as something probably went '
                                         'wrong.'.format(target_label), lang='en')
            # German translation
            comb_label = self.get_comb_label(lang='de')
            target_label = self.get_label_of_target(self.target_var, lang='de')
            output_descr.set_label('{}: BestimmtheitsmaÃŸ'.format(comb_label), lang='de')
            output_descr.set_description('Der Anteil der Variation in den Trainingsdaten, der durch das '
                                         'Regressionsmodell erklÃ¤rt wurde. R^2 misst wie gut die Vorhersagen '
                                         'fÃ¼r {} vom Modell in den Trainingsdaten reproduziert wurden. Ein Wert von '
                                         '1.0 gibt eine perfekte Replikation an. Ein Wert von Null gibt eine schlechte '
                                         'Replikation an, da das Modell keine Variationen erklÃ¤rt. Wenn der Wert '
                                         'negativ ist, ist ein konstantes Modell, das den Mittelwert ergibt, besser '
                                         'als das trainierte Modell.'.format(comb_label, target_label), lang='de')
            # Append final output description
            lst_outputs.append(output_descr)
        if self.pub_feature_rel:
            for feature_var in self.lst_features:
                # Create output description for the feature relevance
                comb_label = self.get_comb_label()
                feature_label = self.get_label_of_feature(feature_var)
                target_label = self.get_label_of_target(self.target_var)
                output_descr = CombOutputDescription(
                    label='{}: {} Relevance'.format(comb_label, feature_label),
                    unit=uris.I_UNIT_PERCENTAGE,
                    description='The feature relevance of {} according to the regression model. A value of '
                                '0% states that {} is not relevant for the regression model to predict'
                                '{}. A higher valuer indicates higher importance for the prediction. This value '
                                'is based on standardization of the feature space and normalized across all '
                                'feature variables of the regression model. Hence, its a measure of predictive '
                                'power (or feature importance) of {} for {} relative to the other'
                                'features (however, it does not make a statement about the '
                                'quality).'.format(
                        feature_label, feature_label, target_label, feature_label, target_label),
                    lst_associations=[feature_var, self.target_var])
                # English translation
                comb_label = self.get_comb_label(lang='en')
                feature_label = self.get_label_of_feature(feature_var, lang='en')
                target_label = self.get_label_of_target(self.target_var, lang='en')
                output_descr.set_label('{}: {} Relevance'.format(comb_label, feature_label), lang='en')
                output_descr.set_description('The feature relevance of {} according to the regression model. A '
                                             'value of 0% states that {} is not relevant for the regression model '
                                             'to predict {}. A higher valuer indicates higher importance for the '
                                             'prediction. This value is based on standardization of the feature '
                                             'space and normalized across all feature variables of the regression '
                                             'model. Hence, its a measure of predictive power (or feature importance) '
                                             'of {} for {} relative to the other features (however, it does not make '
                                             'a statement about the quality).'.format(
                    feature_label, feature_label, target_label, feature_label, target_label), lang='en')
                # German translation
                comb_label = self.get_comb_label(lang='de')
                feature_label = self.get_label_of_feature(feature_var, lang='de')
                target_label = self.get_label_of_target(self.target_var, lang='de')
                output_descr.set_label('{}: {} Relevanz'.format(comb_label, feature_label), lang='de')
                output_descr.set_description('Die Merkmalsrelevanz von {} gemÃ¤ÃŸ dem Regressionsmodell. Ein Wert von '
                                             '0% besagt, dass {} nicht relevant fÃ¼r das Regressionsmodell ist, um '
                                             '{} vorherzusagen. Ein hÃ¶herer Wert zeigt eine hÃ¶here Wichtigkeit fÃ¼r '
                                             'die Vorhersage an. Dieser Wert basiert auf der Standardisierung des '
                                             'Merkmalsraums und wird Ã¼ber alle Merkmalsvariablen des '
                                             'Regressionsmodells normalisiert. Daher ist es ein MaÃŸ fÃ¼r die '
                                             'Vorhersagekraft (oder Merkmalsbedeutung) von {} fÃ¼r {} relativ zu den '
                                             'anderen Merkmalen (es macht jedoch keine Aussage Ã¼ber die '
                                             'QualitÃ¤t).'.format(
                    feature_label, feature_label, target_label, feature_label, target_label), lang='de')
                # Append final output description
                lst_outputs.append(output_descr)
        # Initialize history buffer and auxiliary attributes
        self.history_buffer = HiveDataBuffer(self.lst_features + [self.target_var])
        self.feature_aggregator = ValueVectorAggregator(sorted(self.lst_features))
        self.target_aggregator = ValueVectorAggregator(self.lst_features + [self.target_var])
        self.n_outputs = len(lst_outputs)
        self.pyout('Ridge Regression comb ready.')
        return lst_outputs

    def _analyze(self, feature_name, value, ts, is_abnormal):
        """Analyzes a new data point for aggregation or prediction of the target variable
        and updates the internal history and model.

        This method implements the LearningAlgorithmComb._analyze interface. The comb mainly computes
        a prediction of the target variable with the regression model if a multivariate vector was aggregated
        with this update. Afterwards, if the given data point is not flagged as abnormal, the historic data buffer
        is updated (i.e. the time window is further filled or moved). In case that a labeled feature vector was
        aggregated with the current update, the history contains a new labeled data point and the model is
        retrained from the current history.

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
            point was made. This comb does not update the history of the feature variable with
            abnormal values.

        Returns
        -------
        lst_res : list of {AnalyticResult, None}
            The list will contain an analytic result for a prediction of the target variable at the
            corresponding position each time a complete feature vector was aggregated. If this is the
            case and this comb was instructed to also publish the current coefficient of determination
            and/or the normalized feature relevances (i.e. the standardized feature relevance divided
            by the sum of relevances of all features), at the same time the corresponding results will
            be also published along with the prediction. No result is marked as abnormal by this comb.
        """
        self.pyout('Analyzing "{}": {} ({}), is abnormal: {}'.format(
            feature_name, value, ts, is_abnormal), lvl=HiveMixin.DEBUG)
        # Pre-compile result list and update prediction buffer
        lst_results = [None]*self.n_outputs
        if feature_name in self.lst_features:
            self.pyout('* Adding value to current feature vector.', lvl=HiveMixin.DEBUG)
            X, f_ts = self.feature_aggregator.aggregate(feature_name, value, ts)
            # Generate prediction if feature vector is available
            if X is not None:
                # Predict new output from aggregated feature vector
                self.pyout('* Predicting outcome for target {}...'.format(self.target_var),lvl=HiveMixin.DEBUG)
                try:
                    X_pred = self.scaler.transform(X)
                    # Set prediction output
                    y_pred = self.model.predict(X_pred)[0]
                    lst_results[0] = AnalyticResult(y_pred, f_ts)
                    self.pyout('* \t\tPredicted regression outcome: {} (R2: {})'.format(y_pred, self.r2), lvl=HiveMixin.DEBUG)
                    # Set R2 output
                    if self.pub_r2:
                        lst_results[1] = AnalyticResult(self.r2, f_ts)
                    # Set feature relevance output
                    if self.pub_feature_rel:
                        f_rel = np.abs(self.model.coef_)
                        f_rel = f_rel / np.sum(f_rel)
                        start_idx = 2 if self.pub_r2 else 1
                        for f_idx, feature_var in enumerate(self.lst_features):
                            lst_results[start_idx + f_idx] = AnalyticResult(f_rel[f_idx]*100, f_ts)
                        self.pyout('* \t\tAdded associated feature relevance.',lvl=HiveMixin.DEBUG)
                except NotFittedError:
                    self.pyerr('* \t\tModel not fitted.', lvl=HiveMixin.DEBUG)
        ###
        # Update history buffer and adapt model
        trigger = None
        if self.is_adaptive:
            trigger, _ = self.target_aggregator.aggregate(feature_name, value, ts)
        if self.is_adaptive and not is_abnormal:
            try:
                self.history_buffer.add_data(feature_name, value, ts)
                self.pyout('* Added updated value to history buffer.', lvl=HiveMixin.DEBUG)
                # filter to new time window
                self.history_buffer.filter_recent_time_window(days=self.time_window)
                from_dt, to_dt = self.history_buffer.get_datetime_range_from(feature_name)
                self.pyout('* \t\tFiltered recent time window: {} samples from {} to {}'.format(
                    self.history_buffer.num_samples_for(feature_name), from_dt, to_dt), lvl=HiveMixin.DEBUG)
                # Retrain model on aggregating a complete synchronized vector with target variable
            except ValueError as e:
                self.pywarn('* Failed to add value to history:'.format(e), lvl=HiveMixin.DEBUG)
        if trigger is not None:
            self.pyout('Model retraining triggered...', lvl=HiveMixin.DEBUG)
            self._train_model(verbose_lvl=HiveMixin.DEBUG)
        self.pyout('Analytics finished.', lvl=HiveMixin.DEBUG)
        return lst_results

    def _train(self, data_buffer):
        """Trains a ridge regression model from the given historic data.

        This method implements the LearningAlgorithmComb._train interface. the historic buffer is filled with
        all samples within the most recent time window as specified by self.time_window, skipping all samples
        flagged to be abnormal. The historic data is transformed into an aligned (i.e. time synchronized)
        matrix where the data of the target variable is used as target vector to train the regression model.

        Parameters
        ----------
        data_buffer : HiveDataBuffer
            The data buffer containing all available historic data points for the feature variables
            of this comb.
        """
        self.pyout('#####')
        self.pyout('Received training data with {} variables.'.format(
            len(data_buffer.get_variables())))
        ###
        # Process training data to comb settings
        self.history_buffer = data_buffer.clone()
        for var_name in self.history_buffer.get_variables():
            # Print data information of variable
            from_dt, to_dt = self.history_buffer.get_datetime_range_from(var_name)
            n_samples = self.history_buffer.num_samples_for(var_name)
            self.pyout('* Data for {}: {} samples from {} to {}'.format(
                var_name, n_samples, from_dt, to_dt))
            # Check if enough samples are available to work with
            if n_samples == 0:
                if var_name in self.lst_features:
                    self.pywarn('* Not enough data for a feature variable! Returning.')
                    return
                else:
                    self.pywarn('* Not enough data for a target variable! Skipping target.')
                    continue
            # Filter to recent time window if necessary
            if self.time_window is not None:
                self.history_buffer.filter_recent_time_window(days=self.time_window)
                from_dt, to_dt = self.history_buffer.get_datetime_range_from(var_name)
                self.pyout('*\t\tFiltered recent time window: {} samples from {} to {}'.format(
                    self.history_buffer.num_samples_for(var_name), from_dt, to_dt))
            # Remove all anomalies
            self.history_buffer.filter_anomalies()
        ###
        # Train regression model from the features
        self._train_model()
        if not self.is_adaptive:
            self.history_buffer.clear()

    def _reset_comb(self):
        """Resets the internal comb state.

        This method resets all internal historic buffers and aggregators. If the comb is adaptive and trains
        with real-time updates in its sliding window fashion, the models are also reset to their initial state.
        If the comb is not adaptive and the regression model is trained manually through the train models task,
        the trained model is left untouched and only the current state aggregators are cleared from data.
        """
        self.history_buffer.clear()
        self.feature_aggregator.clear()
        self.target_aggregator.clear()
        if self.is_adaptive:
            self.model = base.clone(self.model)
            self.scaler = base.clone(self.scaler)

    def _get_output_date_range_for(self, from_input_dt, to_input_dt):
        """Retrieves the desired output date range of the comb for a given input range.

        The regression model in this comb produces estimates for the same timestamps given by inputs.
        This means that the timestamps for the produced outputs are in the same range as the timestamps of
        the input data.
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
        if self.is_adaptive:
            if from_output_dt is None:
                return from_output_dt, to_output_dt
            return from_output_dt-timedelta(days=self.time_window), to_output_dt
        # Otherwise, the comb is already trained and does not need to fill up the time window first
        return from_output_dt, to_output_dt

    def _train_model(self, verbose_lvl=HiveMixin.INFO):
        self.pyout('* Training ridge regression model for {}.'.format(self.target_var), lvl=verbose_lvl)
        # Extract synchronized feature matrix and target vector
        X, y = self._extract_features_and_targets(self.target_var)
        self.pyout('* \t\tTraining data shapes X / y: {} / {}'.format(X.shape, y.shape), lvl=verbose_lvl)
        # Scale data and fit model
        X = self.scaler.fit_transform(X)
        self.pyout('* \t\tStandardization mean / scale: {} / {}'.format(
            self.scaler.mean_, self.scaler.scale_), lvl=verbose_lvl)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.r2 = metrics.r2_score(y, y_pred)
        self.pyout('* -> Mean Squared Error: {}, R^2: {}'.format(metrics.mean_squared_error(
            y, y_pred), self.r2), lvl=verbose_lvl)

    def _extract_features_and_targets(self, target_var):
        training_buffer = self.history_buffer.extract_series(self.lst_features + [target_var])
        X, _, _ = training_buffer.to_aligned_matrix()
        target_idx = training_buffer.get_variables().index(target_var)
        y = X[:, target_idx]
        X = np.delete(X, target_idx, axis=1)
        return X, y