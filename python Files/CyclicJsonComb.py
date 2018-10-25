#!/usr/bin/python
# -*- coding: utf-8 -*-

# Package imports
import os
import time
import pytz
import pandas as pd
import numpy as np
from urllib.parse import urlunparse
from datetime import datetime, timedelta
from hive_utilities.hci.static_connector_comb import StaticConnectorComb
from hive_utilities.hci.stream_connector_comb import StreamConnectorComb
from hive_utilities.hci.publication_value import PublicationValue
from hive_utilities.hci.observation_mapping import ObservationMapping
from hive_utilities import static_variables as vars
from hive_utilities.hci.query_result import QueryResult

# Code author(s) that implemented the comb
__author__ = "Karl-Heinz Fiebig"
# Copyright for the comb implementation
__copyright__ = "Copyright 2018, idatase GmbH"
# Credit all people that contributed in some way
__credits__ = ["Karl-Heinz Fiebig", "Paul Englert"]
# State the license of the comb (in accordance with the idatase license)
__license__ = "All rights reserved"
# Version of the comb
__version__ = "0.0.1"
# Person that will maintain the comb
__maintainer__ = "Karl-Heinz Fiebig"
# Contact address regarding issues with the comb
__email__ = "karl-heinz.fiebig@idatase.de"
# Status as one of "Development", "Prototype", or "Production"
__status__ = "Development"


class CSVCyclicConnectorComb(StreamConnectorComb, StaticConnectorComb):
    """Connector comb implementation for CSV files implementing the Hive Connector
    Interface (HCI).

    Attributes
    ----------
    csv_file : str
        The absolute or relative path to the current CSV file storing the data to transfer.
    lst_mappings : list of ObservationMapping
        The current mapping objects association observation URIs with the data and timestamp
        columns of the CSV file.
    csv_header : list of str
        The column names from the header of the current CSV file.
    mod_date : datetime
        The timestamp of the last modification date of the current CSV file. If the last
        modification date can not be retrieved, this attribute contains the current date.
    delim : str
        The delimiter of the current CSV file with which values are separated.
    """

    def __init__(self, **kwargs):
        """Creates a CSV cyclic connector comb to transfer data into the Hive system.

        This comb is a file connector and intended to be run for demos, as it will use a csv file as the basis
        to create a continous stream of data.
        The initialization sets up a first connection to the CSV file and the mapping
        between physical observations and data columns. As soon as the data acquisition is
        triggered, the data values in the CSV are read and published to the hive system by looping over them in 
        the configured interval, but setting the timestamp to by computing the sum of iterations by the real interval
        parameter and adding it to the start date of the comb.
        """
        super(CSVCyclicConnectorComb, self).__init__(**kwargs)
        self.csv_file = None
        self.lst_mappings = None
        self.csv_header = None
        self.mod_date = None
        self.delim = None
        self.real_interval = None
        self.pub_interval = None
        self.current_date = datetime.utcnow().replace(tzinfo=pytz.utc)
        self.sample_idx = -1
        self._cache = []



    ######################
    # HCI Implementation #
    ######################

    def _initialize(self, params, obs_mapping):
        """Sets up the comb up to transfer data from a specific CSV file.

        This method implements the HiveConnectorComb._initialize interface. The header of the
        associated CSV file is already read in order to obtain the column names. Furthermore, the
        modification date of the CSV file is retrieved or set to the current date if the OS query
        fails. The observation mappings are also checked to have matching associations to the
        column names of the CSV file. If a mismatch in the data or timestamp variable association
        to a data column exists, a ValueError is thrown.

        Parameters
        ----------
        params : dict
            The parameter values according to the parameter file associated with this comb.
        obs_mapping : dict
            The mapping from observation URIs to comb-specific data and timestamp variables. In
            this case, both variables contain the column names of the CSV file to which the
            corresponding observation are mapped. The timestamp variable may be `None`, as
            described above.
        """
        self.pyout('Initializing CSV connector:')
        self.lst_mappings = obs_mapping

        ###
        # Extract file path from parameters
        if isinstance(params['csv_file'], str):
            csv_uri = params['csv_file']
        else:
            csv_uri = urlunparse(params['csv_file'])
        if csv_uri[0:4] == "http":
            self.csv_file = csv_uri
        elif os.path.exists(os.path.abspath(csv_uri)):
            self.csv_file = os.path.abspath(csv_uri)
        else:
            raise ValueError("CSV file not found {}.".format(csv_uri))
        self.pyout('* CSV file: {}'.format(self.csv_file))
        ###
        # Extract real interval from parameters
        self.real_interval = int(params['real_interval'])
        self.pyout('* Real Interval: {}'.format(self.real_interval))
        ###
        # Extract pub interval from parameters
        self.pub_interval = int(params['publication_interval'])
        self.pyout('* Publication Interval: {}'.format(self.pub_interval))
        ###
        # Extract base timestamp if necessary from parameters
        if params.get('base_timestamp', None) is not None:
            self.current_date = params['base_timestamp']
        self.pyout('* Base Timestamp: {}'.format(self.current_date))
        ###
        # Extract file modification date from parameters
        try:
            # Try obtaining the datetime modification date from the OS
            self.mod_date = datetime.fromtimestamp(os.path.getmtime(self.csv_file))
            self.pyout('* CSV file modification date: {}'.format(self.mod_date))
        except Exception:
            # Well, something went wrong, we will use the current date instead
            self.mod_date = datetime.today()
            self.pyout('* CSV file modification date could not be retrieved, using current '
                       'date instead: {}'.format(self.mod_date))
        ###
        # Extract delimiter from parameters
        #self.delim = str(params['delimiter'])
        #self.pyout('* Delimiter: \'{}\''.format(self.delim))
        ###
        # Load header file of CSV with pandas
        df = pd.read_json(self.csv_file)
        df = df.pivot_table(values='value',index='timestamp',columns='key')
        df = df.reset_index()
        self.csv_header = df.columns.values.tolist()
        if self.csv_header is None:
            raise ValueError('Could not retrieve a CSV header from the file')
        self.pyout('* CSV column header: {}'.format(self.csv_header))
        ###
        # check if given data and timestamp variables have associations to existing CSV columns
        self.pyout('* Checking variable mappings to CSV file...')
        for om in self.lst_mappings:
            if om.data_var not in self.csv_header:
                raise ValueError('Data variable {} not found in the CSV header'.format(
                    om.data_var))
            self.pyout('* \tMapping for "{}" is valid:'.format(om.obs_uri))
            self.pyout('* \t -> Data associated with CSV column "{}"'.format(om.data_var))
        # Everything is nice and swifty
        return

    def _start_acquisition(self):
        """Transfers in a cyclic manner the CSV file contents into the Hive.

        This method implements the HiveConnectorComb._start_acquisition interface. It loads the
        data from the CSV file set though the comb initialization and publishes it to the Hive
        system. Loading occurs through the pandas library in which atomic data types are
        automatically inferred. Data series are published for an observation URI according to
        the data variable in the observation mapping. Timestamps for each value are parsed from the
        timestamp variable in the observation mappings. However, it is possible to have no timestamp
        variable assigned. If no timestamp variables is provided within an observation mapping, the
        next available variable from an arbitrary other observation is used. If no observation
        mappings have timestamp variables assigned, timestamps are generated based on the
        modification date of the CSV file. In this case, the modification date is increased
        iteratively by one millisecond for each data sample, keeping the data across columns
        synchronized (i.e. each data row gets the same timestamp).

        The comb returns (and therefore triggers a shut-down) after all data series in the CSV
        file is published. Hence, the comb will not stay active after its duties are fulfilled.
        """
        # store current start date
        self._cache.append([0 + self.sample_idx, self.current_date, None])
        # Load CSV data into a pandas data frame
        df = pd.read_json(self.csv_file)
        df = df.pivot_table(values='value',index='timestamp',columns='key')
        df = df.reset_index()
        while self.comb_state == vars.COMB_STATE_ACTIVE:
            # update timestamp
            self.current_date = self.current_date + timedelta(seconds=self.real_interval)
            # update current row index
            self.sample_idx += 1
            if self.sample_idx > df.shape[0] - 1:
                self.sample_idx = 0
            # gather for each mapping the next update
            lst_pub_vals = []
            for om in self.lst_mappings:
                val = self._get_value(df, om.data_var, self.sample_idx)
                lst_pub_vals.append(PublicationValue(om.obs_uri, val, self.current_date))
            # publish
            self.publish(lst_pub_vals)
            # sleep
            time.sleep(self.pub_interval)
        # store current stop date
        self._cache[-1][2] = self.current_date

    def _get_value(self, df, col, idx):
        return df[col][idx]

    def _query(self, mapping, from_date=None, to_date=None, max_count=None,
               offset=None, ascendingly=True, describe=False):
        """Queries the generated data.
        """
        dataset = self._get_series_for_date_range(mapping.data_var, from_date=from_date, to_date=to_date)
        # apply sort
        if ascendingly:
            dataset.sort(order='ts')  # sort ascendingly
        else:
            dataset[::-1].sort(order='ts')  # sort descendingly
        # cut result set
        if offset:
            dataset = dataset[offset:]
        if max_count:
            dataset = dataset[:max_count]
        # parse response
        result = QueryResult(mapping.obs_uri)
        for timestamp, value in dataset:
            result.add_row(timestamp, value)
        if describe:
            return result.to_description()
        return result

    def _get_series_for_date_range(self, csv_column, from_date=None, to_date=None):
        """Computes the series of a variable between two dates that has been published by this comb.

        Parameters
        ----------
        csv_column : str
            The column for which to compute the series.
        from_date : datetime|None
            The minimum datetime constraint, if left None, will be the first available datetime.
        to_date : datetime|None
            The maximum datetime constraint, if left None, will be the current time.
        """
        df = pd.read_json(self.csv_file)
        df = df.pivot_table(values='value',index='timestamp',columns='key')
        df = df.reset_index()
        dataset = []
        for sample_idx, run_range_min, run_range_max in self._cache:
            if run_range_max is None:
                run_range_max = self.current_date
            # ensure date range is respected
            if from_date is not None and (from_date - run_range_min).total_seconds() > 0:
                run_range_min = from_date
            if to_date is not None and (run_range_max - to_date).total_seconds() > 0:
                run_range_max = to_date
            if (run_range_min - run_range_max).total_seconds() >= 0:
                continue
            # compute all values between min&max by using df and add to dataset
            count = int((run_range_max - run_range_min).total_seconds() / self.real_interval)
            for i in range(1, count + 1):
                # mimic exactly the behavior of _start_acquisition
                sample_idx += 1
                if sample_idx >= df.shape[0]:
                    sample_idx = 0
                timestamp = run_range_min + (i * timedelta(seconds=self.real_interval))
                dataset.append((timestamp, self._get_value(df, csv_column, sample_idx)))
        # parse as numpy array and return
        return np.array(dataset, dtype=[('ts', object), ('v', float)])