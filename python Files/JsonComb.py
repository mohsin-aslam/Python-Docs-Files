#!/usr/bin/python
# -*- coding: utf-8 -*-

# Package imports
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import urlunparse
from hive_utilities.hci.static_connector_comb import StaticConnectorComb
from hive_utilities.hci.stream_connector_comb import StreamConnectorComb
from hive_utilities.hci.publication_value import PublicationValue
from hive_utilities.hci.observation_mapping import ObservationMapping
from hive_utilities.hci.query_result import QueryResult, QueryResultDescription
from hive_utilities import static_variables as vars
import pytz
from uuid import uuid4
import tzlocal
import urllib.request, json 

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


class CSVConnectorComb(StreamConnectorComb, StaticConnectorComb):
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
        """Creates a CSV connector comb to transfer data into the Hive system.

        This comb is a file connector and intended to be run once in order to transfer a batch of
        observation data into the Hive system. After initialization and data acquisition, the comb
        shuts down again. However, the process may be repeated to transfer data from multiple CSV
        files.
        The initialization sets up a first connection to the CSV file and the mapping
        between physical observations and data columns. As soon as the data acquisition is
        triggered, the data values in the CSV are read and published to the hive system.
        This comb requires data variables in the observation mapping, but may omit time variables.
        However, data values are always published with a time stamp. The handling of timestamp in
        absence of time variable associations is described in detail in `_start_acquisition`.
        """
        super(CSVConnectorComb, self).__init__(**kwargs)
        self.csv_file = None
        self.lst_mappings = None
        self.csv_header = None
        self.mod_date = None
        self.delim = None
        self.column_wise = False
        self.local_tz = pytz.timezone(tzlocal.get_localzone().zone)
        self.current_date = datetime.now(self.local_tz)


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
        # Extract polling interval from parameters
        #self.polling_interval = params['polling_interval']
        #self.pyout('* Polling Interval: {}'.format(self.polling_interval))
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
        #df = pd.read_csv(self.csv_file, self.delim, nrows=0)
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
            if om.timestamp_var is not None and om.timestamp_var not in self.csv_header:
                raise ValueError('Timestamp variable {} not found in the CSV header'.format(
                    om.timestamp_var))
            self.pyout('* \tMapping for "{}" is valid:'.format(om.obs_uri))
            self.pyout('* \t -> Data associated with CSV column "{}"'.format(om.data_var))
            if om.timestamp_var is None:
                self.pyout('* \t -> Timestamp column not provided, timestamps will be generated')
            else:
                self.pyout('* \t -> Timestamp associated with CSV column "{}"'.format(
                    om.timestamp_var))
        # Everything is nice and swifty
        return

    def _start_acquisition(self):
        """Transfers all data from the CSV file to the Hive system.

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

        The comb will either stay active and poll the csv file for new data according to the
        polling parameter, or it will go to sleep right after loading the newest values once into
        the Hive.
        """
        while self.comb_state == vars.COMB_STATE_ACTIVE:
            # load recent csv data using current_date as base
            dataset = pd.DataFrame()
            # go through mappings and query all 'new' data
            for om in self._obs_mappings:
                series = pd.DataFrame(self._query(om, from_date=self.current_date).to_list(),
                                      columns=['time', 'value', '_'])
                series['obs_uri'] = om.obs_uri
                dataset = dataset.append(series, ignore_index=True)
            # compute and publish updates
            if dataset.shape[0] > 0:
                dataset.sort_values(by=['time', 'obs_uri'], inplace=True)
                # publish updates
                lst_updates = []
                for time, value, _, obs_uri in dataset.values.tolist():
                    lst_updates.append(PublicationValue(obs_uri, value, pd.to_datetime(time)))
                self.pyout('Collected {} new observation values from CSV file since {}'.format(len(lst_updates), self.current_date))
                self.publish(lst_updates)
            # reset clock
            self.current_date = datetime.now(self.local_tz)
            # exit if we're not supposed to be polling the CSV file
            #if self.polling_interval is None or self.polling_interval <= 0:
               # return

    def _query(self, mapping, from_date=None, to_date=None, max_count=None,
               offset=None, ascendingly=True, describe=False):
        """Queries the CSV files data.
        """
        # load the csv file
        #df = pd.read_csv(self.csv_file, self.delim).replace(r'^\s*$', np.nan, regex=True)
        df = pd.read_json(self.csv_file)
        df = df.pivot_table(values='value',index='timestamp',columns='key')
        df['timestamp'] = df.index
        # get the timestamp var to work with
        ts_var = self._get_ts_var(mapping)
        # compute the artificial timestamp if necessary
        if ts_var is None:
            ts_var ='artificial_timestamp_{}'.format(str(uuid4()))
            df[ts_var] = pd.DataFrame([self.mod_date + timedelta(milliseconds=i) for i in range(df.shape[0])])
        # scope down the data set to only the columns we need
        df = df[[ts_var, mapping.data_var]]
        # convert to datetime (utc)
        df[ts_var] = pd.to_datetime(df[ts_var], utc=True).dt.tz_localize(pytz.utc)
        # filter dates
        if from_date is not None:
            df = df.loc[df[ts_var] >= from_date]
        if to_date is not None:
            df = df.loc[df[ts_var] <= to_date]
        # apply sort
        df.sort_values(by=[ts_var], ascending=ascendingly, inplace=True)
        # cut result set
        if offset:
            df = df[offset:]
        if max_count:
            df = df[0:max_count]
        # parse response
        if describe:
            count = df.shape[0]
            extent = [df[ts_var].at[0]]
            if df[ts_var].at[count - 1] < extent[0]:
                extent = [df[ts_var].at[count - 1]] + extent
            else:
                extent.append(df[ts_var].at[count - 1])
            result = QueryResultDescription(mapping.obs_uri, extent[0], extent[1], count)
        else:
            result = QueryResult(mapping.obs_uri)
            result.from_list(df.values.tolist())
        return result

    def _get_ts_var(self, om):
        """Determine timestamp variable for this mapping.

        This function will fallback to any other timestamp variable found in the
        observation mappings before return None.

        Parameter
        ---------
        om : VariableMapping
            The variable mapping from which to get the timestamp variable

        Return
        ------
        ts_var : str
            The timestamp variable of the mapping.
        """
        # Use the assigned variable if available
        if om.timestamp_var is not None:
            return om.timestamp_var
        else:
            # Oh nose, try a timestamp variable assigned to any other mapping to stay synced
            for om2 in self.lst_mappings:
                if om2.timestamp_var is not None:
                    return om2.timestamp_var