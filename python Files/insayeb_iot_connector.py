#!/usr/bin/python
# -*- coding: utf-8 -*-

# Package imports
from hive_utilities.hci.static_connector_comb import StaticConnectorComb
from hive_utilities.hci.stream_connector_comb import StreamConnectorComb
from hive_utilities.hci.query_result import QueryResult, QueryResultDescription
from hive_utilities.hci.publication_value import PublicationValue
from hive_utilities import static_variables as vars
from datetime import datetime
from urllib.parse import urlunparse
import time
import pytz
import requests
import json
import dateutil.parser

# Code author(s) that implemented the comb
__author__ = "Paul Englert"
# Copyright for the comb implementation
__copyright__ = "Copyright 2018, idatase"
# Credit all people that contributed in some way
__credits__ = ["Paul Englert"]
# State the license of the comb (in accordance with the idatase license)
__license__ = "GPL"
# Version of the comb
__version__ = "0.0.1"
# Person that will maintain the comb
__maintainer__ = "Paul Englert"
# Contact address regarding issues with the comb
__email__ = "paul.englert@idatase.com"
# Status as one of "Development", "Prototype", or "Production"
__status__ = "Prototype"
# Label to display for the comb
__label__ = "Insayeb IoT Connector"
# Description to display for the comb
__description__ = "Connector to the IoT Devices of Insayeb managed by an JSON API."


class InsayebIoTConnector(StreamConnectorComb, StaticConnectorComb):

    def __init__(self):
        super(InsayebIoTConnector, self).__init__()
        self._data_var_separator = ':'
        self._timestampkey = 'timestamp'
        self._valuekey = 'value'
        self._api = {
            'token': None,
            'host': None,
            'url': 'api/v1.0/sensor/{group}/data'
        }
        self._poll_interval = None
        self._obsmappings = []


    ######################
    # HCI Implementation #
    ######################

    def _initialize(self, params, obs_mapping):
        """
        Read up the Docstring in ConnectorHiveComb._initialize
        """
        # process parameters
        self._api['host'] = urlunparse(params.get('api_host'))
        if self._api['host'][-1] != '/':
            self._api['host'] + '/'
        self._api['token'] = params.get('api_token')
        self.pyout("Using API: {}".format(self._api))
        self._poll_interval = params.get('poll_interval')
        self.pyout("Poll Interval: {}".format(self._poll_interval))
        if self._poll_interval is None or self._poll_interval == 0:
            self.pyout("-> Polling disabled")

        self._obsmappings = obs_mapping
        for mp in self._obsmappings:
            if self._data_var_separator not in mp.data_var:
                raise ValueError(
                    "Invalid observation mapping data variable, expected a '{}', but didn't find it".format(self._data_var_separator))

    def _start_acquisition(self):
        """
        Read up the Docstring in ConnectorHiveComb._start_acquisition
        """
        # escape
        if self._poll_interval is None or self._poll_interval == 0:
            return
        # setup polling memory and base date
        polling_memory = {}
        base_date = datetime.now(pytz.utc)
        # endless loop
        while self.comb_state == vars.COMB_STATE_ACTIVE:
            # prepare poll
            lst_vals = []
            # load data for all mappings
            for mapping in self._obsmappings:
                last_read_date = polling_memory.get(mapping.obs_uri, base_date)
                res = self._query(mapping, from_date=last_read_date)
                for row in res.to_list():  # list is: [[timestamp, value, ...], ...]
                    # pub value: (obs_uri, value, timestamp)
                    lst_vals.append(PublicationValue(mapping.obs_uri, row[1], row[0]))
                    polling_memory[mapping.obs_uri] = row[0]  # safe as last polled date
            # publish
            self.publish(lst_vals)
            # sleep
            time.sleep(self._poll_interval)

    def _query(self, mapping, from_date=None, to_date=None, max_count=None,
               offset=None, ascendingly=True, describe=False):
        """
        Read up the Docstring in ConnectorHiveComb._query
        """
        group, key = mapping.data_var.split(self._data_var_separator)

        # build params
        params = 'describe={describe}&keys={key}'.format(describe=str(describe).lower(), key=key)
        if self._api['token'] is not None:
            params += '&apitoken={}'.format(self._api['token'])
        if from_date is not None:
            params += '&from-date={}'.format(from_date.isoformat())
        if to_date is not None:
            params += '&to-date={}'.format(to_date.isoformat())

        # build url
        url = '{}{}?{}'.format(self._api['host'], self._api['url'], params).format(group=group)

        r = requests.get(url)
        if r.status_code == 200:
            data = json.loads(r.content.decode('utf-8'))
            # return query result
            if not describe:
                # sort
                data = sorted(
                    data,
                    key=lambda k: k.get(self._timestampkey),
                    reverse=(not ascendingly))
                # apply constraints
                if offset is not None:
                    data = data[offset:]
                if max_count is not None:
                    data = data[:max_count]
                # process to query result
                res = QueryResult(mapping.obs_uri)
                for r in data:
                    res.add_row(
                        dateutil.parser.parse(r.get(self._timestampkey)),
                        r.get(self._valuekey))
                # return
                return res
            # return query result description
            else:
                min = data.get('mindate', None)
                if min is not None:
                    min = dateutil.parser.parse(min)
                max = data.get('maxdate', None)
                if max is not None:
                    max = dateutil.parser.parse(max)
                return QueryResultDescription(mapping.obs_uri, min, max, data.get('count', 0))
        else:
            # empty/erronous response
            self.pyerr("Failed calling API: {}".format(url))
            if not describe:
                return QueryResult(mapping.obs_uri)
            return QueryResultDescription(mapping.obs_uri, None, None, 0)
