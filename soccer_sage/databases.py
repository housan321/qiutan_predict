#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities related to database use.

Created on Tue Apr  2 22:49:30 2019
@author: Juan Beleño
"""
from sqlalchemy import create_engine
import os


def get_mysql_engine(schema=None):
    '''Get a MySQL engine

    Args:
        database (str): The schema name

    Returns:
        (sqlalchemy.engine.Engine): The SQL engine
    '''
    # user = os.environ['MYSQL_USER']
    # password = os.environ['MYSQL_PASSWORD']
    # host = os.environ['MYSQL_HOST']
    # port = os.environ['MYSQL_PORT']
    # auth_plugin = 'mysql_native_password'

    user = 'root'
    password = '123456'
    host = 'localhost'
    port = '3306'
    # auth_plugin = 'mysql_native_password'


    if schema is None:
        # schema = os.environ['MYSQL_SCHEMA']
        schema = 'qiutan'

    # conn_string = 'mysql+mysqlconnector://{0}:{1}@{2}:{3}/{4}?auth_plugin={5}'
    conn_string = 'mysql+mysqlconnector://{0}:{1}@{2}:{3}/{4}?charset=utf8'
    conn_string = conn_string.format(user,
                                     password,
                                     host,
                                     port,
                                     schema)
                                     # auth_plugin)
    engine = create_engine(conn_string, encoding='utf-8')
    return engine
