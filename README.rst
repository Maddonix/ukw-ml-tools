========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |requires|
        | |codecov|
        | |scrutinizer| |codacy| |codeclimate|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/ukw-ml-tools/badge/?style=flat
    :target: https://ukw-ml-tools.readthedocs.io/
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.com/Maddonix/ukw-ml-tools.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/github/Maddonix/ukw-ml-tools

.. |requires| image:: https://requires.io/github/Maddonix/ukw-ml-tools/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/Maddonix/ukw-ml-tools/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/Maddonix/ukw-ml-tools/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/Maddonix/ukw-ml-tools

.. |codacy| image:: https://img.shields.io/codacy/grade/684d451a718d4d519f6493e0deec97f9.svg
    :target: https://www.codacy.com/app/Maddonix/ukw-ml-tools
    :alt: Codacy Code Quality Status

.. |codeclimate| image:: https://codeclimate.com/github/Maddonix/ukw-ml-tools/badges/gpa.svg
   :target: https://codeclimate.com/github/Maddonix/ukw-ml-tools
   :alt: CodeClimate Quality Status

.. |version| image:: https://img.shields.io/pypi/v/ukw-ml-tools.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/ukw-ml-tools

.. |wheel| image:: https://img.shields.io/pypi/wheel/ukw-ml-tools.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/ukw-ml-tools

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/ukw-ml-tools.svg
    :alt: Supported versions
    :target: https://pypi.org/project/ukw-ml-tools

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/ukw-ml-tools.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/ukw-ml-tools

.. |commits-since| image:: https://img.shields.io/github/commits-since/Maddonix/ukw-ml-tools/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/Maddonix/ukw-ml-tools/compare/v0.0.0...master


.. |scrutinizer| image:: https://img.shields.io/scrutinizer/quality/g/Maddonix/ukw-ml-tools/master.svg
    :alt: Scrutinizer Status
    :target: https://scrutinizer-ci.com/g/Maddonix/ukw-ml-tools/


.. end-badges

Frequently used ML utilities.

* Free software: MIT license

Installation
============

::

    pip install ukw-ml-tools

You can also install the in-development version with::

    pip install https://github.com/Maddonix/ukw-ml-tools/archive/master.zip


Documentation
=============


https://ukw-ml-tools.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
