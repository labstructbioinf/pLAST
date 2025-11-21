"""
This module defines custom exceptions used in the PLAST (Plasmid Search Tool) system.
"""


class DataLoadingError(Exception):
    """Exception raised for errors in the data loading process."""


class NotFoundError(Exception):
    """Exception raised when a requested item is not found."""


class BadToken(Exception):
    """Exception raised for invalid query tokens."""
