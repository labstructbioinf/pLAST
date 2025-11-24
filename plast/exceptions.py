"""
This module defines custom exceptions used in the PLAST (Plasmid Search Tool) system.
"""


class DataLoadingError(Exception):
    """
    Exception raised for errors in the data loading process.

    :param message: Error message describing the loading issue.
    :type message: str
    """
    def __init__(self, message: str = ""):
        super().__init__(message)


class NotFoundError(Exception):
    """
    Exception raised when a requested item is not found.

    :param message: Error message describing the missing item.
    :type message: str
    """
    def __init__(self, message: str = ""):
        super().__init__(message)


class BadToken(Exception):
    """
    Exception raised for invalid query tokens.

    :param message: Error message describing the bad token.
    :type message: str
    """
    def __init__(self, message: str = ""):
        super().__init__(message)
