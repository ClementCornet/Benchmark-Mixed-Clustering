# -*- coding: utf-8 -*-
import logging

__author__ = "LN. Levy: <loup-noe.levy@energisme.com>, " \
             "AI Team: <ai@energisme.com>"

log = logging.getLogger(__name__)


class TemperatureConvertor:
    """ Static class to convert temperatures from/to Celsius, Fahrenheit or Kelvin degrees. """

    @staticmethod
    def celsius_to_fahrenheit(temp_celsius):
        """ Takes a temperature `temp` in Celsius and returns it in Fahrenheit

            :param temp_celsius: Temperature in Celsius degrees
            :type temp_celsius: float

            :return: Temperature in Fahrenheit degrees
            :rtype: float
        """
        return 9./5. * temp_celsius + 32.

    @classmethod
    def celsius_to_kelvin(cls, temp_celsius):
        """ Takes a temperature `temp` in Celsius and returns it in Kelvin

            :param temp_celsius: Temperature in Celsius degrees
            :type temp_celsius: float

            :return: Temperature in Kelvin degrees
            :rtype: float
        """
        return temp_celsius + 273.15

    @staticmethod
    def fahrenheit_to_celsius(temp_fahrenheit):
        """ Takes a temperature `temp` in Fahrenheit and returns it in Celsius

            :param temp_fahrenheit: Temperature in Fahrenheit degrees
            :type temp_fahrenheit: float

            :return: Temperature in Celsius degrees
            :rtype: float
        """
        return 5./9. * (temp_fahrenheit - 32.)

    @classmethod
    def fahrenheit_to_kelvin(cls, temp_fahrenheit):
        """ Takes a temperature `temp` in Fahrenheit and returns it in Kelvin

            :param temp_fahrenheit: Temperature in Fahrenheit degrees
            :type temp_fahrenheit: float

            :return: Temperature in Kelvin degrees
            :rtype: float
        """
        return cls.celsius_to_kelvin(cls.fahrenheit_to_celsius(temp_fahrenheit))

    @staticmethod
    def kelvin_to_celsius(temp_kelvin):
        """ Takes a temperature `temp` in Kelvin and returns it in Celsius

            :param temp_kelvin: Temperature in Kelvin degrees
            :type temp_kelvin: float

            :return: Temperature in Celsius degrees
            :rtype: float
        """
        return temp_kelvin - 273.15

    @classmethod
    def kelvin_to_fahrenheit(cls, temp_kelvin):
        """ Takes a temperature `temp` in Kelvin and returns it in Fahrenheit

            :param temp_kelvin: Temperature in Kelvin degrees
            :type temp_kelvin: float

            :return: Temperature in Fahrenheit degrees
            :rtype: float
        """
        return cls.celsius_to_fahrenheit(cls.kelvin_to_celsius(temp_kelvin))
