"""A representation of a 2d Vector."""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Number
from typing import Any, List, Type

import numpy as np
from pyrr.objects.base import BaseMatrix33, BaseVector, NpProxy

from . import vector2aux as vector2


class Vector2(BaseVector):  # type:ignore
    """Represents a 2-dimensional Vector. The Vector2 class is based on the pyrr implementation of vectors."""

    _module = vector2
    _shape = (2,)

    x = NpProxy(0)  #: The X value of this Vector.
    y = NpProxy(1)  #: The Y value of this Vector.
    xy = NpProxy([0, 1])  #: The X,Y values of this Vector as a numpy.ndarray.

    ########################
    # Creation
    def __new__(cls, value: Any = None, w: float = 0.0, dtype: Any = None) -> Any:
        """
        Make a new Vector2.

        :param value: The value of the Vector2.
        :param w: unused rest.
        :param dtype: The data-type.
        :return: A new Vector2.
        """
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)

            # matrix33
            if obj.shape in (
                (
                    3,
                    3,
                )
            ) or isinstance(obj, BaseMatrix33):
                obj = vector2.create_from_matrix33_translation(obj, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Vector2, cls).__new__(cls, obj)

    ########################
    # Operators
    @staticmethod
    def __has_type(other: Any, types: List[Type]) -> bool:
        return any(issubclass(other, t) for t in types)

    @classmethod
    def __is_number(cls, other):
        return cls.__is_number_type(type(other))

    @classmethod
    @lru_cache
    def __is_number_type(cls, t: Type):
        return cls.__has_type(t, [Number, np.number])

    @classmethod
    def __is_vector(cls, other):
        return cls.__is_vector_type(type(other))

    @classmethod
    @lru_cache
    def __is_vector_type(cls, t: Type):
        return cls.__has_type(t, [cls, np.ndarray, list])

    def __add__(self, other: Any) -> Vector2:  # type:ignore
        """
        Add to the existing Vector2.

        :param other: The other Vector2.
        :return: The added Vector2.
        """
        if self.__is_number(other):
            return Vector2(super(Vector2, self).__add__(other))
        elif self.__is_vector(other):
            return Vector2(super(Vector2, self).__add__(other))
        else:
            self._unsupported_type("add", other)

    def __sub__(self, other: Any) -> Vector2:  # type:ignore
        """
        Subtract from the existing Vector2.

        :param other: The other Vector2.
        :return: The subtracted Vector2.
        """
        if self.__is_number(other):
            return Vector2(super(Vector2, self).__sub__(other))
        elif self.__is_vector(other):
            return Vector2(super(Vector2, self).__sub__(other))
        else:
            self._unsupported_type("subtract", other)

    def __mul__(self, other: Any) -> Vector2:  # type:ignore
        """
        Multiply the existing Vector2.

        :param other: The other Vector2.
        :return: the multiplied Vector2.
        """
        if self.__is_number(other):
            return Vector2(super(Vector2, self).__mul__(other))
        else:
            self._unsupported_type("multiply", other)

    def __xor__(self, other: Any) -> Any:
        """
        Calculate the cross-product.

        :param other: The other Vector2.
        :return: The cross-product.
        """
        if self.__is_vector(other):
            return self.cross(other)
        else:
            self._unsupported_type("XOR", other)

    def __or__(self, other: Any) -> Any:
        """
        Calculate the dot-product.

        :param other: The other Vector2.
        :return: The dot-product.
        """
        if self.__is_vector(other):
            return self.dot(other)
        else:
            self._unsupported_type("OR", other)

    def __ne__(self, other: Any) -> bool:  # type:ignore
        """
        Not equal to the existing Vector2.

        :param other: The other Vector2.
        :return: whether they are unequal.
        """
        if self.__is_vector(other):
            return bool(np.any(super(Vector2, self).__ne__(other)))
        else:
            self._unsupported_type("NE", other)

    def __eq__(self, other: Any) -> bool:  # type:ignore
        """
        Equal to the existing Vector2.

        :param other: The other Vector2.
        :return: whether they are equal.
        """
        if self.__is_vector(other):
            return bool(np.all(super(Vector2, self).__eq__(other)))
        else:
            self._unsupported_type("EQ", other)

    ########################
    # Methods and Properties
    @property
    def inverse(self) -> Vector2:
        """:return: the inverted Vector2."""
        return Vector2(-self)

    @property
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    @property
    def normalized(self) -> Vector2:
        return Vector2(self / self.length)
