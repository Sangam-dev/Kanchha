from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine, Type, TypeVar
import logging


from core.events import BaseEvent, SystemError


