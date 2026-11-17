"""Future-clock runner plugin. Import pandas before replacing datetime classes."""
import pandas
import datetime as dt
import ipaddress
import os
from pathlib import Path
import socket

RealDatetime, RealDate = dt.datetime, dt.date
year = int(os.environ.get('REVIEW_YEAR','2027'))
instant = RealDatetime(year,7,15,16,tzinfo=dt.timezone.utc)

class ReviewDatetime(RealDatetime):
    @classmethod
    def now(cls,tz=None):
        value=instant.astimezone(tz) if tz else instant.replace(tzinfo=None)
        return cls.fromtimestamp(value.timestamp(),tz) if tz else cls(*value.timetuple()[:6])
    @classmethod
    def utcnow(cls):
        return cls(year,7,15,16)

class ReviewDate(RealDate):
    @classmethod
    def today(cls):
        return cls(year,7,15)

dt.datetime,dt.date=ReviewDatetime,ReviewDate
real_connect,real_connect_ex=socket.socket.connect,socket.socket.connect_ex

def allowed(address):
    return isinstance(address,tuple) and (str(address[0])=='localhost' or ipaddress.ip_address(address[0]).is_loopback)

def connect(sock,address):
    if not allowed(address): raise AssertionError('review: external network forbidden')
    return real_connect(sock,address)

def connect_ex(sock,address):
    if not allowed(address): raise AssertionError('review: external network forbidden')
    return real_connect_ex(sock,address)

socket.socket.connect,socket.socket.connect_ex=connect,connect_ex

def pytest_sessionstart(session):
    import bts.contest_state as cs
    expected=os.environ.get('REVIEW_SOURCE')
    if expected: assert str(Path(cs.__file__).resolve()).startswith(str(Path(expected).resolve()))
    print('review clock',instant.isoformat(),'source',cs.__file__)
