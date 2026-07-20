python3 rtc_sync.py

import fcntl
import struct
import time
from datetime import datetime, timezone

RTC_RD_TIME = 0x80247009
RTC_STRUCT = "9i"

with open("/dev/rtc0", "rb", buffering=0) as rtc:
    data = bytearray(struct.calcsize(RTC_STRUCT))
    fcntl.ioctl(rtc.fileno(), RTC_RD_TIME, data, True)

sec, minute, hour, day, month, year, *_ = struct.unpack(RTC_STRUCT, data)

rtc_time = datetime(
    year=year + 1900,
    month=month + 1,
    day=day,
    hour=hour,
    minute=minute,
    second=sec,
    tzinfo=timezone.utc,
)

time.clock_settime(time.CLOCK_REALTIME, rtc_time.timestamp())
