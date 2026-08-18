#!/usr/bin/env python3

import email.utils
import subprocess
import sys
import time
from datetime import datetime, timezone

URL = "https://www.baidu.com/"


def get_http_time(insecure: bool) -> float:
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--head",
        "--location",
        "--max-time",
        "10",
    ]

    if insecure:
        command.append("--insecure")

    command.append(URL)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    # 重定向时可能出现多个 Date，取最后一个有效值。
    date_values = []
    for line in result.stdout.splitlines():
        if line.lower().startswith("date:"):
            date_values.append(line.split(":", 1)[1].strip())

    if not date_values:
        raise RuntimeError("百度响应中没有 Date 头")

    http_time = email.utils.parsedate_to_datetime(date_values[-1])

    if http_time.tzinfo is None:
        http_time = http_time.replace(tzinfo=timezone.utc)

    return http_time.timestamp()


def set_system_time(timestamp: float) -> None:
    time.clock_settime(time.CLOCK_REALTIME, timestamp)


def show_time(label: str, timestamp: float) -> None:
    utc_time = datetime.fromtimestamp(timestamp, timezone.utc)
    print(f"{label}: {utc_time.isoformat()}")


def main() -> int:
    try:
        # 第一次：系统时间可能错几年，TLS 证书可能因此校验失败。
        bootstrap_time = get_http_time(insecure=True)
        show_time("bootstrap UTC", bootstrap_time)
        set_system_time(bootstrap_time)

        # 第二次：时间已经大致恢复，启用正常的 TLS 证书验证。
        verified_time = get_http_time(insecure=False)
        show_time("verified UTC", verified_time)
        set_system_time(verified_time)

        print("system time synchronized from verified HTTPS Date")
        return 0

    except subprocess.CalledProcessError as error:
        print(f"curl failed: {error.stderr}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as error:
        print(f"time synchronization failed: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
