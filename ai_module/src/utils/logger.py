import os
import time
import sys
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
from collections import OrderedDict, deque
import threading
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.console import Group
from collections import OrderedDict
try:
    import rospy
except:
    pass


LOG_DIR = '/ws/external/log'
os.makedirs(LOG_DIR, exist_ok=True)


@dataclass
class LoggerConfig:
    quiet: bool = False
    prefix: Optional[str] = None
    log_path: Optional[str] = None
    no_intro: bool = False


def build_logger(logger=None, logger_cfg: Optional[LoggerConfig] = None):
    if logger:
        return logger
    elif logger_cfg:
        return RichLogger(**asdict(logger_cfg))
    else:
        return RichLogger()


class Logger:
    def __init__(self, quiet=False, prefix=None, log_path=None, no_intro=False):
        try:
            import rospy
            use_rospy = True
        except:
            use_rospy = False

        self.log_funcs = {
            'warn': rospy.logwarn if use_rospy else print,
            'error': rospy.logerr if use_rospy else print,
            'debug': rospy.logdebug if use_rospy else print,
            'info': rospy.loginfo if use_rospy else print,
        }

        if not no_intro:
            print(f"[{self.__class__.__name__}] {prefix if prefix else 'Logger'} is set to {'quiet' if quiet else 'verbose'} mode!")
        self.quiet = quiet
        self.prefix = prefix

        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            open(log_path, 'w').close()
        self.log_path = log_path

    def loginfo(self, msg, **kwargs):
        self.log(msg, 'info', **kwargs)

    def logwarn(self, msg, **kwargs):
        self.log(msg, 'warn', **kwargs)

    def logerr(self, msg, **kwargs):
        self.log(msg, 'error', **kwargs)
        # exit(0)  # TODO: Need to remove for safety

    def logrich(self, msg, **kwargs):
        self.log(msg, 'info', **kwargs)

    def log(self, msg, level=None, force=False, **kwargs):
        assert level in [None, 'info', 'warn', 'error', 'debug', 'progress']

        if self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as file:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"[{now}] {msg}\n")

        if (not self.quiet) or force:
            if self.prefix:
                msg = f"[{self.prefix}] {msg}"
            if level in self.log_funcs:
                log_fn = self.log_funcs[level]
                if log_fn.__name__ == 'print':
                    log_fn(msg)
                else:
                    log_fn(msg, **kwargs)
            else:
                print(msg)

    def start(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass



class StreamToLogger:
    def __init__(self, rich_logger, level="INFO"):
        self.rich_logger = rich_logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        # print()는 끝에 '\n' 붙이니까 라인 단위로 모아줌
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.rich_logger.stream(line, level=self.level)

    def flush(self):
        if self._buffer.strip():
            self.rich_logger.stream(self._buffer.strip(), level=self.level)
            self._buffer = ""


class RichLogger(Logger):
    def __init__(self, tee_to_console=True, stream_max_lines=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_funcs['rich'] = self.logrich

        # Save the original stream (ignore redirection)
        self._stdout = sys.__stdout__
        self._stderr = sys.__stderr__

        self.console = Console(file=self._stdout)
        self.live = None
        self.sticky_msgs = OrderedDict()
        self.stream_lines = deque(maxlen=stream_max_lines)
        self.sticky_reserved = 20
        self._lock = threading.Lock()
        self.tee_to_console = tee_to_console

    def _format_prefix(self, msg, style="bold yellow"):
        if isinstance(msg, Text):
            return msg
        s = str(msg)
        if ":" not in s:
            return Text(s)
        head, tail = s.split(":", 1)
        t = Text()
        t.append(head, style=style)  # ← 여기 스타일만 바꾸면 전역 테마 변경 가능
        t.append(":")
        if tail.startswith(" "):
            t.append(" ")
            tail = tail[1:]
        t.append(tail)
        return t

    def _render(self):
        # Sticky
        if self.sticky_msgs:
            sticky_group = Group(*self.sticky_msgs.values())
        else:
            sticky_group = "(no sticky messages)"
        sticky_panel = Panel(
            sticky_group, title="Status (Sticky)", border_style="cyan", padding=(0, 1)
        )

        # Stream (Recent N lines)
        term_width, term_height = self.console.size
        avail_height = max(5, term_height - self.sticky_reserved - 4)  # 4는 패널 border/타이틀 여유
        lines = list(self.stream_lines)[-avail_height:]  # 최근 avail_height 줄만
        stream_text = "\n".join(lines) if lines else "(no stream logs)"
        stream_panel = Panel(Text(stream_text), title="Logs (Stream)", border_style="magenta", padding=(0, 1))

        # Layout
        layout = Layout()
        layout.split(
            Layout(sticky_panel, name="sticky", size=self.sticky_reserved),  # 필요시 높이 조절
            Layout(stream_panel, name="stream")
        )
        return layout

    def _ensure_live(self):
        if self.live is None:
            self.live = Live(self._render(), console=self.console, refresh_per_second=self.sticky_reserved, screen=False)
            self.live.start()

    def _update_live_unlocked(self):
        self._ensure_live()
        self.live.update(self._render())

    def sticky(self, name: str, msg: str):
        """한 줄 고정 영역에 name으로 갱신"""
        with self._lock:
            renderable = self._format_prefix(msg)
            prev = self.sticky_msgs.get(name)
            if isinstance(prev, Text) and prev.plain == renderable.plain:
                return
            self.sticky_msgs[name] = renderable
            self._update_live_unlocked()
            if self.log_path:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[{now}] {msg}\n")

    def clear_sticky(self, name: str):
        with self._lock:
            if name in self.sticky_msgs:
                del self.sticky_msgs[name]
                self._update_live_unlocked()

    def stream(self, msg: str, level: str = "INFO"):
        with self._lock:
            self.stream_lines.append(msg)
            if self.tee_to_console:
                self.console.print(msg, highlight=False, soft_wrap=False)
            self._update_live_unlocked()

    def logrich(self, msg, name=None, level="INFO", **kwargs):
        if name:
            self.sticky(name, msg)
        else:
            self.stream(msg, level=level)

    def log(self, msg, level=None, force=False, **kwargs):
        assert level in [None, 'info', 'warn', 'error', 'debug', 'progress']

        if self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as file:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"[{now}] {msg}\n")

        if (not self.quiet) or force:
            if self.prefix:
                msg = f"[{self.prefix}] {msg}"
            name = kwargs.pop('name', None)
            if name:
                return self.sticky(name, f"❌ {msg}")
            self.stream(msg, level=level)

    def stop(self, name):
        with self._lock:
            if name in self.sticky_msgs:
                del self.sticky_msgs[name]
            if self.live:
                if self.sticky_msgs:
                    self.live.update(self._render())
                else:
                    self.live.stop()
                    self.live = None

    def stop_all(self):
        with self._lock:
            self.sticky_msgs.clear()
            self.stream_lines.clear()
            if self.live:
                self.live.stop()
                self.live = None


if __name__ == "__main__":
    logger = Logger(quiet=False, prefix='test', log_path=os.path.join(LOG_DIR, 'test.log'))
    logger.loginfo('test loginfo')
    logger.logwarn('test logwarn')
    logger.logerr('test logerr')
    logger.log('test print')

    rich_logger = RichLogger(quiet=False, prefix='test', log_path=os.path.join(LOG_DIR, 'test.log'))
    for step in range(20):
        rich_logger.logrich(f'status: step {step}', name='status')
        time.sleep(0.5)
