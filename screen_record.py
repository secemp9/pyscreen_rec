# for the mouse/cursor part and DPI scaling support
from __future__ import annotations
import ctypes, win32gui, win32ui
import ctypes.wintypes
import enum
import itertools
import tkinter
from typing import Tuple, Any

# The different libraries it support. Will need to add an if block to prevent the needs of installing them all.
import pyautogui
from PIL import Image, ImageGrab
import mss

# This is for writing the image/frame to a file + manipulating it inside an array
import cv2
import numpy as np


class DPIAware(enum.Enum):
    """
    Code taken from:
    Credit:

    This is for handling DPI scaling, but that doesn't seem to work correctly :/
    """

    # https://learn.microsoft.com/windows/win32/hidpi/dpi-awareness-context
    SYSTEM_AWARE = ctypes.wintypes.HANDLE(-2)
    PER_MONITOR_AWARE = ctypes.wintypes.HANDLE(-3)
    PER_MONITOR_AWARE_V2 = ctypes.wintypes.HANDLE(-4)


class DPIUnaware(enum.Enum):
    # https://learn.microsoft.com/windows/win32/hidpi/dpi-awareness-context
    UNAWARE = ctypes.wintypes.HANDLE(-1)
    UNAWARE_GDISCALED = ctypes.wintypes.HANDLE(-5)


class ProcessDPIAwareness(enum.Enum):
    # https://learn.microsoft.com/windows/win32/api/shellscalingapi/ne-shellscalingapi-process_dpi_awareness
    UNAWARE = 0
    SYSTEM_AWARE = 1
    PER_MONITOR_AWARE = 2


class DeviceCapsIndex(enum.Enum):
    """Item to be returned by the GetDeviceCaps function."""

    # The values are defined as macros in C, so they are not exposed to the
    # windll API. They can alternatively be obtained from the pywin32 package:
    # pip install pywin32
    # Example:
    #   >>> import win32con
    #   >>> win32con.HORZSIZE
    #   ... 4
    # https://pypi.org/project/pywin32/
    # https://learn.microsoft.com/windows/win32/api/wingdi/nf-wingdi-getdevicecaps

    HORZSIZE = 4
    """Width, in millimeters, of the physical screen."""

    VERTSIZE = 6
    """Height, in millimeters, of the physical screen."""

    HORZRES = 8
    """Width, in pixels, of the screen."""

    VERTRES = 10
    """Height, in raster lines, of the screen."""

    ASPECTX = 40
    """Relative width of a device pixel used for line drawing."""

    ASPECTY = 42
    """Relative height of a device pixel used for line drawing."""

    ASPECTXY = 44
    """Diagonal width of the device pixel used for line drawing."""

    LOGPIXELSX = 88
    """Number of pixels per logical inch along the screen width. In a system
    with multiple display monitors, this value is the same for all monitors."""

    LOGPIXELSY = 90
    """Number of pixels per logical inch along the screen height. In a system
    with multiple display monitors, this value is the same for all monitors."""


class DPIManager:
    """DPI awareness manager."""

    def __init__(self,
                 aware_state: DPIAware = DPIAware.PER_MONITOR_AWARE_V2,
                 unaware_state: DPIUnaware = DPIUnaware.UNAWARE_GDISCALED):
        """Parameters:
        - `aware_state`: Default DPI-aware state. Possible values:
           - `DPIAware.SYSTEM_AWARE`
           - `DPIAware.PER_MONITOR_AWARE`
           - `DPIAware.PER_MONITOR_AWARE_V2`
        - `unaware_state`: Default DPI-unaware state. Possible values:
           - `DPIUnaware.UNAWARE`
           - `DPIUnaware.UNAWARE_GDISCALED`
        """
        ctx = ctypes.windll.user32.GetThreadDpiAwarenessContext()
        self._original_awareness = self.get_awareness(ctx)
        self.aware_state = aware_state
        self.unaware_state = unaware_state

    @staticmethod
    def set(awareness: DPIAware | DPIUnaware | ProcessDPIAwareness):
        """Set the DPI awareness state."""
        if awareness in itertools.chain(DPIAware, DPIUnaware):
            ctypes.windll.user32.SetThreadDpiAwarenessContext(awareness.value)
        elif awareness in ProcessDPIAwareness:
            # WARNING: This affects all threads in the current process and
            # can only be reversed thread-by-thread
            ctypes.windll.shcore.SetProcessDpiAwareness(awareness.value)
        else:
            raise ValueError(f'Invalid argument type {type(awareness)!r}')

    def toggle(self=None):
        """Toggle DPI awareness states."""
        if self.is_aware():
            self.set(self.unaware_state)
        else:
            self.set(self.aware_state)

    def restore(self=None):
        """Restore the original DPI awareness state."""
        self.set(self._original_awareness)

    def is_aware(self, awareness: DPIAware | DPIUnaware | None = None) -> bool:
        """Check if the state is DPI-aware."""
        if awareness is None:
            awareness = self.get_awareness()
        return awareness in DPIAware

    def get_awareness(self=None, _ctx: int | None = None) -> DPIAware | DPIUnaware:
        """Get the current DPIAwarenessContext parameter."""
        if _ctx is None:
            _ctx = ctypes.windll.user32.GetThreadDpiAwarenessContext()
        for ctx_type in itertools.chain(DPIAware, DPIUnaware):
            if ctypes.windll.user32.AreDpiAwarenessContextsEqual(
                    _ctx, ctx_type.value
            ):
                return ctx_type
        raise ValueError(f'Unknown DPI context type ({_ctx!r})')

    def get_dpi(self=None, _ctx: int | None = None) -> int:
        if _ctx is None:
            _ctx = ctypes.windll.user32.GetThreadDpiAwarenessContext()
        # This might fail and return 0 for PER_MONITOR_AWARE and
        # PER_MONITOR_AWARE_V2. This is because the DPI of a per-monitor-aware
        # window can change, and the actual DPI cannot be returned without the
        # window's HWND.
        if dpi := ctypes.windll.user32.GetDpiFromDpiAwarenessContext(_ctx):
            return dpi
        # Set up the DPI awareness context to be checked
        original_ctx = ctypes.windll.user32.GetThreadDpiAwarenessContext(_ctx)
        # Create a temporary window to probe the screen DPI
        temp_window = tkinter.Tk()
        dc = ctypes.windll.user32.GetDC(temp_window.winfo_id())
        dpi = ctypes.windll.gdi32.GetDeviceCaps(
            dc, DeviceCapsIndex.LOGPIXELSX.value
        )
        temp_window.destroy()
        # Restore the original DPI awareness context
        ctypes.windll.user32.GetThreadDpiAwarenessContext(original_ctx)
        return dpi

    def get_scale(self=None) -> float:
        """Get the scale factor of the monitor."""
        DEVICE_PRIMARY = 0
        # DEVICE_IMMERSIVE = 1
        # https://learn.microsoft.com/windows/win32/api/shellscalingapi/ne-shellscalingapi-display_device_type
        return ctypes.windll.shcore.GetScaleFactorForDevice(DEVICE_PRIMARY) / 100

    def fullscreen_size(self) -> Tuple[Any, Any]:
        """Get the scale factor of the monitor."""
        toggled = False
        if self.is_aware():
            pass
        else:
            self.toggle()  # needed to get real windows size
            toggled = True
        w, h = [ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)]
        if toggled:
            self.restore()
        return w, h
        # DEVICE_IMMERSIVE = 1
        # https://learn.microsoft.com/windows/win32/api/shellscalingapi/ne-shellscalingapi-display_device_type
        # return ctypes.windll.shcore.GetScaleFactorForDevice(DEVICE_PRIMARY) / 100


def get_cursor():
    """
    Taken from: https://stackoverflow.com/a/72471744/12349101
    Credit: crazycat256
    This gets the cursor style
    """

    hcursor = win32gui.GetCursorInfo()[1]
    hdc = win32ui.CreateDCFromHandle(win32gui.GetDC(0))
    hbmp = win32ui.CreateBitmap()
    hbmp.CreateCompatibleBitmap(hdc, 36, 36)
    hdc = hdc.CreateCompatibleDC()
    hdc.SelectObject(hbmp)
    hdc.DrawIcon((0, 0), hcursor)

    bmpinfo = hbmp.GetInfo()
    bmpstr = hbmp.GetBitmapBits(True)
    cursor = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1).convert(
        "RGBA")

    win32gui.DestroyIcon(hcursor)
    win32gui.DeleteObject(hbmp.GetHandle())
    hdc.DeleteDC()

    pixdata = cursor.load()

    width, height = cursor.size
    for y in range(height):
        for x in range(width):

            if pixdata[x, y] == (0, 0, 0, 255):
                pixdata[x, y] = (0, 0, 0, 0)

    hotspot = win32gui.GetIconInfo(hcursor)[1:3]

    return (cursor, hotspot)


def float_unzero(x):
    """
    Taken from: https://stackoverflow.com/a/2440786/12349101
    Credit: Alex Martelli
    param x: float with tailing zeros
    :return: int
    """
    return int(f'{x:.2f}'.rstrip('0').rstrip('.'))


def fullscreen(x):
    """
    :param x:
    :return:
    """
    w, h = x.fullscreen_size()
    return 0, 0, float_unzero(w), float_unzero(h)


def mss_shot(region=fullscreen):
    """
    Taken from:
    Credit:
    :param region: By default, use fullscreen size
    :return: image data
    This is for supporting the mss library
    """

    with mss.mss() as sct:
        # The screen part to capture
        region_keys = ('top', 'left', 'width', 'height')
        size = dict(zip(region_keys, region))

        # Grab the data
        img = sct.grab(size)
        return img


def pilshot(region=fullscreen(DPIManager())):
    """
    Taken from: https://stackoverflow.com/a/72471744/12349101
    Credit: crazycat256
    This is for supporting the Pillow library
    """
    img = ImageGrab.grab(bbox=region, include_layered_windows=True)
    pos_win = win32gui.GetCursorPos()
    pos = (round(pos_win[0] * ratio - hotspotx), round(pos_win[1] * ratio - hotspoty))
    img.paste(cursor, pos, cursor)
    return img


def stop_recording():
    global recordme
    recordme = False


def record_screen(file_format="mp4", region=None, filename="Recording.mp4", fps=14.0, display=False,
                  lib="pillow"):
    """
    Taken from:
    Credit:
    Description: Main function for recording the screen using different ways

    file_format: mp4 by default
    region: By default using your full screen size, inside a tuple
    filename: By default, it uses the filename Recording.mp4
    fps: 14.0 FPS, but I don't think those are the real fps given the visual speed...looks more like 30-60 to me.
    display: whether to display the output on another Window (by default using opencv)

    """

    if region is None:
        region = fullscreen(dpi_manager)

    if file_format == "avi":
        codec = cv2.VideoWriter_fourcc(*"XVID")
    else:
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    w, h = region[2], region[3]
    video_writer = cv2.VideoWriter(filename, codec, fps, (w, h))

    while recordme:
        if lib == "pillow":
            img = pilshot(region=region)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif lib == "pyautogui":
            img = pyautogui.screenshot(region=region)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif lib == "mss":
            img = mss_shot()  # doesn't support region or mouse yet, working on it...
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        video_writer.write(frame)

        # display recording when necessary
        if display:
            cv2.imshow('Recording', frame)
        # quit recording when pressing 'q' or esc
        if cv2.waitKey(1) in [ord('q'), 27]:
            break
    # release the Video writer
    video_writer.release()
    # destroy all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    dpi_manager = DPIManager()
    ratio = dpi_manager.get_scale()
    cursor, (hotspotx, hotspoty) = get_cursor()  # get the cursor style, and hotspot position
    cursor.save("cursor.png")  # save the cursor style to a file
    recordme = True  # turn on the recorder flag
    try:
        record_screen()
    except KeyboardInterrupt:
        recordme = False
