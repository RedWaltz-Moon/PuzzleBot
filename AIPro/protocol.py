#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocol.py — UART 通信协议处理

协议格式（来自 通信.md）:
  帧格式: AA CMD(1B) payload BB

  READY帧 (MCU→上位机, 4B): AA 00 mode BB
    mode = 01 → A路（模板）
    mode = 02 → B路（随机白纸和扑克牌）
    mode = 03 → C路（扑克牌拼接）

  坐标指令帧 (上位机→MCU, 17B):
    AA 01 'X' X_grab(2B) 'Y' Y_grab(2B) 'X' X_place(2B) 'Y' Y_place(2B) Yaw(2B) BB
    坐标值均为 int16 大端，实际值 = 原始值 / 100.0

  完成确认帧 (MCU→上位机, 3B): AA 02 BB

通信流程:
  MCU 上电 → 发 AA 00 mode BB（READY）
  上位机   → 发 AA 01 ... BB（坐标指令）
  MCU 执行 → pick-and-place 动作序列
  MCU     → 发 AA 02 BB（完成确认）
  上位机   → 等待下一条指令
"""

import struct
import time
from typing import Optional, Tuple

try:
    import serial
except ImportError:
    serial = None


# =============================================================================
# 帧常量
# =============================================================================
FRAME_HEADER = 0xAA
FRAME_FOOTER = 0xBB

CMD_READY = 0x00       # MCU → 上位机: 就绪
CMD_COORD = 0x01       # 上位机 → MCU: 坐标指令
CMD_ACK   = 0x02       # MCU → 上位机: 完成确认

MODE_A = 0x01  # A路（模板）
MODE_B = 0x02  # B路（随机白纸和扑克牌）
MODE_C = 0x03  # C路（扑克牌拼接）

READY_FRAME_LEN    = 4   # AA 00 mode BB
COORD_FRAME_LEN    = 17  # AA 01 X X_grab Y Y_grab X X_place Y Y_place Yaw BB
ACK_FRAME_LEN      = 3   # AA 02 BB


# =============================================================================
# 帧解析
# =============================================================================

def parse_ready_frame(data: bytes) -> Optional[int]:
    """
    解析 READY 帧。
    返回 mode (1=A路, 2=B路, 3=C路)，解析失败返回 None。
    """
    if len(data) < READY_FRAME_LEN:
        return None
    if data[0] != FRAME_HEADER or data[3] != FRAME_FOOTER:
        return None
    if data[1] != CMD_READY:
        return None
    mode = data[2]
    if mode not in (MODE_A, MODE_B, MODE_C):
        return None  # 拒绝未定义模式
    return mode


def parse_ack_frame(data: bytes) -> bool:
    """
    解析 ACK 帧 (AA 02 BB)。
    返回 True 表示收到有效 ACK。
    """
    if len(data) < ACK_FRAME_LEN:
        return False
    return (data[0] == FRAME_HEADER and
            data[1] == CMD_ACK and
            data[2] == FRAME_FOOTER)


# =============================================================================
# 帧构建
# =============================================================================

def build_coord_frame(
    x_grab_mm: float,
    y_grab_mm: float,
    x_place_mm: float,
    y_place_mm: float,
    yaw_deg: float,
) -> bytes:
    """
    构建坐标指令帧 (17 字节)。

    坐标值 = 实际值 × 100，转为 int16 大端。

    Args:
        x_grab_mm:  吸取点 X 坐标 (mm)
        y_grab_mm:  吸取点 Y 坐标 (mm)
        x_place_mm: 放置点 X 坐标 (mm)
        y_place_mm: 放置点 Y 坐标 (mm)
        yaw_deg:    旋转角度 (度, +顺时针)

    Returns:
        17 字节的坐标指令帧
    """
    # 坐标值 × 100 取整，钳位到 int16 范围
    def _to_int16(val: float) -> int:
        raw = int(round(val * 100.0))
        return max(-32768, min(32767, raw))

    x_grab  = _to_int16(x_grab_mm)
    y_grab  = _to_int16(y_grab_mm)
    x_place = _to_int16(x_place_mm)
    y_place = _to_int16(y_place_mm)
    yaw     = _to_int16(yaw_deg)

    frame = struct.pack(
        '>BB Bh Bh Bh Bh h B',  # '>' = 大端, 'h' = int16
        FRAME_HEADER,           # Byte 0:  AA
        CMD_COORD,              # Byte 1:  01
        ord('X'), x_grab,       # Byte 2-4: 'X' + int16 (大端)
        ord('Y'), y_grab,       # Byte 5-7: 'Y' + int16
        ord('X'), x_place,      # Byte 8-10:'X' + int16
        ord('Y'), y_place,      # Byte 11-13:'Y' + int16
        yaw,                    # Byte 14-15: Yaw int16 (大端)
        FRAME_FOOTER,           # Byte 16: BB
    )
    # 格式: B B (B h) (B h) (B h) (B h) h B = 12元素
    # 字节: 1+1+1+2 + 1+2 + 1+2 + 1+2 + 2 + 1 = 17 ✓
    return frame


# =============================================================================
# UART 通信类
# =============================================================================

class ProtocolUART:
    """UART 协议通信封装。

    使用示例:
        uart = ProtocolUART(port='/dev/ttyS1', baudrate=115200)
        uart.open()

        # 等待 READY
        mode = uart.wait_ready(timeout=30.0)
        print(f"MCU 就绪，模式: {'A路' if mode == MODE_A else 'B路'}")

        # 发送坐标
        frame = build_coord_frame(100.5, 200.3, 150.0, 250.0, 45.0)
        uart.send_frame(frame)

        # 等待确认
        if uart.wait_ack(timeout=10.0):
            print("MCU 完成")
    """

    def __init__(
        self,
        port: str = '/dev/ttyS1',
        baudrate: int = 115200,
        timeout: float = 0.1,
    ):
        if serial is None:
            raise ImportError("请安装 pyserial: pip install pyserial")
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None

    def open(self) -> None:
        """打开串口。等待 ModemManager 释放后重试。"""
        for attempt in range(20):
            try:
                self._ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout,
                )
            except Exception:
                self._ser = None
            if self._ser is not None and self._ser.is_open:
                self._ser.reset_input_buffer()
                self._ser.reset_output_buffer()
                break
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
            if attempt < 19:
                time.sleep(0.5)
                continue
            raise IOError(f"无法打开串口 {self.port}")
        print(f"[UART] 串口 {self.port} 已打开, 波特率 {self.baudrate}")

    def close(self) -> None:
        """关闭串口。"""
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[UART] 串口已关闭")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # 字节级读写
    # ------------------------------------------------------------------

    def _read_byte(self) -> int:
        """读取一个字节，超时返回 -1。"""
        if self._ser is None:
            return -1
        try:
            b = self._ser.read(1)
            if len(b) == 1:
                return b[0]
            return -1
        except Exception:
            return -1

    def _read_exact(self, n: int) -> Optional[bytes]:
        """精确读取 n 字节，失败返回 None。"""
        if self._ser is None:
            return None
        data = self._ser.read(n)
        return data if len(data) == n else None

    def send_frame(self, frame: bytes) -> None:
        """发送一帧原始字节。"""
        if self._ser is None:
            raise RuntimeError("串口未打开")
        self._ser.write(frame)
        self._ser.flush()
        print(f"[UART] TX ({len(frame)}B): {frame.hex(' ').upper()}")

    # ------------------------------------------------------------------
    # 帧级协议
    # ------------------------------------------------------------------

    def wait_ready(self, timeout: float = 30.0) -> int:
        """
        阻塞等待 MCU 发送 READY 帧 (AA 00 mode BB)。

        Args:
            timeout: 超时时间（秒）

        Returns:
            mode (1=A路, 2=B路)

        Raises:
            TimeoutError: 超时未收到 READY
        """
        print(f"[UART] 等待 MCU READY 信号 (timeout={timeout}s)...")
        deadline = time.monotonic() + timeout

        buf = bytearray()
        last_print = 0
        while time.monotonic() < deadline:
            b = self._read_byte()
            if b < 0:
                time.sleep(0.01)
                continue

            # 打印收到的每个字节
            now = time.monotonic()
            if now - last_print >= 0.5 or b == FRAME_HEADER:
                print(f"  [RX] 0x{b:02X}", end="", flush=True)
                last_print = now
            else:
                print(f" 0x{b:02X}", end="", flush=True)

            buf.append(b)

            # 搜索帧头 AA
            while len(buf) > 0 and buf[0] != FRAME_HEADER:
                print(f"\n  [DROP] 丢弃 0x{buf.pop(0):02X}")
                last_print = 0

            # 收到足够字节时尝试解析
            if len(buf) >= READY_FRAME_LEN:
                mode = parse_ready_frame(bytes(buf[:READY_FRAME_LEN]))
                if mode is not None:
                    print(f"\n[UART] RX READY: mode={mode} ({'A路' if mode == MODE_A else 'B路' if mode == MODE_B else 'C路'})")
                    del buf[:READY_FRAME_LEN]
                    return mode
                else:
                    print(f"\n  [DROP] 无效READY帧，丢弃首字节 0x{buf.pop(0):02X}")
                    last_print = 0

        raise TimeoutError(f"等待 READY 信号超时 ({timeout}s)")

    def wait_ack(self, timeout: float = 15.0) -> bool:
        """
        阻塞等待 MCU 发送 ACK 帧 (AA 02 BB)。

        Args:
            timeout: 超时时间（秒）

        Returns:
            True 表示收到 ACK

        Raises:
            TimeoutError: 超时未收到 ACK
        """
        print(f"[UART] 等待 MCU ACK 信号 (timeout={timeout}s)...")
        deadline = time.monotonic() + timeout

        buf = bytearray()
        last_print = 0
        while time.monotonic() < deadline:
            b = self._read_byte()
            if b < 0:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            if now - last_print >= 0.5 or b == FRAME_HEADER:
                print(f"  [RX] 0x{b:02X}", end="", flush=True)
                last_print = now
            else:
                print(f" 0x{b:02X}", end="", flush=True)

            buf.append(b)

            while len(buf) > 0 and buf[0] != FRAME_HEADER:
                print(f"\n  [DROP] 丢弃 0x{buf.pop(0):02X}")
                last_print = 0

            if len(buf) >= ACK_FRAME_LEN:
                if parse_ack_frame(bytes(buf[:ACK_FRAME_LEN])):
                    print("\n[UART] RX ACK: MCU 完成确认")
                    del buf[:ACK_FRAME_LEN]
                    return True
                else:
                    print(f"\n  [DROP] 无效ACK帧，丢弃首字节 0x{buf.pop(0):02X}")
                    last_print = 0

        raise TimeoutError(f"等待 ACK 信号超时 ({timeout}s)")

    def send_coord_and_wait_ack(
        self,
        x_grab_mm: float,
        y_grab_mm: float,
        x_place_mm: float,
        y_place_mm: float,
        yaw_deg: float,
        ack_timeout: float = 15.0,
    ) -> None:
        """
        发送坐标指令并等待 ACK 确认。

        封装了 build_coord_frame → send_frame → wait_ack 的完整流程。
        """
        frame = build_coord_frame(x_grab_mm, y_grab_mm,
                                  x_place_mm, y_place_mm, yaw_deg)
        self.send_frame(frame)
        self.wait_ack(timeout=ack_timeout)


# =============================================================================
# 便捷函数
# =============================================================================

def format_coord_frame(frame: bytes) -> str:
    """将 17 字节坐标帧格式化为可读字符串（调试用）。"""
    if len(frame) != COORD_FRAME_LEN:
        return f"<无效长度 {len(frame)}>"

    header, cmd = frame[0], frame[1]
    x1_ch = chr(frame[2])
    x_grab = struct.unpack('>h', frame[3:5])[0]
    y1_ch = chr(frame[5])
    y_grab = struct.unpack('>h', frame[6:8])[0]
    x2_ch = chr(frame[8])
    x_place = struct.unpack('>h', frame[9:11])[0]
    y2_ch = chr(frame[11])
    y_place = struct.unpack('>h', frame[12:14])[0]
    yaw = struct.unpack('>h', frame[14:16])[0]
    footer = frame[16]

    return (
        f"COORD: grab=({x_grab/100:.1f},{y_grab/100:.1f})mm "
        f"place=({x_place/100:.1f},{y_place/100:.1f})mm "
        f"yaw={yaw/100:.1f}°"
    )


# =============================================================================
# 自测
# =============================================================================

if __name__ == '__main__':
    # 测试 build_coord_frame
    frame = build_coord_frame(123.45, -67.89, 200.00, 150.00, 45.0)
    print(f"帧 ({len(frame)}B): {frame.hex(' ').upper()}")
    print(format_coord_frame(frame))

    # 验证长度
    assert len(frame) == 17, f"帧长度错误: {len(frame)}"
    assert frame[0] == 0xAA and frame[16] == 0xBB, "帧头/尾错误"
    assert frame[1] == 0x01, "命令码错误"
    assert chr(frame[2]) == 'X' and chr(frame[5]) == 'Y', "坐标标记错误"
    print("[OK] 自测通过")
