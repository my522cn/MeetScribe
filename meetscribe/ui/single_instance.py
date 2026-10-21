"""单实例守卫：同一时间只允许运行一个 MeetScribe。

做法（QLocalServer / QLocalSocket，Windows 下就是命名管道，不依赖任何第三方库）：
- 启动时先尝试连接已存在的实例：连得上 → 说明已经开着一个，
  发一句 "activate" 让它把窗口抬到最前，然后自己静默退出；
- 连不上 → 说明自己是第一个，先清掉上次崩溃可能残留的管道名，再开始监听；
- 已运行的实例收到 activate → 从托盘恢复 / 置顶 / 激活窗口。

注意：不要用“锁文件存在就退出”的写法——进程被强杀后锁文件会残留，
用户就再也打不开软件了；管道名可以在启动时安全清理。
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

# 管道名带版本号：以后换实现方式也不会和旧版本互锁
DEFAULT_KEY = "MeetScribe.SingleInstance.v1"


class SingleInstance(QObject):
    """acquire() 返回 True 表示自己是唯一实例；False 表示已有一个在跑。"""

    activated = Signal()  # 另一个实例试图启动（应把主窗口抬到最前）

    def __init__(self, key: str = DEFAULT_KEY, parent=None) -> None:
        super().__init__(parent)
        self.key = key
        self._server: QLocalServer | None = None
        self._conns: list[QLocalSocket] = []

    def acquire(self) -> bool:
        # 已经在监听：说明本对象就是那个唯一实例（重复调用要幂等返回 True）
        if self._server is not None and self._server.isListening():
            return True

        sock = QLocalSocket()
        sock.connectToServer(self.key)
        if sock.waitForConnected(400):
            # 已有实例：通知它露脸，然后自己退出
            sock.write(b"activate")
            sock.flush()
            sock.waitForBytesWritten(400)
            sock.disconnectFromServer()
            return False

        # 没有实例在监听。可能是上次崩溃残留的管道名，先清掉再监听
        QLocalServer.removeServer(self.key)
        self._server = QLocalServer(self)
        self._server.newConnection.connect(self._on_new_connection)
        if not self._server.listen(self.key):  # 权限/命名异常：宁可放行也不卡住用户
            return True
        return True

    def release(self) -> None:
        for c in self._conns:
            c.abort()
        self._conns.clear()
        if self._server is not None:
            self._server.close()
            self._server = None
        QLocalServer.removeServer(self.key)

    # ---------- 内部 ----------
    def _on_new_connection(self) -> None:
        if self._server is None:
            return
        got = False
        while self._server.hasPendingConnections():
            conn = self._server.nextPendingConnection()
            self._conns.append(conn)
            conn.readyRead.connect(lambda c=conn: self._drain(c))
            conn.disconnected.connect(lambda c=conn: self._cleanup(c))
            self._drain(conn)
            got = True
        if got:
            self.activated.emit()

    def _drain(self, conn: QLocalSocket) -> None:
        conn.waitForReadyRead(200)
        conn.readAll()  # 内容无意义，只是握个手

    def _cleanup(self, conn: QLocalSocket) -> None:
        if conn in self._conns:
            self._conns.remove(conn)
        conn.deleteLater()
