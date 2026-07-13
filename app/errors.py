class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        # 仅保存可安全返回给调用方的字段, 避免泄露原始输入或内部路径。
        self.code = code
        self.public_message = message
        self.status_code = status_code
