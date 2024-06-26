from datetime import datetime
from logging import basicConfig, StreamHandler, FileHandler, Formatter, getLogger
from logging import INFO, DEBUG, NOTSET

def get_logger(dir_path=None):
    # ストリームハンドラの設定
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(Formatter("%(message)s"))

    if dir_path is None:
        basicConfig(level=NOTSET, handlers=[stream_handler])
    else:
        # ファイルハンドラの設定
        file_handler = FileHandler(
            f"{dir_path}/log{datetime.now():%Y%m%d%H%M%S}.log"
        )
        file_handler.setLevel(DEBUG)
        file_handler.setFormatter(
            Formatter("%(asctime)s@ %(name)s [%(levelname)s] %(funcName)s: %(message)s")
        )

        # ルートロガーの設定
        basicConfig(level=NOTSET, handlers=[stream_handler, file_handler])
    return getLogger(__name__)
