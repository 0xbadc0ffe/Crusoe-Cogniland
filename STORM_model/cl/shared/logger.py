import logging


class CustomFormatter(logging.Formatter):
    def format(self, record):
        original_name = record.name
        parts = original_name.split(".")
        if len(parts) > 1:
            record.name = ".".join(parts[1:])
        else:
            record.name = original_name
        result = super().format(record)
        record.name = original_name
        return result


def setup_logger(filename: str) -> logging.Logger:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = CustomFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    return logging.getLogger(filename)
