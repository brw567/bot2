import os
import time
import h5py
import logging
from logging.handlers import RotatingFileHandler
from config import HDF5_DIR

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')

def cleanup():
    """
    Remove HDF5 files older than 30 days from the data/hdf5 directory to manage storage.

    Uses h5py to validate files by opening them before deletion, ensuring only valid HDF5 files are removed.
    Logs deleted files and their sizes for traceability.

    Notes:
        - Retention period of 30 days balances data availability for backtesting/ML
          and storage efficiency. Adjustable via config if needed.
        - h5py validation prevents deletion of corrupted or non-HDF5 files.
    """
    try:
        if not os.path.exists(HDF5_DIR):
            logging.warning(f"HDF5 directory {HDF5_DIR} does not exist; skipping cleanup")
            return

        now = time.time()
        retention_seconds = 30 * 24 * 60 * 60  # 30 days in seconds
        total_deleted_size = 0

        for file in os.listdir(HDF5_DIR):
            if file.endswith('.h5'):
                file_path = os.path.join(HDF5_DIR, file)
                try:
                    # Validate HDF5 file
                    with h5py.File(file_path, "r") as _:
                        pass
                    file_mtime = os.stat(file_path).st_mtime
                    file_size = os.path.getsize(file_path)
                    if file_mtime < now - retention_seconds:
                        os.remove(file_path)
                        total_deleted_size += file_size
                        logging.info(f"Deleted HDF5 file: {file_path} ({file_size / 1024 / 1024:.2f} MB)")
                except (h5py.HDF5ExtError, OSError) as e:
                    logging.error(f"Failed to validate or delete {file_path}: {e}")
        if total_deleted_size > 0:
            logging.info(f"Total deleted size: {total_deleted_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        logging.error(f"HDF5 cleanup failed: {e}")

if __name__ == '__main__':
    cleanup()
