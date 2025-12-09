import os
from data_loading.loader import (
list_subjects,
process_all_subjects
)

from config.path import DATA_ROOT


def main():
    subjects = list_subjects(DATA_ROOT)
    process_all_subjects(subjects)

if __name__ == "__main__":
    main()