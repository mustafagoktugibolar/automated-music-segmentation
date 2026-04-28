from workers.dataset_worker import DatasetWorker


def main():
    w = DatasetWorker()
    w.start()


if __name__ == "__main__":
    main()
