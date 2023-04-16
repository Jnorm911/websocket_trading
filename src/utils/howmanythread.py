import threading
import time


def simple_worker(thread_id):
    print(f"Thread {thread_id} started.")
    time.sleep(1)
    print(f"Thread {thread_id} finished.")


def test_thread_count():
    for num_threads in range(1, 101):  # Test from 1 to 100 threads
        print(f"Testing {num_threads} threads.")
        start_time = time.time()

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=simple_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        elapsed_time = time.time() - start_time
        print(f"Time taken for {num_threads} threads: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    test_thread_count()
