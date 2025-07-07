from concurrent.futures.thread import ThreadPoolExecutor
from time import sleep
from unittest import TestCase
from threading import Thread, Lock

import numpy as np
import asyncio



lock = Lock()
results = []


def processor(x: float):
    sleep(3)
    with lock:
        results.append(x * 3.0 + 3.0)



class TestParallel(TestCase):
    def test_parallel(self):
        arr = [0, 1, 2, 3]

        with ThreadPoolExecutor(max_workers=4) as executor:
            for v in arr:
                executor.submit(processor, v)

        print(results)


# async def processor(x: float):
#     await asyncio.sleep(10 - x)
#     return x * 3.0 + 3.0


# async def main():
#     arr = [0, 1, 2, 3]
#     executor = ThreadPoolExecutor()
#
#     futures = []
#     for v in arr:
#         futures.append(asyncio.create_task(processor(v)))
#
#     for completed_task in asyncio.as_completed(futures):
#         result = await completed_task
#         print(f"Received result: {result}")
#
#
# if __name__ == "__main__":
#     asyncio.run(main())




