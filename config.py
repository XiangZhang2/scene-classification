# -*- coding:utf-8 -*-a
from multiprocessing import Process, Pool
import os, time

def run(fn):
    time.sleep(1)
    return fn * fn

num_list = [1,2,3,4,5,6]
# start_time = time.time()
# result = [run(a) for a in num_list]
# end_time = time.time()
# print result
# print end_time - start_time

start_time = time.time()
pool = Pool()
result = pool.map(run, num_list)
pool.close()
pool.join()
end_time = time.time()
print result
print end_time - start_time



























