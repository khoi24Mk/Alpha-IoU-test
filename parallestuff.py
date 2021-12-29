import concurrent.futures
import time

start  = time.perf_counter()


def func(sth,id):
    print(f'func {id} start sleeping in {sth} second(s)')
    time.sleep(sth)
    return (f"Func {id} done sleep")


with concurrent.futures.ThreadPoolExecutor() as executor:
    result = [executor.submit(func,1,i) for i in range(5)]

    for f in concurrent.futures.as_completed(result):
        print(f.result())


done = time.perf_counter()

print(f'{round(done - start,2)}')
