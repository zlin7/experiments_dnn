import tqdm
import numpy as np

class TaskPartitioner():
    def __init__(self):
        self.task_list = None

    def add_task(self, func, *args, **kwargs):
        if self.task_list is None:
            self.task_list = []
        else:
            assert isinstance(self.task_list, list), "Trying to add a task without key to a keyed TaskPartitioner"
        self.task_list.append((func, args, kwargs))


    def add_task_with_key(self, key, func, *args, **kwargs):
        if self.task_list is None:
            self.task_list = dict()
        else:
            assert isinstance(self.task_list, dict), "Trying to add a keyed task without key to a non-eyed TaskPartitioner"
        self.task_list[key] = (func, args, kwargs)

    def __len__(self):
        return len(self.task_list)

    def copy(self):
        import copy
        o = TaskPartitioner()
        o.task_list = copy.copy(self.task_list)
        return o

    def run(self, ith, shuffle=True, npartition=3, suppress_exception=False, cache_only=False, debug=False, process_kwarg=None):
        import tqdm
        n = len(self.task_list)
        keyed = isinstance(self.task_list, dict)
        if ith is None:
            ith, npartition = 0, 1
        if shuffle:
            np.random.seed(npartition)  # being lazy
            perm = np.random.permutation(len(self.task_list))
        else:
            perm= np.arange(n)
        if keyed:
            task_ids = [key for i, key in enumerate(self.task_list.keys()) if perm[i] % npartition == ith]
        else:
            task_ids = [perm[i] for i in range(n) if i % npartition == ith]
        res = {}
        for task_id in tqdm.tqdm(task_ids):
            func, arg, kwargs = self.task_list[task_id]
            if process_kwarg is not None: kwargs[process_kwarg] = ith
            if debug:
                print(func, arg, kwargs)
            try:
                res[task_id] = func(*arg, **kwargs)
                if cache_only: res[task_id] = True
            except Exception as err:
                if suppress_exception:
                    print(err)
                else:
                    raise err
        return res

    def run_multi_process(self, nprocesses=1, cache_only=True, process_kwarg=None):
        if nprocesses == 1: return self.run(None, shuffle=False, debug=False, process_kwarg=process_kwarg)
        if not cache_only: o2 = self.copy()
        from multiprocessing import Process
        ps = []
        for i in range(nprocesses):
            p = Process(target=self.run, args=(i,), kwargs={'npartition': nprocesses, 'suppress_exception': True, 'cache_only': True, 'process_kwarg': process_kwarg})
            p.start()
            ps.append(p)
        for i,p in enumerate(ps):
            p.join()
        if not cache_only:
            return o2.run(None, shuffle=False)