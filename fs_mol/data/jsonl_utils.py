import gzip
import json
import concurrent.futures

def parse_line(line):
    return json.loads(line)

def parallel_process_jsonl(file_path, process_func=None, parallel=True):
    with gzip.open(file_path, "rt") as f:
        lines = f.readlines()
        
    if parallel == True:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            data = list(executor.map(parse_line, lines))
        if process_func:
            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                data = list(executor.map(process_func, data))
        return data

    else:
        before_process = list([parse_line(line) for line in lines])
        
        if process_func == None:
            return before_process

        return [process_func(sample) for sample in before_process]

