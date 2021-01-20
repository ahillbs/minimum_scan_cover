import os, sys, inspect, fnmatch, zipfile

def add_parent_dir_to_path():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    pp_dir = os.path.dirname(parentdir)
    sys.path.append(pp_dir)
    sys.path.append(parentdir)

def find_file_pattern(pattern, path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                yield os.path.join(root, name)

def extract_files(file):
    dirname = ""
    for split in file.split(".")[:-1]:
            dirname += split
    try:        
        os.mkdir(dirname)
        z = zipfile.ZipFile(file)
        z.extractall(dirname)        
    except OSError as e:
        if e.errno != 17:
            raise e # Only raise error if some other error than "File already exists" occurs    
    return dirname
    