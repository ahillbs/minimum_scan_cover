"""These utils can be used to get an StringIO objects that
can be written like a file but does not close at the end of a "with" statement.
Objects can be retrieved with the virtual path.

This was implemented to use in configargparser in mind.
To use it, call for example:
holder = StringIOHolder()
parser = configargparse.ArgumentParser(description="Parser for the instance evolver", config_file_open_func=holder)
...
parsed = parser.parse_args()
parser.write_config_file(parsed, ["virt_path"])
holder["virt_path"].getvalue() # Holds the config data 
"""
import io

class NotWithCloseStringIO(io.StringIO):
    """This class is just the normal StringIO with the exception of not closing the memory file on exit of a "with" statement
    """
    def __exit__(self, type, value, traceback):
        pass

class StringIOHolder():
    """Holds NotWithCloseStringIO objects and can be called to replace an "open" call and write to memory file.
    File content is then 
    """
    def __init__(self):
        self._string_ios = {}

    def __call__(self, virt_path, bla):
        self._string_ios[virt_path] = NotWithCloseStringIO()
        return self.string_ios[virt_path]

    def __get__(self, key):
        return self._string_ios[key]

    def close(self):
        for key in self._string_ios:
            self._string_ios[key].close()