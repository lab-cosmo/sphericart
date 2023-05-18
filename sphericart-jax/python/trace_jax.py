#@title Helper functions (execute this cell)
import functools
import traceback

_indentation = 0
def _trace(msg=None):
    """Print a message at current indentation."""
    if msg is not None:
        print("  " * _indentation + msg)

def _trace_indent(msg=None):
    """Print a message and then indent the rest."""
    global _indentation
    _trace(msg)
    _indentation = 1 + _indentation

def _trace_unindent(msg=None):
    """Unindent then print a message."""
    global _indentation
    _indentation = _indentation - 1
    _trace(msg)

def trace(name):
  """A decorator for functions to trace arguments and results."""

  def trace_func(func):  # pylint: disable=missing-docstring
    def pp(v):
        """Print certain values more succinctly"""
        vtype = str(type(v))
        if "jax._src.xla_bridge._JaxComputationBuilder" in vtype:
            return "<JaxComputationBuilder>"
        elif "jaxlib.xla_extension.XlaOp" in vtype:
            return "<XlaOp at 0x{:x}>".format(id(v))
        elif ("partial_eval.JaxprTracer" in vtype or
              "batching.BatchTracer" in vtype or
              "ad.JVPTracer" in vtype):
            return "Traced<{}>".format(v.aval)
        elif isinstance(v, tuple):
            return "({})".format(pp_values(v))
        else:
            return str(v)
    def pp_values(args):
        return ", ".join([pp(arg) for arg in args])

    @functools.wraps(func)
    def func_wrapper(*args):
      _trace_indent("call {}({})".format(name, pp_values(args)))
      res = func(*args)
      _trace_unindent("|<- {} = {}".format(name, pp(res)))
      return res

    return func_wrapper

  return trace_func

class expectNotImplementedError(object):
  """Context manager to check for NotImplementedError."""
  def __enter__(self): pass
  def __exit__(self, type, value, tb):
    global _indentation
    _indentation = 0
    if type is NotImplementedError:
      print("\nFound expected exception:")
      traceback.print_exc(limit=3)
      return True
    elif type is None:  # No exception
      assert False, "Expected NotImplementedError"
    else:
      return False