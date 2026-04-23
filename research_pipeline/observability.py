import functools, logging, time
log = logging.getLogger("pipeline")

def traced(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            t0 = time.monotonic()
            log.info("chain.start", extra={"chain": name})
            try:
                out = fn(*a, **kw)
                log.info("chain.ok", extra={
                    "chain": name,
                    "ms": int((time.monotonic() - t0) * 1000),
                    "out_type": type(out).__name__,
                    "out_len": len(out) if hasattr(out, "__len__") else None,
                })
                return out
            except Exception as e:
                log.exception("chain.fail", extra={"chain": name, "err": str(e)})
                raise
        return wrap
    return deco
